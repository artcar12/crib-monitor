# Crib Monitor — Design

Date: 2026-09-24
Status: Approved

## Purpose

Detect when the baby rolls onto his stomach in the crib and alert a parent's Android phone,
so he can be put back on his back before he is in distress. It is a redundant layer alongside an
Owlet sock, which only alarms once vitals are affected.

This is not a medical device. It can miss a roll (stream loss, model error, host down). The health
alerts below exist so that failures are loud, not silent.

### Success criteria

- A roll onto the stomach produces an emergency phone alert within about 60 seconds.
- Any condition that stops the monitor from watching (camera lost, both detectors down, host dead)
  produces an alert within about 2–3 minutes.
- On real frames from his crib, the combined detector catches every stomach frame in the labeled
  evaluation set, and the false-alarm rate is low enough to sleep through.
- Costs nothing outside the schedule or a manual session (no stream, no model calls).

### Context and constraints

- Camera: Tapo C210, RTSP enabled via a Camera Account.
- Crib: no blankets, no sleep sack.
- The baby is at this house every other weekend; naps happen at no fixed time.
- Host: an always-on Ubuntu server (CPU only) runs the service.
- Local model: an Alienware laptop (Ryzen 9, 64 GB RAM, RTX 3070 Ti 8 GB) serves a vision model
  on the LAN.
- Cloud model: accessed through OpenRouter.
- Both models are called through LiteLLM.
- Alerts: Pushover on Android.
- A false alarm is cheap: the parent checks the Tapo live feed from bed.

## Architecture

A single Python service runs under systemd on the Ubuntu server.

```
Tapo C210 --RTSP--> capture --> crop --> motion gate --> classify (local ‖ cloud) --> decision --> notifier
                                                                                         |
                                     health watchdog (camera, models, heartbeat) --------+
                                     control page (LAN) --> arming state
                                     storage (frames + JSONL log)
```

### Modules (`crib_monitor/`)

| Module | Responsibility | Depends on |
|---|---|---|
| `config.py` | Loads `config.toml` and secrets from `.env`, validates them into typed settings | — |
| `schedule.py` | Pure function: is time *t* inside a scheduled window? | config |
| `arming.py` | Arming state: scheduled, manual on/off, pause; which source is active and why | schedule, clock |
| `capture.py` | Long-lived `ffmpeg` subprocess reading RTSP at 1 fps; yields frames; reconnects with backoff | ffmpeg |
| `motion.py` | Frame-difference score inside the crib crop | OpenCV |
| `classifier.py` | `classify(image) -> Label \| Unavailable` through LiteLLM; one class, parameterized by model | LiteLLM |
| `decision.py` | Pure state machine: combined labels + time in, actions out | — |
| `notifier.py` | Pushover send, emergency receipts (poll ack, cancel) | httpx |
| `health.py` | Tracks last-good timestamps and raises or clears health conditions; healthchecks.io ping | clock |
| `storage.py` | Saves checked frames and JSONL records; retention cleanup | — |
| `web.py` | LAN control page, status and labeling UI | FastAPI |
| `evaluate.py` | CLI: runs model configs over the labeled set and reports metrics | classifier |
| `main.py` | Orchestration loop tying the above together | all |

`schedule.py` and `decision.py` are pure and take time as input. `arming.py` and `health.py` take an
injected clock. All four are unit-tested against a fake clock.

## Capture

- One persistent `ffmpeg -rtsp_transport tcp` process on `stream1`, emitting 1 fps JPEG/raw
  frames to stdout. It does not reconnect per frame: reconnects are slow, and the C210 allows only a
  few RTSP clients, which leaves room for the Tapo app.
- The capture process runs only while armed or paused.
- The crib region is a rectangle in `config.toml`. Frames are cropped to it, then resized so the
  longest side is 768 px before classification.
- If no frame arrives for 5 s, kill and restart `ffmpeg` with backoff (1, 2, 4 … capped at 15 s).

## Motion gate

- Grayscale plus Gaussian blur, then absolute difference against the previous frame. The score is
  the fraction of pixels whose difference exceeds a threshold.
- Classify a frame when either:
  - the score exceeds `motion.threshold` (default 0.02) and at least `min_check_interval_s`
    (default 10) has passed since the last check, or
  - `max_check_interval_s` (default 300) has passed since the last check.
- The decision engine can force faster checking (WATCH, CONFIRMING, PAUSED, NO_VIEW), which
  bypasses the motion requirement.
- An IR day/night switch registers as motion and causes one check. This is harmless.

## Classification

- The same prompt goes to both models. It describes the view (a top-down or angled crib camera,
  possibly grayscale IR) and asks for JSON `{"position": "back"|"stomach"|"side"|"unclear"|"not_visible"}`.
- Requests go through `litellm.completion` with a JSON-schema `response_format`, image as a
  base64 data URL.
  - Cloud: `model = "openrouter/<model>"`. OpenRouter provider routing is set to deny providers
    that store or train on requests.
  - Local: `model = "openai/<name>"` with `api_base` pointing at the laptop's OpenAI-compatible
    server (llama.cpp `llama-server` or Ollama). First candidate: Qwen3-VL-30B-A3B.
- Both models are called concurrently for each check.
- Timeouts: local 30 s, cloud 20 s.
- A timeout, transport error, non-JSON output or unknown label returns `Unavailable`. It never
  counts as `back`.
- The model names are config values. The model actually used is chosen by the evaluation
  (see Testing).

### Combining the two answers

Only models that returned a label are considered.

| Combined result | Rule |
|---|---|
| `STOMACH` | any model says `stomach` |
| `SIDE` | else any says `side` |
| `BACK` | else any says `back` |
| `NO_VIEW` | else (every available model said `unclear` or `not_visible`) |
| `FAILED` | no model available; health tracking only, no position state change |

## Decision logic

States: `MONITORING`, `WATCH`, `CONFIRMING`, `ALERTED`, `NO_VIEW`, plus `PAUSED` (driven by arming).

| From | Input | Action / next state |
|---|---|---|
| any watching state | `STOMACH` (first) | → `CONFIRMING`; force a new check ~10 s later |
| `CONFIRMING` | `STOMACH` | Emergency alert with the frame attached → `ALERTED` |
| `CONFIRMING` | `BACK` | Log false positive → `MONITORING` |
| `CONFIRMING` | `SIDE` / `NO_VIEW` / `FAILED` | Check again in ~10 s. After 3 non-`BACK` confirmations without `STOMACH`, → `WATCH` |
| `MONITORING` | `SIDE` | → `WATCH`: forced checks every 30 s |
| `WATCH` | 2 consecutive `BACK` | → `MONITORING` |
| `MONITORING` / `WATCH` | `NO_VIEW` | → `NO_VIEW`: forced checks every 20 s |
| `NO_VIEW` | 3 consecutive `NO_VIEW` spanning ≥ 60 s | High-priority alert "can't see him", sent once per episode |
| `NO_VIEW` | `BACK` / `SIDE` | → `MONITORING` / `WATCH` |
| `ALERTED` | Pushover receipt acknowledged | Suppress stomach alerts for 3 min; forced checks every 30 s |
| `ALERTED` | still `STOMACH` after the suppression window | New emergency alert |
| `ALERTED` | 2 consecutive `BACK` | Cancel any open receipt, send normal "back on his back" → `MONITORING` |

- Emergency alerts use Pushover priority 2 with `retry=60`, `expire=1800`. The receipt is polled
  every 15 s for acknowledgement.
- Worst-case latency from a roll to the phone is about 30–60 s: motion check ≤ 10 s, model
  ≤ 15 s, confirmation ~10–20 s.

## Arming

Sources, in priority order:

1. **Off** (manual): disarms the current session until the next scheduled window starts.
2. **On** (manual): arms now. It ends when **Off** is pressed or after `manual_max_hours`
   (default 4). At the cap, send a normal "nap monitoring ended" notice.
3. **Schedule**: armed during configured windows.

**Pause** applies on top of any armed session. It ends after 45 min, or once he has been out of view
(`NO_VIEW`) and then 2 consecutive checks see him back in the crib (`BACK`/`SIDE`/`STOMACH`). Sightings
before he has left the view don't count, so pressing Pause before picking him up doesn't end it early. While paused: no stomach or "can't see him" alerts, forced
checks every 30 s, and health alerts stay active.

### Schedule config

```toml
[schedule]
cycle_anchor = "2026-09-25"      # a Friday that starts one of his weekends
cycle_days = 14
nights = ["fri", "sat", "sun"]   # nights within the cycle's first week that are armed
window_start = "19:00"
window_end = "07:00"             # next morning; windows may cross midnight
extra_dates = []                 # nights to arm outside the cycle, "YYYY-MM-DD"
skip_dates = []                  # cycle nights to skip
timezone = "America/New_York"
```

A night's date is the date the window *starts*. Times are wall-clock in `timezone`, so DST
transitions shorten or lengthen the window rather than shifting it.

When a session arms, the service runs a self-test: grab a frame and classify it with both models.
It then sends a normal "Monitoring: camera ✓/✗ local ✓/✗ cloud ✓/✗" notice with the frame
attached.

## Health alerts

| Condition | Priority | Behavior |
|---|---|---|
| No frames for 2 min while armed | High (1) | "Monitor blind." Repeats every 15 min; "recovered" when frames return |
| One model `Unavailable` on 3 consecutive checks | Normal (0) | "Running on one detector (local/cloud)." Sent once; "recovered" on the first success |
| Both models unavailable on 3 consecutive checks | High (1) | "No detectors." Repeats every 15 min; "recovered" on recovery |
| Service or host dead | via healthchecks.io | The service pings every 60 s whenever it is running, armed or not. healthchecks.io uses a 1 min period and 2 min grace, and its Pushover integration alerts |
| Pushover send fails | — | Retry with backoff and log. If it keeps failing, healthchecks.io is signalled with `/fail` so the outside alert path fires |

## Control page

- FastAPI, bound to the server's LAN address, port 8080. It requires a token in the URL, so the
  phone bookmark includes it. No access from outside the LAN.
- Main view: current state and why (schedule, manual, paused), last check time, last labels from
  each model, latest frame, and health status.
- Buttons: **On**, **Off**, **Pause 45 min**, **Resume**, **Test alert**. **Test alert** sends one
  emergency-priority message marked TEST to confirm it gets through Do Not Disturb.
- Labeling view (`/label`): steps through logged frames that are not yet labeled. Buttons for
  back / stomach / side / not visible / skip. Labels are written to the evaluation set.

## Storage

- `data/frames/YYYY-MM-DD/HHMMSS_mmm.jpg`: every checked frame (the crop, as sent to the models).
- `data/log/YYYY-MM-DD.jsonl`: one record per check, with timestamp, motion score, each model's
  label and latency (or error), combined result, state before and after, and any action taken.
- Frames and logs are deleted after `retention_days` (default 30). Labeled frames are copied into
  `data/eval/` and never deleted automatically.

## Configuration and secrets

- `config.toml` (committed as `config.example.toml`): schedule, crib crop, thresholds, intervals,
  model names, LiteLLM options, shadow mode flag.
- `.env` (not committed, mode 600): `TAPO_RTSP_URL`, `OPENROUTER_API_KEY`, `PUSHOVER_TOKEN`,
  `PUSHOVER_USER`, `HEALTHCHECKS_URL`, `CONTROL_TOKEN`.
- `shadow_mode = true` sends position alerts as normal-priority messages prefixed "[TEST]" instead
  of emergency. Health alerts are unchanged.

## Testing

- **Unit (pytest, fake clock):**
  - `schedule.py`: alternating weeks, windows across midnight, DST, extra and skip dates.
  - `decision.py`: every row of the transition table, plus label sequences that interleave with
    `FAILED`.
  - `arming.py`: source priority, pause ending on timeout and on sight, manual cap.
  - `health.py`: timers, repeat and recover.
  - Combining the two answers.
  - Classifier response parsing, including malformed output → `Unavailable`.
- **Integration:** mediamtx serves a recorded clip as RTSP. Test capture, reconnect after the
  server is killed, and the "monitor blind" alert. Pushover and healthchecks are stubbed with a
  local HTTP fake.
- **Model evaluation (`evaluate.py`):** runs one or more model configs over `data/eval/`. It reports
  per-model and combined stomach recall, false-alarm rate per night-equivalent, and latency.
  - Stomach examples come from supervised, awake tummy time in the crib, with room lights on and
    with them off (IR).
  - Back and side examples come from normal sleep logs.
  - Acceptance bar before leaving shadow mode: combined stomach recall of 100% on the eval set.
- **Rollout:** the first weekend runs in shadow mode. Review the log and label the frames, then
  switch `shadow_mode = false`.

## Deployment

- **Server:** Python 3.12, managed with `uv`; the `ffmpeg` system package; a systemd unit with
  `Restart=always` that runs as a dedicated user.
- **Laptop:** the model server runs as a systemd service. Lid-close suspend is disabled. The router
  gives the laptop a fixed DHCP reservation. It only needs to be on during armed sessions; the
  arm-time self-test reports if it isn't.
- **Dependencies** are pinned to exact versions in `uv.lock`, LiteLLM included.

## Out of scope

Audio or cry detection, breathing detection, access from outside the LAN, multiple cameras,
wake-on-LAN for the laptop, training a custom classifier (possible later using `data/eval/`).

## Risks to verify early

- Whether the C210 serves RTSP to this service while the Tapo app is viewing the live feed at the
  same time.
- How long the local model takes per image, including vision-encoder time, with partial GPU offload.
  If it is over 30 s, pick a smaller model or lower the input size.
- Whether each OpenRouter candidate model accepts images plus a JSON-schema `response_format`
  through LiteLLM. Otherwise fall back to prompt-only JSON with strict parsing.
