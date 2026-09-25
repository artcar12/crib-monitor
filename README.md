# Crib monitor

Watches a Tapo C210 over the crib during scheduled nights or manual nap sessions and sends a Pushover
emergency alert if he rolls onto his stomach. It is a backup to the Owlet, not a medical device.
Design: `docs/superpowers/specs/2026-09-24-crib-monitor-design.md`.

## One-time setup

**Camera.** In the Tapo app: camera → Settings → Advanced Settings → Camera Account; create a username and
password. The stream is `rtsp://USER:PASS@CAMERA_IP:554/stream1`. Give the camera a DHCP reservation.

**Pushover.** Install the Android app, note your user key, and create an application at pushover.net to get
an API token. In the Android app's settings, allow emergency-priority alerts to override Do Not Disturb and
silent mode.

**healthchecks.io.** Create a check with period 1 minute and grace 2 minutes. Add a Pushover integration
(high priority) so a dead server reaches your phone. Copy the ping URL.

**OpenRouter.** Create an API key. In OpenRouter's privacy settings, disallow providers that train on or
retain prompts. The config also sends `provider.data_collection = "deny"` on every request.

**Laptop (local model).** Install Ollama, set `OLLAMA_HOST=0.0.0.0` so the server can reach it, and pull the
vision model named in `config.toml` (for example `ollama pull qwen3-vl:30b`; check the exact tag in the
Ollama library). Disable suspend on lid close (on Ubuntu: `HandleLidSwitch=ignore` in
`/etc/systemd/logind.conf`, then `sudo systemctl restart systemd-logind`). Give it a DHCP reservation. It only
needs to be on during armed sessions.

## Server install (Ubuntu)

uv is installed system-wide, and the Python it downloads lives inside `/opt/crib-monitor` so the
`cribmon` user can reach it.

```bash
sudo apt install ffmpeg git
curl -LsSf https://astral.sh/uv/install.sh | sudo env UV_INSTALL_DIR=/usr/local/bin sh
sudo useradd --system --home /opt/crib-monitor --shell /usr/sbin/nologin cribmon
sudo git clone <this repo> /opt/crib-monitor
sudo chown -R cribmon:cribmon /opt/crib-monitor
cd /opt/crib-monitor
sudo -u cribmon env UV_PYTHON_INSTALL_DIR=/opt/crib-monitor/.uv/python UV_CACHE_DIR=/opt/crib-monitor/.uv/cache uv sync --frozen --no-dev
sudo -u cribmon cp config.example.toml config.toml
sudo -u cribmon install -m 600 /dev/null .env
```

Put these in `.env`. It is mode 600 and owned by `cribmon`, so edit it as that user
(`sudo -u cribmon nano .env`, likewise for `config.toml`). No quotes; URL-encode any special characters in
the camera password:

```
TAPO_RTSP_URL=rtsp://USER:PASS@CAMERA_IP:554/stream1
PUSHOVER_TOKEN=...
PUSHOVER_USER=...
HEALTHCHECKS_URL=https://hc-ping.com/...
CONTROL_TOKEN=<long random string, e.g. from: openssl rand -hex 16>
OPENROUTER_API_KEY=...
```

**Crib crop.** Save one full, uncropped frame (this reads `TAPO_RTSP_URL` from `.env` and classifies
nothing), copy `full.jpg` to a machine with an image viewer that shows cursor position, and find the crib
rectangle in pixel coordinates. Put it in `[camera.crop]`; all four values must be even.

```bash
sudo -u cribmon .venv/bin/crib-monitor-probe --full --out full.jpg
```

Set `[web] host` to the server's LAN address, then check everything end to end:

```bash
sudo -u cribmon .venv/bin/crib-monitor-probe
sudo cp deploy/crib-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now crib-monitor
journalctl -u crib-monitor -f
```

Bookmark `http://SERVER_IP:8080/?t=<CONTROL_TOKEN>` on your phone's home screen and press **Test alert** to
confirm Pushover gets through Do Not Disturb.

## Before trusting it

1. First weekend: leave `shadow_mode = true`. Roll alerts arrive as normal-priority `[TEST]` messages.
2. Collect stomach frames safely: press **On**, then **Pause**, and do supervised tummy time in the crib while
   he is awake, once with the lights on and once with them off (infrared). Pause keeps checking every 30 s
   without alerting.
3. Label frames on the `/label` page.
4. Run `sudo -u cribmon .venv/bin/crib-monitor-eval` from `/opt/crib-monitor`. It must print `PASS` (every labeled stomach frame caught) before you set
   `shadow_mode = false` and restart the service. Re-run it when you change models.

## Tests

```bash
uv run pytest                    # unit tests
uv run pytest -m integration     # needs ffmpeg and mediamtx on PATH
```

The service install uses `uv sync --no-dev`, so run the integration test from a dev checkout instead (for
example a clone in your home directory on the server): `uv sync`, with ffmpeg and mediamtx installed.

## On-server verification checklist

Do this on the Ubuntu server after the service is running; it exercises the spec's "Risks to verify early"
and cannot be checked from a dev machine. Record the results in the PR or a note.

1. `crib-monitor-probe` prints `camera: ok` while the Tapo app is showing the live feed on the phone at the
   same time.
2. `crib-monitor-probe` shows the local model's latency under 30 s. If it is slower, choose a smaller model
   or lower `max_side_px`.
3. `crib-monitor-probe` gets a valid answer from the cloud model through OpenRouter (no
   `unavailable (… response_format …)` error). If the chosen model rejects `json_schema`, pick another model;
   parsing already tolerates plain JSON.
4. `uv run pytest -m integration` passes on the server, run from a dev checkout (`uv sync`, with ffmpeg and
   mediamtx installed), not from `/opt/crib-monitor`, which has no dev dependencies.
5. Unplug the camera during a manual session: a "Monitor blind" alert arrives within about 2 minutes, and
   "Camera recovered" after plugging it back in.
6. `sudo systemctl stop crib-monitor`: healthchecks.io alerts within about 3 minutes.
