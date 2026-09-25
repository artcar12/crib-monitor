You are deploying crib-monitor on this Ubuntu server. It is a Python 3.12 service that watches a Tapo C210 crib camera over RTSP. It classifies the baby's sleeping position with two vision models through LiteLLM: a local Ollama model on a laptop, and an OpenRouter model. It sends a Pushover emergency alert if he rolls onto his stomach. It is a backup to an Owlet sock. The code is finished and tested: 182 unit tests pass. Your job is to install it, configure it, and verify it on real hardware. You are not here to change how it works.

Repo: https://github.com/artcar12/crib-monitor (public, branch main)
Install location: /opt/crib-monitor, run as the system user `cribmon`
The repo is already cloned at /opt/crib-monitor. Skip the README's `git clone` command, but run every other install command, including the `chown`. If /opt/crib-monitor doesn't exist or isn't this repo (`git -C /opt/crib-monitor remote -v`), stop and ask me where the clone is.
Follow README.md in the repo exactly: "Server install (Ubuntu)", then "On-server verification checklist".

HARD RULES (secrets and safety)
- Never read, cat, grep, print or copy /opt/crib-monitor/.env, and never type a secret into anything. I fill in .env myself. When you reach that step, create the empty mode-600 file as the README says, stop, and tell me to fill it with `sudo -u cribmon nano /opt/crib-monitor/.env`. Wait until I say it's done.
- Never put the RTSP URL (it contains the camera password), Pushover token or user, CONTROL_TOKEN, OpenRouter key or healthchecks URL on a command line, in a log, or in your notes. The tools already read them from .env.
- Do not run ffmpeg against the camera yourself. Use `crib-monitor-probe`; it redacts the password from ffmpeg's output.
- Leave `shadow_mode = true` in config.toml. Only I turn it off, after the evaluation passes.
- Don't change any code under crib_monitor/ or tests/, and don't push anything to GitHub. If something in the code is broken, stop and report it with the exact error, with any secrets redacted.
- If sudo asks for a password, stop and ask me to run the command. Don't try to work around it.

STEPS
1. Pre-checks. Report the Ubuntu version and whether ffmpeg, git and uv are installed. Report the server's LAN IPv4 address (`ip -4 addr`). Check whether the laptop's Ollama is reachable: `curl -s http://alien.lan:11434/v1/models`. If it isn't, report that and carry on, since the laptop only needs to be on during sessions.
2. Do the "Server install (Ubuntu)" commands (without the clone) up to and including creating .env. Then stop for me to fill in .env (see HARD RULES).
3. In config.toml, set `[web] host` to the LAN address from step 1. Show me the `[schedule]` block and ask me to confirm that `cycle_anchor` is a Friday that starts one of his weekends. If `alien.lan` didn't resolve, ask me for the laptop's hostname or IP and set it in the local model's `api_base`.
4. Crop: run `sudo -u cribmon .venv/bin/crib-monitor-probe --full --out full.jpg` from /opt/crib-monitor. If you can view images, open full.jpg, propose the crib rectangle as x, y, w, h in pixels (all even numbers, with a small margin around the mattress), and describe what you see inside it. Otherwise give me the file path so I can measure it myself. I'll confirm before you write it into `[camera.crop]`.
5. Run `sudo -u cribmon .venv/bin/crib-monitor-probe`. Record the camera result and each model's answer and latency. The local model's latency must be under 30 s. The cloud model must not report `unavailable`. If either fails, stop and report.
6. Install and start the systemd unit exactly as the README says. Then check `journalctl -u crib-monitor -n 50` for errors, and confirm the page answers: `curl -s -o /dev/null -w '%{http_code}' http://LAN_IP:8080/` should print 403, because there's no token. Tell me to bookmark the page with the token on my phone and press "Test alert".
7. On-server checklist. Item 1: ask me to open the live feed in the Tapo app while you re-run the probe. Items 2 and 3 are done in step 5. Item 4: use a separate dev checkout in ~/crib-monitor-dev, never /opt: install mediamtx from its official GitHub releases (bluenviron/mediamtx), then `uv sync` and `uv run pytest -m integration`. Ask me before downloading anything. Item 5: ask me to press On and unplug the camera, then watch the journal. Item 6: ask me before you stop the service, and restart it afterwards.

REPORT
Write ~/crib-monitor-deploy-notes.md with:
- each step's result, as pass or fail, with timings
- the crop you used
- the probe output, with secrets redacted
- every checklist item's result
- anything you skipped or couldn't do, and why

Then give me a short summary. Don't say something passed unless you saw it pass.
