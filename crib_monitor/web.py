"""LAN-only control page: status, On/Off/Pause, test alert, frame labeling."""

from __future__ import annotations

import html
import secrets
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Form, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from .monitor import Snapshot
from .storage import Storage

_CSS = (
    "body{font-family:system-ui,sans-serif;margin:0 auto;max-width:640px;padding:16px;background:#fff;color:#111}"
    "@media (prefers-color-scheme:dark){body{background:#111;color:#eee}a{color:#9cf}}"
    "img{width:100%;border-radius:8px;background:#333}"
    ".row{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}"
    ".row form{flex:1 1 40%}"
    "button{width:100%;font-size:1.1rem;padding:14px 18px;border-radius:10px;border:1px solid #888}"
    ".warn{background:#fde68a;color:#111;padding:8px;border-radius:8px}"
)

_BUTTONS = [("on", "On"), ("off", "Off"), ("pause", "Pause 45 min"), ("resume", "Resume"), ("test", "Test alert")]
_LABEL_BUTTONS = [("back", "Back"), ("stomach", "Stomach"), ("side", "Side"), ("not_visible", "Not visible"), ("skip", "Skip")]


def describe(snapshot: Snapshot, tz: ZoneInfo) -> str:
    st = snapshot.status
    if st is None or not st.armed:
        return "Off"
    if st.paused and st.paused_until:
        return f"Paused until {st.paused_until.astimezone(tz):%H:%M}"
    if st.source == "manual" and st.manual_until:
        return f"On (nap) until {st.manual_until.astimezone(tz):%H:%M}"
    if st.window_end:
        return f"On (schedule) until {st.window_end.astimezone(tz):%H:%M}"
    return "On"


def _html(title: str, body: str, refresh: int | None = None) -> str:
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"{meta}<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>{body}</body></html>"
    )


def _status_page(s: Snapshot, q: str, tz: ZoneInfo) -> str:
    parts = [f"<h1>{html.escape(describe(s, tz))}</h1>"]
    if s.shadow_mode:
        parts.append('<p class="warn">Shadow mode: roll alerts are sent as [TEST] messages.</p>')
    for issue in s.health:
        parts.append(f'<p class="warn">Health: {html.escape(issue)}</p>')
    if s.last_check:
        labels = ", ".join(
            f"{html.escape(r.model_name)}={html.escape(r.position.value if r.position else 'unavailable')}"
            for r in s.last_results
        )
        parts.append(
            f"<p>Last check {s.last_check.astimezone(tz):%H:%M:%S}: "
            f"{html.escape(s.last_combined or '')} ({labels}) &middot; state {html.escape(s.state)}</p>"
        )
    parts.append(f'<img src="/frame.jpg{q}" alt="Latest frame">')
    buttons = "".join(
        f'<form method="post" action="/{action}{q}"><button>{label}</button></form>' for action, label in _BUTTONS
    )
    parts.append(f'<div class="row">{buttons}</div>')
    parts.append(f'<p><a href="/label{q}">Label frames</a></p>')
    return _html("Crib monitor", "".join(parts), refresh=15)


def _label_page(remaining: list[str], q: str) -> str:
    if not remaining:
        body = "<h1>Nothing to label</h1>"
    else:
        rel = remaining[0]
        buttons = "".join(
            f'<button name="label" value="{value}" style="flex:1 1 40%">{text}</button>' for value, text in _LABEL_BUTTONS
        )
        body = (
            f"<h1>Label ({len(remaining)} left)</h1><p>{html.escape(rel)}</p>"
            f'<img src="/frames/{quote(rel)}{q}" alt="Frame to label">'
            f'<form method="post" action="/label{q}"><input type="hidden" name="image" value="{html.escape(rel)}">'
            f'<div class="row">{buttons}</div></form>'
        )
    return _html("Label frames", body + f'<p><a href="/{q}">Back to status</a></p>')


def create_app(controller: Any, storage: Storage, token: str, tz: ZoneInfo) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    q = f"?t={quote(token)}"

    def auth(t: str = Query("")) -> None:
        if not secrets.compare_digest(t.encode(), token.encode()):
            raise HTTPException(status_code=403)

    def home() -> RedirectResponse:
        return RedirectResponse("/" + q, status_code=303)

    @app.get("/", response_class=HTMLResponse, dependencies=[Depends(auth)])
    async def index() -> HTMLResponse:
        return HTMLResponse(_status_page(controller.snapshot(), q, tz))

    @app.post("/on", dependencies=[Depends(auth)])
    async def on() -> RedirectResponse:
        controller.on()
        return home()

    @app.post("/off", dependencies=[Depends(auth)])
    async def off() -> RedirectResponse:
        controller.off()
        return home()

    @app.post("/pause", dependencies=[Depends(auth)])
    async def pause() -> RedirectResponse:
        controller.pause()
        return home()

    @app.post("/resume", dependencies=[Depends(auth)])
    async def resume() -> RedirectResponse:
        controller.resume()
        return home()

    @app.post("/test", dependencies=[Depends(auth)])
    async def test() -> RedirectResponse:
        await controller.test_alert()
        return home()

    @app.get("/frame.jpg", dependencies=[Depends(auth)])
    async def frame() -> Response:
        data = controller.latest_jpeg()
        if data is None:
            raise HTTPException(status_code=404)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/label", response_class=HTMLResponse, dependencies=[Depends(auth)])
    async def label_page() -> HTMLResponse:
        return HTMLResponse(_label_page(storage.unlabeled(), q))

    @app.post("/label", dependencies=[Depends(auth)])
    async def label(image: str = Form(...), label: str = Form(...)) -> RedirectResponse:
        try:
            storage.add_label(image, label)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return RedirectResponse("/label" + q, status_code=303)

    @app.get("/frames/{rel:path}", dependencies=[Depends(auth)])
    async def frames(rel: str) -> FileResponse:
        try:
            path = storage.frame_path(rel)
        except ValueError as exc:
            raise HTTPException(status_code=404) from exc
        return FileResponse(path, media_type="image/jpeg")

    return app
