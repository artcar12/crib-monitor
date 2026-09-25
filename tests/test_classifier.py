import asyncio
import base64
from types import SimpleNamespace

import pytest

from crib_monitor.classifier import Classifier, build_classifiers, classify_all, parse_position
from crib_monitor.config import ModelConfig
from crib_monitor.labels import Position


def reply(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeCompletion:
    def __init__(self, content=None, exc=None, delay=0.0):
        self.content, self.exc, self.delay, self.calls = content, exc, delay, []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return reply(self.content)


CLOUD = ModelConfig(
    name="cloud", model="openrouter/vendor/model", api_key_env="OPENROUTER_API_KEY",
    timeout_s=1, extra_body={"provider": {"data_collection": "deny"}},
)
LOCAL = ModelConfig(name="local", model="openai/qwen", api_base="http://alien:11434/v1", timeout_s=1)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"position": "stomach"}', Position.STOMACH),
        ('```json\n{"position": "back"}\n```', Position.BACK),
        ('  {"position": "not_visible"} ', Position.NOT_VISIBLE),
    ],
)
def test_parse_position(content, expected):
    assert parse_position(content) is expected


@pytest.mark.parametrize("content", [None, "", "stomach", '{"pos": "back"}', '{"position": "upside_down"}', "[1]"])
def test_parse_position_rejects(content):
    with pytest.raises(ValueError):
        parse_position(content)


async def test_classify_sends_image_and_options():
    fake = FakeCompletion('{"position": "side"}')
    result = await Classifier(CLOUD, "sk-or", completion=fake).classify(b"\xff\xd8jpeg")
    assert result.position is Position.SIDE and result.error is None and result.model_name == "cloud"
    call = fake.calls[0]
    assert call["model"] == "openrouter/vendor/model"
    assert call["api_key"] == "sk-or"
    assert call["extra_body"] == {"provider": {"data_collection": "deny"}}
    assert call["response_format"]["type"] == "json_schema"
    image = call["messages"][0]["content"][1]["image_url"]["url"]
    assert image == "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8jpeg").decode()


async def test_local_uses_api_base():
    fake = FakeCompletion('{"position": "back"}')
    await Classifier(LOCAL, None, completion=fake).classify(b"x")
    assert fake.calls[0]["api_base"] == "http://alien:11434/v1"
    assert fake.calls[0]["api_key"] == "none"
    assert "extra_body" not in fake.calls[0]


async def test_malformed_answer_is_unavailable():
    result = await Classifier(LOCAL, None, completion=FakeCompletion("I think back")).classify(b"x")
    assert result.position is None and result.error.startswith("JSONDecodeError")


async def test_exception_is_unavailable():
    result = await Classifier(LOCAL, None, completion=FakeCompletion(exc=RuntimeError("boom"))).classify(b"x")
    assert result.position is None and "boom" in result.error


async def test_timeout_is_unavailable():
    cfg = LOCAL.model_copy(update={"timeout_s": 0.05})
    result = await Classifier(cfg, None, completion=FakeCompletion('{"position": "back"}', delay=1)).classify(b"x")
    assert result.position is None and "Timeout" in result.error


async def test_exception_redacts_api_key():
    fake = FakeCompletion(exc=RuntimeError("bad key sk-secret-123"))
    result = await Classifier(CLOUD, "sk-secret-123", completion=fake).classify(b"x")
    assert result.position is None
    assert "sk-secret-123" not in result.error
    assert "RuntimeError" in result.error


async def test_classify_all_runs_concurrently():
    slow = [Classifier(m, None, completion=FakeCompletion('{"position": "back"}', delay=0.2)) for m in (LOCAL, CLOUD)]
    loop = asyncio.get_running_loop()
    start = loop.time()
    results = await classify_all(slow, b"x")
    assert loop.time() - start < 0.35
    assert [r.model_name for r in results] == ["local", "cloud"]


def test_build_classifiers_reads_keys(config):
    classifiers = build_classifiers(config.classifier, {"OPENROUTER_API_KEY": "sk-or"})
    assert [c.name for c in classifiers] == ["local", "cloud"]
