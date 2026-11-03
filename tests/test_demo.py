"""The reviewer client must distinguish partial/wrong-model text from success."""
import importlib.util
import json
from pathlib import Path
import sys

import httpx
import pytest


spec = importlib.util.spec_from_file_location("platform_demo", Path(__file__).parents[1] / "scripts/demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
MODEL = "a" * 64


def run_demo(monkeypatch, body, *, stream=True):
    real_client = httpx.Client

    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": MODEL, "loaded": True}]})
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(demo.httpx, "Client", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr(sys, "argv", ["demo.py"] + (["--stream"] if stream else []))
    demo.main()


def event(*, model=MODEL, text="", finish=None):
    return "data: " + json.dumps({"model": model, "choices": [{
        "delta": {"content": text}, "finish_reason": finish}]}) + "\n\n"


@pytest.mark.parametrize("finish,done", [(None, False), (None, True), ("stop", False), ("error", True)])
def test_demo_rejects_partial_or_unsuccessful_stream(monkeypatch, finish, done):
    body = event(text="PARTIAL", finish=finish) + ("data: [DONE]\n\n" if done else "")
    with pytest.raises(SystemExit, match="incomplete stream"):
        run_demo(monkeypatch, body)


@pytest.mark.parametrize("model", [None, "b" * 64])
def test_demo_rejects_stream_without_expected_artifact(monkeypatch, model):
    with pytest.raises(SystemExit, match="identity mismatch"):
        run_demo(monkeypatch, event(model=model, text="wrong", finish="stop") + "data: [DONE]\n\n")


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_demo_accepts_completed_expected_model_stream(monkeypatch, capsys, finish):
    run_demo(monkeypatch, event(text="hello") + event(finish=finish) + "data: [DONE]\n\n")
    assert "hello" in capsys.readouterr().out


@pytest.mark.parametrize("model,finish,error", [("b" * 64, "stop", "identity mismatch"),
                                               (MODEL, None, "incomplete completion")])
def test_demo_rejects_wrong_or_incomplete_sync_completion(monkeypatch, model, finish, error):
    body = json.dumps({"model": model, "choices": [{"message": {"content": "partial"},
                                                    "finish_reason": finish}]})
    with pytest.raises(SystemExit, match=error):
        run_demo(monkeypatch, body, stream=False)


def test_demo_accepts_expected_sync_completion(monkeypatch, capsys):
    body = json.dumps({"model": MODEL, "choices": [{"message": {"content": "hello"},
                                                     "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}})
    run_demo(monkeypatch, body, stream=False)
    assert "Measured token usage" in capsys.readouterr().out
