import numpy as np
import pytest

from robolivemem import clock, metrics
from robolivemem.commit import changed
from robolivemem.models import Chat, parse_json_reply
from robolivemem.tabletop import Episode, Scripted, make_instance, run, PAUSED, LIVE


def test_clocks():
    assert clock.PAUSED.kappa(4.2) == 0.0
    assert clock.LIVE.kappa(4.2) == 4.2
    assert clock.fixed(2).kappa(9.0) == 2.0
    assert clock.parse("fixed:8").kappa(0.1) == 8.0


def test_decomposition_adds_up():
    eps = []
    for i in range(40):
        eps += [dict(instance=i, condition="P", success=True),
                dict(instance=i, condition="V", success=i % 4 != 0),
                dict(instance=i, condition="L", success=i % 2 == 0)]
    d = metrics.decompose(eps)
    assert d["drop"]["diff"] == pytest.approx(d["execution"]["diff"] + d["memory"]["diff"])
    assert d["drop"]["lo"] <= d["drop"]["diff"] <= d["drop"]["hi"]


def test_reference_policies_validate_the_family():
    """Paper Sec. 4 (validation): privileged solves every seed; stale fails every event seed live; assume-change fails
    every decoy and null seed."""
    seeds = range(12)
    priv = run(lambda: Scripted("privileged"), seeds)
    assert all(r["success"] for r in priv)
    stale = run(lambda: Scripted("stale"), seeds)
    assert all(r["success"] for r in stale if r["condition"] == "P")
    assert not any(r["success"] for r in stale if r["condition"] == "L" and r["variant"] == "event")
    always = run(lambda: Scripted("assume-change"), seeds)
    assert not any(r["success"] for r in always if r["variant"] in ("decoy", "null"))


def test_paused_clock_never_shows_the_change_to_a_fast_agent():
    ep = Episode(make_instance(0), "P", PAUSED)
    row = ep.run(Scripted("stale", think_s=30.0))
    assert row["success"] and row["tau_s"] is None


def test_commit_check_detects_a_change():
    a = np.zeros((240, 320, 3), np.uint8)
    b = a.copy()
    b[100:140, 140:180] = 200
    assert changed(a, [a, b], (160, 120))
    assert not changed(a, [a, a], (160, 120))


def test_model_policy_end_to_end_with_mock():
    from robolivemem.tabletop import ModelPolicy
    rows = run(lambda: ModelPolicy('mock:I will pick it. {"action": "pick", "mark": 1}'), range(3), ("P",))
    assert len(rows) == 3 and all(r["decisions"] == 1 for r in rows)


@pytest.mark.parametrize("spec,key", [("openai:some-model", "OPENAI_API_KEY"), ("anthropic:some-model", "ANTHROPIC_API_KEY"),
                                      ("gemini:some-model", "GEMINI_API_KEY"), ("opus-5.5", "AWS_BEARER_TOKEN_BEDROCK"),
                                      ("gpt-6-astra", "AWS_BEARER_TOKEN_BEDROCK")])
def test_request_payloads(monkeypatch, spec, key):
    monkeypatch.setenv(key, "test-key")
    chat = Chat(spec)
    from robolivemem.models import _turns
    turns = _turns([("user", ["hello", np.zeros((8, 8, 3), np.uint8)]), ("assistant", ["ok"]), ("user", ["again"])])
    url, headers, body = getattr(chat, f"_req_{chat.route}")("system prompt", turns)
    assert url.startswith("https://") and "test-key" in " ".join(headers.values())
    assert "test-key" not in str(body)


def test_missing_key_is_reported_not_raised(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    text, meta = Chat("anthropic:some-model")("", [("user", ["hi"])])
    assert text == "" and "ANTHROPIC_API_KEY" in meta["error"]


def test_parse_json_reply():
    assert parse_json_reply('think... {"action": "wait", "seconds": 2}') == {"action": "wait", "seconds": 2}
    assert parse_json_reply('{"a": 1} then {"b": {"c": 2}}') == {"b": {"c": 2}}
    assert parse_json_reply("no json") is None
