"""Real offline weights: compare the displayed distribution to native single-choice."""
import json
import math
import socket

from app import DecisionEngine, DecisionRequest


def forbidden_network(*args, **kwargs):
    raise AssertionError("Inference attempted a network connection")


if __name__ == "__main__":
    socket.socket.connect = forbidden_network
    engine = DecisionEngine()
    engine.load()
    assert engine.status == "ready", engine.error
    cases = [
        DecisionRequest(state="My package arrived damaged and I want a refund.", question="Which queue should handle this?", options=["billing", "shipping", "technical", "general"]),
        DecisionRequest(state="The treaty was signed in Paris in 1992. It entered into force the following year.", question="Did the treaty enter into force in 1992?", options=["yes", "no", "unknown"]),
        DecisionRequest(state="A recorded clip shows two people standing apart. One raises a hand to wave. No physical contact is visible. The depth measurement is unavailable.", question="Which review label best matches the described evidence?", options=["ordinary interaction", "possible physical conflict", "insufficient evidence"]),
    ]
    for payload in cases:
        result = engine.decide(payload)
        native = engine.model.classify_text(payload.state, {"answer": {"labels": payload.options, "prompt": payload.question}}, include_confidence=True)["answer"]
        assert result["answer"] == native["label"]
        assert math.isclose(result["scores"][0]["score"], native["confidence"], abs_tol=1e-6)
        assert set(row["label"] for row in result["scores"]) == set(payload.options)
        again = engine.decide(payload)
        assert result["scores"] == again["scores"]
        print(json.dumps(result), flush=True)
    print("PASS: real local weights, no network, complete distributions, native score parity, repeatability.")
