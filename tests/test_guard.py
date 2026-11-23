
from physguard import Bound, Claim, Guard, Verdict, guarded_answer

BEAM = {"F": "1 kN", "L": "2 m", "E": "200 GPa", "I": "8e-6 m**4"}  # reference 1.6667 mm


def claim(answer, **kw):
    return Claim(answer=answer, law="cantilever_tip_deflection", inputs=BEAM, **kw)


def test_correct_answer_passes():
    assert Guard().verify(claim("1.6667 mm")).verdict == Verdict.PASS


def test_unit_conversion_is_handled():
    assert Guard().verify(claim("0.0016667 m")).verdict == Verdict.PASS


def test_prefix_error_is_flagged_with_hint():
    r = Guard().verify(claim("1.6667 m"))
    assert r.verdict == Verdict.FAIL
    assert "10^3" in r.feedback()


def test_wrong_dimension_fails():
    r = Guard().verify(claim("1.6667 kg"))
    assert r.verdict == Verdict.FAIL
    assert any(c.name == "dimension" for c in r.checks if not c.passed)


def test_wrong_magnitude_fails():
    assert Guard().verify(claim("2.5 mm")).verdict == Verdict.FAIL


def test_tolerance_is_configurable():
    assert Guard(rtol=0.05).verify(claim("1.70 mm")).verdict == Verdict.PASS
    assert Guard(rtol=0.001).verify(claim("1.70 mm")).verdict == Verdict.FAIL


def test_unparsable_answer_fails():
    assert Guard().verify(Claim(answer="about a millimetre")).verdict == Verdict.FAIL


def test_no_law_gives_warn_not_pass():
    assert Guard().verify(Claim(answer="3 m")).verdict == Verdict.WARN


def test_universal_limit_speed_of_light():
    assert Guard().verify(Claim(answer="4e8 m/s")).verdict == Verdict.FAIL


def test_negative_kelvin_fails():
    assert Guard().verify(Claim(answer="-5 kelvin")).verdict == Verdict.FAIL


def test_user_bound():
    g = Guard(bounds=Bound(min=0, max=0.01, unit="m"))
    assert g.verify(claim("1.6667 mm")).verdict == Verdict.PASS
    assert g.verify(claim("1.6667 m")).verdict == Verdict.FAIL


class ScriptedLLM:
    """Returns canned replies in order, recording prompts. No network."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def _json(answer):
    import json
    return json.dumps({"answer": answer, "law": "cantilever_tip_deflection", "inputs": BEAM})


def test_repair_loop_recovers_after_feedback():
    llm = ScriptedLLM([_json("1.6667 m"), "```json\n" + _json("1.6667 mm") + "\n```"])
    res = guarded_answer("Tip deflection?", llm)
    assert res.accepted and res.attempts == 2
    assert "failed verification" in llm.prompts[1]


def test_repair_loop_gives_up():
    llm = ScriptedLLM([_json("9 m")] * 3)
    res = guarded_answer("Tip deflection?", llm, max_attempts=3)
    assert not res.accepted and res.attempts == 3


def test_invalid_json_is_retried():
    llm = ScriptedLLM(["I think it is small", _json("1.6667 mm")])
    assert guarded_answer("q", llm).accepted

