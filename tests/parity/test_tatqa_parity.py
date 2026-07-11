"""TAT-QA metric parity: our port vs. the real official evaluator.

Both implementations are handed identical ``(predicted_answer, predicted_scale)`` pairs against
identical gold, and must return identical EM and F1.

The pairs are fed **directly**, bypassing our answer-extraction step, and that is deliberate: the
official evaluator is given a ``[answer, scale]`` pair because TAT-QA's reference model has a
scale-prediction head. An LLM does not, so we infer the scale from its prose — but that inference
is *ours*, not TAT-QA's, and folding it into a parity test would be testing our code against
itself. This suite proves the **metric** is official. The extraction is tested separately, on its
own terms, in ``tests/unit/test_native_tatqa.py``.

The cases target exactly where a plausible port drifts:
  - multi-span answers, where the official F1 uses a Hungarian assignment (not bag overlap);
  - arithmetic/count answers, where ``f1`` is forced equal to ``em``;
  - scale folding, where ``percent`` means x0.01 and ``million`` means x1e6;
  - the 0.2342-vs-23.42% equivalence that ``add_percent_pred`` exists to resolve.
"""

from __future__ import annotations

import json

import pytest
from tests.parity.official_runner import REFERENCES, requires_official, run_official

from financebench.evaluation.native.tatqa import (
    get_metrics,
    normalize_answer,
    tatqa_em_and_f1,
    to_number,
)

TATQA_REPO = REFERENCES / "tatqa"
DEV_JSON = TATQA_REPO / "dataset_raw" / "tatqa_dataset_dev.json"

pytestmark = requires_official(TATQA_REPO / "tatqa_metric.py", DEV_JSON)


# (predicted, pred_scale, gold, gold_scale, answer_type)
CASES: list[tuple[list[str], str, list[str], str, str]] = [
    # -- exact hits
    (["$1,496.5"], "million", ["$1,496.5"], "million", "span"),
    (["12.5"], "percent", ["12.5"], "percent", "arithmetic"),
    # -- the 0.2342 vs 23.42% equivalence
    (["0.2342"], "", ["23.42"], "percent", "arithmetic"),
    (["23.42"], "percent", ["23.42"], "percent", "arithmetic"),
    # -- wrong scale, right number: must NOT be a match
    (["1496.5"], "thousand", ["1496.5"], "million", "arithmetic"),
    (["1496.5"], "", ["1496.5"], "million", "arithmetic"),
    # -- multi-span: the Hungarian assignment case
    (
        ["cost-plus type", "fixed-price type", "time-and-material type"],
        "",
        ["fixed-price type", "cost-plus type", "time-and-material type"],
        "",
        "multi-span",
    ),
    (
        ["fixed-price type", "cost-plus type"],
        "",
        ["fixed-price type", "cost-plus type", "time-and-material type"],
        "",
        "multi-span",
    ),
    (["fixed-price type"], "", ["fixed-price type", "cost-plus type"], "", "multi-span"),
    (["wrong one", "another wrong"], "", ["fixed-price type", "cost-plus type"], "", "multi-span"),
    # -- partial token overlap on a single span (F1 must be between 0 and 1)
    (["allowable incurred costs"], "", ["our allowable incurred costs plus a profit"], "", "span"),
    # -- arithmetic: f1 must be forced to em, so a partial token overlap earns NOTHING
    (["100"], "", ["100.5"], "", "arithmetic"),
    (["3"], "", ["3"], "", "count"),
    (["4"], "", ["3"], "", "count"),
    # -- negatives in parentheses
    (["(134)"], "", ["-134"], "", "arithmetic"),
    # -- empty prediction
    ([], "", ["something"], "", "span"),
]


_OFFICIAL = """
import sys, json
sys.path.insert(0, ".")
from tatqa_metric import TaTQAEmAndF1
cases = json.load(sys.stdin)
out = []
for c in cases:
    m = TaTQAEmAndF1()
    ground_truth = {
        "uid": "x",
        "answer": c["gold"] if c["answer_type"] in ("span", "multi-span") else c["gold"][0],
        "answer_type": c["answer_type"],
        "scale": c["gold_scale"],
        "answer_from": "table",
    }
    m(ground_truth=ground_truth, prediction=c["predicted"], pred_scale=c["pred_scale"])
    em, f1, scale, op = m.get_overall_metric()
    out.append([em, f1])
print(json.dumps(out))
"""


def _payload() -> list[dict]:
    return [
        {
            "predicted": predicted,
            "pred_scale": pred_scale,
            "gold": gold,
            "gold_scale": gold_scale,
            "answer_type": answer_type,
        }
        for predicted, pred_scale, gold, gold_scale, answer_type in CASES
    ]


def test_em_and_f1_match_the_official_evaluator() -> None:
    official = run_official(_OFFICIAL, cwd=TATQA_REPO, payload=_payload())

    ours = [
        list(tatqa_em_and_f1(predicted, pred_scale, gold, gold_scale, answer_type))
        for predicted, pred_scale, gold, gold_scale, answer_type in CASES
    ]

    assert len(ours) == len(official)
    for case, mine, theirs in zip(CASES, ours, official, strict=True):
        assert mine == pytest.approx(theirs, abs=1e-9), f"disagreement on {case!r}"


def test_em_and_f1_match_on_real_dev_gold_answers_scored_against_themselves() -> None:
    """A gold answer scored against itself must be a perfect 1.0/1.0 in BOTH implementations.

    Run over a slice of the real dev split — it exercises the actual distribution of answer types,
    scales and multi-span shapes rather than the hand-picked cases above.
    """
    records = json.loads(DEV_JSON.read_text(encoding="utf-8"))
    cases: list[dict] = []
    for record in records[:60]:
        for question in record["questions"]:
            answer = question["answer"]
            gold = [str(a) for a in answer] if isinstance(answer, list) else [str(answer)]
            cases.append(
                {
                    "predicted": gold,
                    "pred_scale": question["scale"],
                    "gold": gold,
                    "gold_scale": question["scale"],
                    "answer_type": question["answer_type"],
                }
            )

    official = run_official(_OFFICIAL, cwd=TATQA_REPO, payload=cases)
    ours = [
        list(
            tatqa_em_and_f1(
                c["predicted"], c["pred_scale"], c["gold"], c["gold_scale"], c["answer_type"]
            )
        )
        for c in cases
    ]

    assert len(cases) > 100, "expected a meaningful number of real questions"
    for case, mine, theirs in zip(cases, ours, official, strict=True):
        assert mine == pytest.approx(theirs, abs=1e-9), f"disagreement on {case!r}"
        assert mine[0] == 1.0, f"gold scored against itself must be an exact match: {case!r}"


_OFFICIAL_UTILS = """
import sys, json
sys.path.insert(0, ".")
from tatqa_utils import normalize_answer, to_number
payload = json.load(sys.stdin)
print(json.dumps({
    "normalize": [normalize_answer(t) for t in payload["normalize"]],
    "to_number": [to_number(t) for t in payload["to_number"]],
}))
"""


def test_answer_normalization_matches_the_official_utils() -> None:
    """Normalization drives both EM and F1, so a drift here silently moves every number."""
    texts = [
        "$1,496.5",
        "(134)",
        "12.5%",
        "The Fixed-Price Type",
        "1 million",
        "  spaced   out  ",
        "-32",
        "a an the",
        "0.2342",
    ]
    official = run_official(
        _OFFICIAL_UTILS, cwd=TATQA_REPO, payload={"normalize": texts, "to_number": texts}
    )

    assert [normalize_answer(t) for t in texts] == official["normalize"]
    for text, theirs in zip(texts, official["to_number"], strict=True):
        mine = to_number(text)
        assert mine == pytest.approx(theirs), f"to_number disagreement on {text!r}"
        # The int/float distinction is load-bearing: str(round(-134, 4)) is "-134", but
        # str(round(-134.0, 4)) is "-134.0", and that string is what EM compares.
        assert type(mine) is type(theirs), (
            f"to_number({text!r}) returned {type(mine).__name__}, official returned "
            f"{type(theirs).__name__} — this changes the normalized answer string"
        )


def test_get_metrics_matches_official_on_raw_string_pairs() -> None:
    """The innermost function: bag-of-tokens F1 with the Hungarian alignment."""
    pairs = [
        ("fixed price", "fixed price"),
        ("fixed price type", "fixed price"),
        ("completely different", "fixed price"),
        ("a b c", "c b a"),
    ]
    script = """
import sys, json
sys.path.insert(0, ".")
from tatqa_metric import get_metrics
pairs = json.load(sys.stdin)
print(json.dumps([list(get_metrics(p, g)) for p, g in pairs]))
"""
    official = run_official(script, cwd=TATQA_REPO, payload=pairs)
    ours = [list(get_metrics(p, g)) for p, g in pairs]
    for pair, mine, theirs in zip(pairs, ours, official, strict=True):
        assert mine == pytest.approx(theirs), f"disagreement on {pair!r}"
