"""Run with python paper1/test_adjusted_template_intervals.py."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from adjusted_template_intervals import factorial, interval


def test():
    weighted = interval([("A", 1), ("A", 1), ("B", 0)], 4, 20261002)
    assert weighted["point"] == 2 / 3  # Equal-template weighting would give 1/2.
    assert weighted["native_templates"] == 2 and weighted["groups"] == 3

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        folder = root / "paper1/results/qwen_policy_factorial_20261002_v1_evaluation"
        folder.mkdir(parents=True)
        rows = []
        for group in ("g1", "g2", "g3"):
            for realization in (0, 1):
                for renderer in ("original", "predicate", "truth_table"):
                    for policy in ("all_hate", "directed_hate", "general_hate"):
                        rows.append({"dataset": "hatecheck", "exact_text_group": group,
                                     "realization": realization, "renderer": renderer, "policy": policy,
                                     "valid": not (group == "g3" and renderer == "truth_table" and policy == "all_hate"),
                                     "correct": 1})
        (folder / "outcomes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (folder / "summary.json").write_text(json.dumps({"datasets": {"hatecheck": {
            "groups": 3, "renderers": {"truth_table": {"correctness_cross_minus_fresh": {"mean": 0}}},
            "balanced_cross_minus_wording": {"effect": {"mean": 0}}}}}))
        result = factorial(root, "qwen", {"g1": "A", "g2": "A", "g3": "B"}, {})
        assert all(r["groups"] == 2 and r["selected"] == 3 and r["coverage"] == 2 / 3
                   for r in result.values())


if __name__ == "__main__":
    test()
    print("ok")
