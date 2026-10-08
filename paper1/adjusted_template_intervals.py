"""Post-review native-template intervals for frozen single-post Paper 1 outcomes.

Usage: python paper1/adjusted_template_intervals.py --root PROJECT --output DIR
The root supplies existing prepared and scored files; the output is a fresh directory.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def interval(pairs, family, seed):
    """Resample templates, taking the ratio of sampled group totals to counts."""
    blocks = defaultdict(list)
    for template, value in pairs:
        if template is None:
            raise ValueError("A native template ID is missing")
        blocks[template].append(value)
    if not blocks:
        raise ValueError("Empty conditional panel")
    ordered = [blocks[k] for k in sorted(blocks)]
    totals = np.array([sum(b) for b in ordered])
    sizes = np.array([len(b) for b in ordered])
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, 10000, 128):
        idx = rng.integers(len(ordered), size=(min(128, 10000 - start), len(ordered)))
        draws.extend(totals[idx].sum(axis=1) / sizes[idx].sum(axis=1))
    return {"point": float(totals.sum() / sizes.sum()), "ci": np.quantile(draws, [.025 / family, 1 - .025 / family]).tolist(),
            "groups": int(sizes.sum()), "native_templates": len(ordered), "family_size": family,
            "nominal_confidence": 1 - .05 / family, "replicates": 10000, "seed": seed}


def checked(pairs, family, seed, expected, selected):
    result = interval(pairs, family, seed)
    if not np.isclose(result["point"], expected, atol=1e-12, rtol=0):
        raise ValueError(f"Frozen point mismatch: {result['point']} != {expected}")
    result["selected"] = selected
    result["coverage"] = result["groups"] / selected
    return result


def factorial(root, model, templates, receipt):
    folder = root / "paper1/results" / f"{model}_policy_factorial_20261002_v1_evaluation"
    outcomes, summary = folder / "outcomes.jsonl", folder / "summary.json"
    receipt[str(outcomes.relative_to(root))] = digest(outcomes)
    receipt[str(summary.relative_to(root))] = digest(summary)
    by_group = defaultdict(dict)
    for r in read_jsonl(outcomes):
        if r["dataset"] == "hatecheck":
            key = (r["realization"], r["renderer"], r["policy"])
            if key in by_group[r["exact_text_group"]]:
                raise ValueError("Duplicate factorial cell")
            by_group[r["exact_text_group"]][key] = r
    report = json.loads(summary.read_text())["datasets"]["hatecheck"]
    policies = ("all_hate", "directed_hate", "general_hate")
    renderers = ("original", "predicate", "truth_table")
    truth, balanced = [], []
    for group, cells in by_group.items():
        if group not in templates:
            raise ValueError("Unmatched factorial group")
        def cell(i, renderer, policy):
            return cells[i, renderer, policy]
        def crossings(a, b):
            return int(a["correct"] != b["correct"])
        if all(cell(i, "truth_table", p)["valid"] for i in (0, 1) for p in policies):
            cross = sum(crossings(cell(i, "truth_table", a), cell(i, "truth_table", b))
                        for i in (0, 1) for j, a in enumerate(policies) for b in policies[j + 1:])
            fresh = sum(crossings(cell(0, "truth_table", p), cell(1, "truth_table", p)) for p in policies)
            truth.append((templates[group], cross / 6 - fresh / 3))
        if all(cell(i, r, p)["valid"] for i in (0, 1) for r in renderers for p in policies):
            across = sum(crossings(cell(i, r, a), cell(i, r, b)) for i in (0, 1) for r in renderers
                         for j, a in enumerate(policies) for b in policies[j + 1:])
            wording = sum(crossings(cell(i, a, p), cell(i, b, p)) for i in (0, 1) for p in policies
                          for j, a in enumerate(renderers) for b in renderers[j + 1:])
            balanced.append((templates[group], across / 18 - wording / 18))
    return {
        "truth_table_cross_minus_fresh": checked(truth, 4, 20261002,
            report["renderers"]["truth_table"]["correctness_cross_minus_fresh"]["mean"], report["groups"]),
        "balanced_cross_minus_wording": checked(balanced, 4, 20261002,
            report["balanced_cross_minus_wording"]["effect"]["mean"], report["groups"]),
    }


def original(root, model, templates, receipt):
    folder = root / "paper1/runs" / f"{model}_hatecheck_formal_v1/evaluation"
    paths = {name: folder / name for name in ("case_outcomes.jsonl", "repeat_case_outcomes.jsonl", "summary.json")}
    receipt.update({str(p.relative_to(root)): digest(p) for p in paths.values()})
    case_rows, repeat_rows = read_jsonl(paths["case_outcomes.jsonl"]), read_jsonl(paths["repeat_case_outcomes.jsonl"])
    reference = json.loads(paths["summary.json"].read_text())["functional_probe"]["native_template_sensitivity"]
    by_case = {(r["exact_text_group"], r["method"]): r for r in case_rows}
    result = {}
    for method, family in (("policy_compiler", 2), ("direct", 6), ("guarded_direct", 6), ("full_two_stage", 6)):
        key = "neutral_minus_" + method
        values = []
        for group, template in templates.items():
            neutral, control = by_case[group, "neutral_compiler"], by_case[group, method]
            values.append((template, neutral["mean_policy_joint_correct"] - control["mean_policy_joint_correct"]))
        result[key] = checked(values, family, 20260930, reference[key]["difference"], len(templates))
    for method in ("direct", "guarded_direct", "full_record"):
        key = "repeat_correctness_cross_minus_same/" + method
        valid = [r for r in repeat_rows if r["method"] == method and r["all_six_valid"]]
        values = [(templates[r["exact_text_group"]], r["cross_policy"]["correctness_crossings"] / 6 -
                   r["same_prompt"]["correctness_crossings"] / 3) for r in valid]
        selected = sum(r["method"] == method for r in repeat_rows)
        result[key] = checked(values, 6, 20260930, reference[key]["difference"], selected)
    return result


def run(root, output):
    if output.exists():
        raise ValueError("A fresh output directory is required")
    receipt = {}
    refs = {
        "factorial": root / "paper1/prepared/policy_factorial_v1/references.jsonl",
        "original": root / "paper1/prepared/hatecheck_test_formal_v1/references.jsonl",
    }
    for path in refs.values():
        receipt[str(path.relative_to(root))] = digest(path)
    templates = {name: {r["group"]: r["native_template_id"] for r in read_jsonl(path) if name == "original" or r["dataset"] == "hatecheck"}
                 for name, path in refs.items()}
    if len(templates["factorial"]) != 512 or len(templates["original"]) != 3728:
        raise ValueError("Frozen HateCheck population changed")
    result = {model: {"factorial": factorial(root, model, templates["factorial"], receipt),
                      "original": original(root, model, templates["original"], receipt)} for model in ("qwen", "granite")}
    output.mkdir(parents=True)
    (output / "intervals.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "input_sha256.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
