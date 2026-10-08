"""Descriptive rule-pair/field decomposition of existing truth-table crossings.

Run after the frozen factorial evaluation; no generation, new labels, or tests
of significance. The smallest exact-text hash selects the Qwen/HateCheck example.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

from policy_factorial import task


def crossing(left, right, gold, decision, used_fields):
    a, b = left["predicted_attributes"], right["predicted_attributes"]
    if (a == gold) == (b == gold):
        return None
    wrong = right if a == gold else left
    fields = sorted(f for f in gold if wrong["predicted_attributes"][f] != gold[f])
    rule = wrong["policy"]
    return {"wrong_rule": rule, "wrong_fields": fields,
            "off_rule_only": not set(fields).intersection(used_fields[rule]),
            "wrong_side_verdict_correct": decision(wrong["predicted_attributes"], rule) == decision(gold, rule)}


def self_test():
    gold = {"offensive": 1, "intent": 0, "sexual": 0}
    _, _, _, decision, _ = task("sbic")
    used = {"offense": {"offensive"}, "sexual": {"sexual"}, "intentional": {"offensive", "intent"}}
    correct = {"policy": "sexual", "predicted_attributes": gold}
    wrong = {"policy": "offense", "predicted_attributes": {**gold, "sexual": 1}}
    result = crossing(correct, wrong, gold, decision, used)
    assert result["wrong_fields"] == ["sexual"] and result["off_rule_only"] and result["wrong_side_verdict_correct"]
    wrong = {"policy": "intentional", "predicted_attributes": {**gold, "offensive": 0}}
    result = crossing(correct, wrong, gold, decision, used)
    assert not result["off_rule_only"] and result["wrong_side_verdict_correct"]  # AND masks this field error.
    assert crossing(wrong, wrong, gold, decision, used) is None
    assert crossing(correct, correct, gold, decision, used) is None


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def analyze(root):
    reference_path = root / "paper1/prepared/policy_factorial_v1/references.jsonl"
    references = {(r["dataset"], r["group"]): r for r in read_rows(reference_path)}
    if len(references) != 1024:
        raise ValueError("Expected the frozen 512-group population in each native task")
    report, inputs = {}, {str(reference_path.relative_to(root)): sha256(reference_path.read_bytes()).hexdigest()}
    supports = {"sbic": {"offense": {"offensive"}, "sexual": {"sexual"}, "intentional": {"offensive", "intent"}},
                "hatecheck": {"all_hate": {"hateful"}, "directed_hate": {"directed_hate"}, "general_hate": {"hateful", "directed_hate"}}}
    for model in ("qwen", "granite"):
        folder = root / f"paper1/results/{model}_policy_factorial_20261002_v1_evaluation"
        outcome_path, summary_path = folder / "outcomes.jsonl", folder / "summary.json"
        for path in (outcome_path, summary_path):
            inputs[str(path.relative_to(root))] = sha256(path.read_bytes()).hexdigest()
        summaries = json.loads(summary_path.read_text())["datasets"]
        for dataset in ("sbic", "hatecheck"):
            fields, policies, _, decision, _ = task(dataset)
            groups = defaultdict(dict)
            for row in read_rows(outcome_path):
                if row["dataset"] != dataset or row["renderer"] != "truth_table":
                    continue
                key = row["realization"], row["policy"]
                if key in groups[row["exact_text_group"]]:
                    raise ValueError("Duplicate truth-table cell")
                groups[row["exact_text_group"]][key] = row
            if len(groups) != 512:
                raise ValueError("Wrong selected population")
            pair_counts, field_sets, wrong_rules = Counter(), Counter(), Counter()
            off_rule = verdict_correct = fresh = complete = 0
            crossing_groups, example = [], None
            for group, cells in sorted(groups.items()):
                if set(cells) != {(i, p) for i in (0, 1) for p in policies}:
                    raise ValueError("Missing truth-table cell")
                if not all(row["valid"] for row in cells.values()):
                    continue
                complete += 1
                gold = references[dataset, group]["attributes"]
                if any(set(row["predicted_attributes"]) != set(fields) or row["correct"] != (row["predicted_attributes"] == gold) for row in cells.values()):
                    raise ValueError("Correctness/reference mismatch")
                fresh += sum(cells[0, p]["correct"] != cells[1, p]["correct"] for p in policies)
                has_crossing = False
                for i in (0, 1):
                    for p, q in combinations(policies, 2):
                        event = crossing(cells[i, p], cells[i, q], gold, decision, supports[dataset])
                        if event is None:
                            continue
                        has_crossing = True
                        pair_counts[p + "/" + q] += 1
                        field_sets["/".join(event["wrong_fields"])] += 1
                        wrong_rules[event["wrong_rule"]] += 1
                        off_rule += event["off_rule_only"]
                        verdict_correct += event["wrong_side_verdict_correct"]
                if has_crossing:
                    crossing_groups.append(group)
                    if model == "qwen" and dataset == "hatecheck" and example is None:
                        ref = references[dataset, group]
                        example = {"selection": "smallest exact-original-text SHA-256 among Qwen/HateCheck complete truth-table crossing groups",
                                   "group": group, "source_record": ref.get("source_record"), "text": ref["text"], "gold": gold,
                                   "outputs": [{"realization": i, "policy": p, "attributes": cells[i, p]["predicted_attributes"],
                                                "attribute_correct": bool(cells[i, p]["correct"]),
                                                "verdict_correct": decision(cells[i, p]["predicted_attributes"], p) == decision(gold, p)}
                                               for i in (0, 1) for p in policies]}
            total = sum(pair_counts.values())
            expected = summaries[dataset]["renderers"]["truth_table"]
            if complete != expected["complete"] or abs((total / 6 - fresh / 3) / complete - expected["correctness_cross_minus_fresh"]["mean"]) > 1e-12:
                raise ValueError("Decomposition does not recover the frozen primary point")
            if sum(field_sets.values()) != total or sum(wrong_rules.values()) != total or not 0 <= off_rule <= verdict_correct <= total:
                raise ValueError("Invalid event accounting")
            report[model + "/" + dataset] = {"selected": 512, "complete": complete, "possible_cross_rule_pairs": 6 * complete,
                "crossings": total, "fresh_crossings": fresh, "crossing_groups": len(crossing_groups),
                "rule_pair_counts": dict(pair_counts), "wrong_field_sets": dict(field_sets), "wrong_rule_counts": dict(wrong_rules),
                "off_rule_only": off_rule, "wrong_side_verdict_correct": verdict_correct,
                "count_unit": "within-realization rule-pair opportunity, not independent posts", "example": example}
    return {"scope": "post-review descriptive decomposition; complete-valid truth-table panels; no significance family or internal causal inference",
            "input_sha256": inputs, "results": report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("crossing structure self-check passed")
    else:
        if args.root is None or args.output is None:
            parser.error("--root and --output are required")
        args.output.mkdir(parents=True, exist_ok=False)
        result = analyze(args.root.resolve())
        (args.output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: {x: v[x] for x in ("complete", "crossings", "crossing_groups", "off_rule_only", "wrong_side_verdict_correct")} for k, v in result["results"].items()}))
