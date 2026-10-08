#!/usr/bin/env python3
"""Paired, denominator-preserving SBIC policy evaluation; outputs contain no text."""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import random

from prepare import (FIELDS, POLICIES, decision, demo_verdicts_enabled, digest, parse_decision, parse_json,
                     parse_output, request)
from generate import load_predictions, read_requests, sha

METHODS = ("direct", "guarded_direct", "full_two_stage", "freeze_only",
           "mask_only", "ap_commit", "neutral_compiler", "policy_compiler")
SECOND_STAGE = ("ap_commit", "full_two_stage", "freeze_only", "mask_only")
CONTRASTS = tuple(("ap_commit", control) for control in
                  ("direct", "guarded_direct", "full_two_stage", "freeze_only",
                   "mask_only", "neutral_compiler")) + (("neutral_compiler", "guarded_direct"),)
BOOTSTRAP_SEED = 20260929


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(rows):
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate artifact ID")
    return result


def fixed_config(config):
    return {key: value for key, value in config.items()
            if key not in ("input_sha256", "shard_index", "num_shards")}


def first_id(case_id, policy, method):
    exposed = method in ("full_two_stage", "freeze_only", "policy_compiler")
    return f"{case_id}/{policy if exposed else 'neutral'}/{ 'full_record' if exposed else 'neutral_record'}"


def load_bundle(prepared, stage1_paths, stage2, stage2_paths):
    """Rebuild requests/states, rather than trusting a self-asserted case ID."""
    manifest = json.loads((prepared / "preparation.json").read_text())
    demo_verdicts = demo_verdicts_enabled(manifest)
    if manifest.get("policies") != POLICIES:
        raise ValueError("Use the frozen corrected policies")
    cases = read_jsonl(prepared / "references.jsonl")
    bank = read_jsonl(prepared / "demonstrations.jsonl")
    indexed(cases)
    if not cases or len({c["group"] for c in cases}) != len(cases):
        raise ValueError("Expected one record per original exact-text group")
    for case in cases + bank:
        if (case["group"] != digest(case["text"]) or case["id"] != f"sbic:{case['group']}"
                or set(case["attributes"]) != set(FIELDS)
                or any(type(v) is not int or v not in (0, 1) for v in case["attributes"].values())):
            raise ValueError("Invalid complete native reference identity")
    patterns = [tuple(row["attributes"][field] for field in FIELDS) for row in bank]
    if patterns != [(0, 0, 0), (1, 1, 0), (1, 0, 0), (0, 0, 1)]:
        raise ValueError("Wrong frozen demonstration pattern bank")
    if ({c["group"] for c in cases} & {c["group"] for c in bank}
            or any(c["partition"] != manifest["partition"] for c in cases)
            or any(c["partition"] != "train" for c in bank)
            or manifest["selected_cases"] != len(cases)
            or manifest["demonstration_ids"] != [c["id"] for c in bank]):
        raise ValueError("Preparation population/bank mismatch")
    stage1_file = prepared / "stage1_requests.jsonl"
    expected1 = []
    for case in cases:
        expected1.append(request(case, bank, "neutral_record", demo_verdicts=demo_verdicts))
        for policy in POLICIES:
            expected1.extend(request(case, bank, method, policy, demo_verdicts=demo_verdicts)
                             for method in ("direct", "guarded_direct", "full_record"))
    actual1 = read_requests(stage1_file)
    if actual1 != expected1 or manifest["requests"] != len(actual1):
        raise ValueError("Stage1 requests do not reconstruct from frozen references/bank")
    pred1, config1, manifests1 = load_predictions(stage1_paths, stage1_file)
    second_manifest = json.loads((stage2 / "preparation.json").read_text())
    second_demo_verdicts = second_manifest.get("demo_verdicts", True)
    if type(second_demo_verdicts) is not bool or second_demo_verdicts != demo_verdicts:
        raise ValueError("Stage2 demonstration verdict condition changed")
    for key, path in (("stage1_requests_sha256", stage1_file),
                      ("references_sha256", prepared / "references.jsonl"),
                      ("demonstrations_sha256", prepared / "demonstrations.jsonl")):
        if second_manifest.get(key) != file_sha(path):
            raise ValueError(f"Stage2 preparation binding mismatch: {key}")
    if second_manifest.get("first_run_config") != config1:
        raise ValueError("Stage2 used a different first-stage run configuration")
    signatures = {m["run_signature"] for m in manifests1}
    if set(second_manifest.get("first_run_signatures", [])) != signatures:
        raise ValueError("Stage2 first-stage manifests differ")
    expected_states, expected2 = [], []
    for case in cases:
        for policy in POLICIES:
            for method in SECOND_STAGE:
                source_id = first_id(case["id"], policy, method)
                if source_id not in pred1:
                    raise ValueError("Cannot verify stage2 derivation without its first-stage output")
                source = pred1[source_id]
                parsed = parse_output(source.get("response")) if source.get("status") == "ok" else None
                attributes = parsed["attributes"] if parsed else None
                expected_states.append({"id": f"{case['id']}/{policy}/{method}", "first_id": source_id,
                                        "first_request_sha256": source["request_sha256"],
                                        "first_run_signature": source["run_signature"],
                                        "attributes": attributes, "first_stage_valid": parsed is not None})
                if parsed:
                    expected2.append(request(case, bank, method, policy, attributes, demo_verdicts=demo_verdicts))
    states = read_jsonl(stage2 / "states.jsonl")
    actual2 = read_jsonl(stage2 / "requests.jsonl")
    if states != expected_states or actual2 != expected2:
        raise ValueError("Stage2 states/requests are not the derived paired v2 intervention")
    invalid = sum(not row["first_stage_valid"] for row in states)
    if (second_manifest.get("stage") != 2 or second_manifest.get("expected_cases") != len(states)
            or second_manifest.get("requests") != len(actual2)
            or second_manifest.get("invalid_first_stage") != invalid):
        raise ValueError("Stage2 manifest counts do not match its states/requests")
    if actual2:
        pred2, config2, manifests2 = load_predictions(stage2_paths, stage2 / "requests.jsonl")
        if fixed_config(config1) != fixed_config(config2):
            raise ValueError("Model/seed/decoding/budgets changed between stages")
    else:
        if stage2_paths:
            raise ValueError("No second-stage requests exist; omit second-stage prediction paths")
        pred2, config2, manifests2 = {}, None, []
    provenance = {"preparation_sha256": file_sha(prepared / "preparation.json"),
                  "preparation_version": manifest["version"], "demo_verdicts": demo_verdicts,
                  "stage2_preparation_sha256": file_sha(stage2 / "preparation.json"),
                  "references_sha256": file_sha(prepared / "references.jsonl"),
                  "demonstrations_sha256": file_sha(prepared / "demonstrations.jsonl"),
                  "states_sha256": file_sha(stage2 / "states.jsonl"),
                  "stage1_requests_sha256": file_sha(stage1_file),
                  "stage2_requests_sha256": file_sha(stage2 / "requests.jsonl"),
                  "stage1_config": config1, "stage2_config": config2,
                  "stage1_run_signatures": sorted(signatures),
                  "stage2_run_signatures": sorted(m["run_signature"] for m in manifests2),
                  "prediction_files": [{"path": str(p), "sha256": file_sha(p),
                                        "manifest_sha256": file_sha(p.with_suffix(p.suffix + ".manifest.json"))}
                                       for p in list(stage1_paths) + list(stage2_paths)],
                  "expected_stage1": len(actual1), "observed_stage1": len(pred1),
                  "expected_stage2": len(actual2), "observed_stage2": len(pred2),
                  "first_stage_invalid_downstream_cases": invalid,
                  "all_second_stage_requests_suppressed": not actual2}
    return cases, pred1, pred2, indexed(states), provenance


def response_fields(result):
    text = result.get("response") if result.get("status") == "ok" else None
    parsed = parse_output(text)
    value = parse_json(text)
    attributes = value.get("attributes", {}) if value else {}
    attributes = attributes if isinstance(attributes, dict) else {}
    fields = {key: int(attributes[key]) if type(attributes.get(key)) in (int, bool)
              and attributes[key] in (0, 1) else None for key in FIELDS}
    return fields, parsed is not None, parse_decision(text)


def outcomes(cases, pred1, pred2, states, methods=METHODS):
    rows = []
    for case in cases:
        for policy in POLICIES:
            gold_decision = decision(case["attributes"], policy)
            for method in methods:
                key = f"{case['id']}/{policy}/{method}"
                dependencies, first_status = [], None
                if method in ("direct", "guarded_direct"):
                    result = pred1.get(key, {})
                    fields, valid, verdict = response_fields(result)
                    dependencies.append((1, key))
                    status = result.get("status", "missing")
                else:
                    source_id = first_id(case["id"], policy, method)
                    source = pred1.get(source_id, {})
                    dependencies.append((1, source_id))
                    parsed = parse_output(source.get("response")) if source.get("status") == "ok" else None
                    first_status = ("valid" if parsed else "invalid_format" if source.get("status") == "ok"
                                    else source.get("status", "missing"))
                    fields, valid, verdict = dict.fromkeys(FIELDS), False, None
                    if not parsed:
                        status = "first_stage_failure"
                    elif method.endswith("_compiler"):
                        fields, valid = parsed["attributes"], True
                        verdict, status = decision(fields, policy), "compiled"
                    else:
                        if states[key]["attributes"] != parsed["attributes"]:
                            raise ValueError("State does not match first-stage record")
                        dependencies.append((2, key))
                        result = pred2.get(key, {})
                        fields, valid, verdict = response_fields(result)
                        status = result.get("status", "missing")
                        if method in ("ap_commit", "freeze_only"):
                            fields, valid = parsed["attributes"], True
                exact = valid and fields == case["attributes"]
                verdict_valid = verdict is not None
                verdict_correct = verdict_valid and verdict == gold_decision
                rule_valid = valid and verdict_valid
                rows.append({"id": key, "case_id": case["id"], "exact_text_group": case["group"],
                             "partition": case["partition"], "policy": policy, "method": method,
                             "reference_attributes": case["attributes"], "reference_decision": gold_decision,
                             "predicted_attributes": fields, "predicted_decision": verdict,
                             "attributes_valid": valid, "decision_valid": verdict_valid,
                             "attributes_exact": int(exact), "verdict_correct": int(verdict_correct),
                             "joint_correct": int(exact and verdict_correct),
                             "field_correct": {f: int(fields[f] is not None and fields[f] == case["attributes"][f])
                                               for f in FIELDS},
                             "rule_consistency_eligible": rule_valid,
                             "rule_consistent": bool(rule_valid and verdict == decision(fields, policy)),
                             "output_status": status, "first_stage_status": first_status,
                             "dependencies": [{"stage": stage, "id": identity} for stage, identity in dependencies]})
    return rows


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics(rows):
    n = len(rows)
    valid_rule = sum(r["rule_consistency_eligible"] for r in rows)
    consistent = sum(r["rule_consistent"] for r in rows)
    return {"n": n, "exact_text_groups": len({r["exact_text_group"] for r in rows}),
            "attributes_exact_accuracy": fraction(sum(r["attributes_exact"] for r in rows), n),
            "verdict_accuracy": fraction(sum(r["verdict_correct"] for r in rows), n),
            "joint_accuracy": fraction(sum(r["joint_correct"] for r in rows), n),
            "field_accuracy": {f: fraction(sum(r["field_correct"][f] for r in rows), n) for f in FIELDS},
            "attribute_coverage": fraction(sum(r["attributes_valid"] for r in rows), n),
            "decision_coverage": fraction(sum(r["decision_valid"] for r in rows), n),
            "field_coverage": {f: fraction(sum(r["predicted_attributes"][f] is not None for r in rows), n)
                               for f in FIELDS},
            "joint_coverage": fraction(valid_rule, n),
            "invalid_attribute_outputs_with_ok_status": sum(r["output_status"] == "ok" and not r["attributes_valid"]
                                                            for r in rows),
            "invalid_decision_outputs_with_ok_status": sum(r["output_status"] == "ok" and not r["decision_valid"]
                                                           for r in rows),
            "rule_consistency": {"valid_pairs": valid_rule, "consistent": consistent,
                                 "rate_on_valid_pairs": fraction(consistent, valid_rule),
                                 "valid_pair_coverage": fraction(valid_rule, n)},
            "output_status_counts": dict(Counter(r["output_status"] for r in rows)),
            "first_stage_status_counts": dict(Counter(r["first_stage_status"] for r in rows
                                                       if r["first_stage_status"] is not None))}


def case_metrics(rows):
    groups = defaultdict(dict)
    for row in rows:
        group = groups[(row["exact_text_group"], row["method"])]
        if row["policy"] in group:
            raise ValueError("Repeated case-policy-method")
        group[row["policy"]] = row
    result = []
    for (identity, method), policies in sorted(groups.items()):
        if set(policies) != set(POLICIES):
            raise ValueError("Missing policy in paired population")
        items = list(policies.values())
        pairs = [(a, b) for a, b in combinations(items, 2) if a["attributes_valid"] and b["attributes_valid"]]
        result.append({"case_id": items[0]["case_id"], "exact_text_group": identity, "method": method,
                       "all_policy_joint_correct": int(all(r["joint_correct"] for r in items)),
                       "all_policy_attributes_valid": all(r["attributes_valid"] for r in items),
                       "mean_policy_joint_correct": sum(r["joint_correct"] for r in items) / len(items),
                       "attribute_pair_count": len(pairs), "possible_attribute_pairs": 3,
                       "attribute_pair_flips": sum(a["predicted_attributes"] != b["predicted_attributes"] for a, b in pairs),
                       "field_pair_flips": {f: sum(a["predicted_attributes"][f] != b["predicted_attributes"][f]
                                                   for a, b in pairs) for f in FIELDS},
                       "any_attribute_flip": any(a["predicted_attributes"] != b["predicted_attributes"] for a, b in pairs)})
    return result


def paired_effect(rows, treatment, control, endpoint="joint_correct", policy=None,
                  replicates=2000, seed=BOOTSTRAP_SEED):
    if type(replicates) is not int or replicates < 1:
        raise ValueError("Positive bootstrap replicates required")
    pairs = defaultdict(dict)
    for row in rows:
        if row["method"] in (treatment, control) and (policy is None or row.get("policy") == policy):
            key = (row["exact_text_group"], row.get("policy"))
            if row["method"] in pairs[key]:
                raise ValueError("Repeated paired outcome")
            pairs[key][row["method"]] = row
    differences = defaultdict(list)
    for (group, _), pair in pairs.items():
        if set(pair) != {treatment, control}:
            raise ValueError("Incomplete paired population")
        differences[group].append(pair[treatment][endpoint] - pair[control][endpoint])
    totals = [(sum(values), len(values)) for _, values in sorted(differences.items())]
    if not totals:
        return {"treatment": treatment, "control": control, "endpoint": endpoint, "policy": policy,
                "resampling_groups": 0, "paired_rows": 0, "difference": None, "ci95": None}
    n = sum(count for _, count in totals)
    rng, samples = random.Random(seed), []
    for _ in range(replicates):
        sample = [totals[rng.randrange(len(totals))] for _ in totals]
        samples.append(sum(value for value, _ in sample) / sum(count for _, count in sample))
    samples.sort()
    return {"treatment": treatment, "control": control, "endpoint": endpoint, "policy": policy,
            "resampling_groups": len(totals), "paired_rows": n,
            "difference": sum(value for value, _ in totals) / n,
            "ci95": [samples[int(.025 * (replicates - 1))], samples[int(.975 * (replicates - 1))]],
            "bootstrap_replicates": replicates, "bootstrap_seed": seed,
            "interval": "paired exact-text-group percentile; pointwise, not multiplicity-adjusted"}


def logical_cost(dependencies, predictions):
    results = [predictions.get((d["stage"], d["id"])) for d in dependencies]
    return {"logical_calls": len(results), "observed_calls": sum(r is not None for r in results),
            "missing_calls": sum(r is None for r in results),
            "input_too_long_calls": sum(r is not None and r["status"] == "input_too_long" for r in results),
            "logical_input_tokens": sum(r["input_tokens"] for r in results if r is not None),
            "logical_output_tokens": sum(r["output_tokens"] for r in results if r is not None)}


def costs(rows, pred1, pred2, methods=METHODS):
    predictions = {(stage, key): row for stage, source in ((1, pred1), (2, pred2)) for key, row in source.items()}
    for row in predictions.values():
        for key in ("input_tokens", "output_tokens", "charged_input_tokens", "charged_output_tokens", "batch_size"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f"Missing/invalid recorded cost: {key}")
        if (type(row.get("cached")) is not bool or row["status"] not in ("ok", "input_too_long")
                or not isinstance(row.get("elapsed_batch_seconds"), (int, float))
                or not math.isfinite(row["elapsed_batch_seconds"]) or row["elapsed_batch_seconds"] < 0):
            raise ValueError("Invalid generation status/cache/time accounting")
        if ((row["cached"] or row["status"] != "ok")
                and (row["charged_input_tokens"] or row["charged_output_tokens"] or row["batch_size"]
                     or row["elapsed_batch_seconds"])):
            raise ValueError("Cached/rejected request improperly charged")
        if row["status"] == "ok" and not row["cached"] and row["batch_size"] < 1:
            raise ValueError("Physical generation requires a recorded batch size")
    by_arm = {}
    for method in methods:
        members = [r for r in rows if r["method"] == method]
        dependencies = [dep for row in members for dep in row["dependencies"]]
        unique = {(d["stage"], d["id"]): d for d in dependencies}
        by_arm[method] = {"standalone_per_policy_sum": logical_cost(dependencies, predictions),
                          "shared_across_three_policies": logical_cost(list(unique.values()), predictions),
                          "standalone_by_policy": {policy: logical_cost([dep for row in members if row["policy"] == policy
                                                                          for dep in row["dependencies"]], predictions)
                                                   for policy in POLICIES}}
    return {"arms": by_arm,
            "physical_run": {"observed_requests": len(predictions),
                             "cached_requests": sum(r["cached"] for r in predictions.values()),
                             "physical_generated_requests": sum(r["status"] == "ok" and not r["cached"] for r in predictions.values()),
                             "charged_input_tokens": sum(r["charged_input_tokens"] for r in predictions.values()),
                             "charged_output_tokens": sum(r["charged_output_tokens"] for r in predictions.values()),
                             "generation_batch_seconds": sum(r["elapsed_batch_seconds"] / r["batch_size"]
                                                             for r in predictions.values() if r["batch_size"] > 0)},
            "notes": ["Logical inputs include recorded lengths of rejected requests; these were not processed.",
                      "Missing output tokens are unknown, not measured zero; consult missing_calls.",
                      "Standalone cost repeats a needed first-stage extraction for each policy; shared cost reuses it.",
                      "Physical costs count each supplied run output once across all arms; cache order is not a method advantage.",
                      "Batch time excludes loading/host work and is not total wall time; partial output sets underestimate it.",
                      "Charges describe persisted run records; interrupted work lost before logging is not recoverable here."]}


def summarize(rows, pred1, pred2, replicates=2000):
    per_case = case_metrics(rows)
    cells = {f"{method}/{policy}": metrics([r for r in rows if r["method"] == method and r["policy"] == policy])
             for method in METHODS for policy in POLICIES}
    cross = {}
    for method in METHODS:
        members = [r for r in per_case if r["method"] == method]
        valid_pairs = sum(r["attribute_pair_count"] for r in members)
        possible = sum(r["possible_attribute_pairs"] for r in members)
        cross[method] = {"n_cases": len(members),
                         "all_policy_joint_accuracy": fraction(sum(r["all_policy_joint_correct"] for r in members), len(members)),
                         "all_policy_attribute_coverage": fraction(sum(r["all_policy_attributes_valid"] for r in members), len(members)),
                         "valid_attribute_pairs": valid_pairs, "possible_attribute_pairs": possible,
                         "pair_coverage": fraction(valid_pairs, possible),
                         "attribute_flips_on_valid_pairs": sum(r["attribute_pair_flips"] for r in members),
                         "attribute_flip_rate_on_valid_pairs": fraction(sum(r["attribute_pair_flips"] for r in members), valid_pairs),
                         "cases_with_observed_attribute_flip": sum(r["any_attribute_flip"] for r in members),
                         "field_flips_on_valid_pairs": {f: sum(r["field_pair_flips"][f] for r in members) for f in FIELDS}}
    strata = {}
    for left, right in combinations(POLICIES, 2):
        for flip in (False, True):
            subset = [r for r in rows if r["policy"] in (left, right)
                      and (decision(r["reference_attributes"], left) != decision(r["reference_attributes"], right)) == flip]
            strata[f"{left}__{right}/reference_{'flip' if flip else 'same'}"] = {
                "n_cases": len({r["case_id"] for r in subset}),
                "cells": {f"{m}/{p}": metrics([r for r in subset if r["method"] == m and r["policy"] == p])
                          for m in METHODS for p in (left, right)}}
    contrasts = []
    for treatment, control in CONTRASTS:
        for endpoint in ("attributes_exact", "verdict_correct", "joint_correct"):
            contrasts.extend(paired_effect(rows, treatment, control, endpoint, policy, replicates)
                             for policy in (None, *POLICIES))
        contrasts.append(paired_effect(per_case, treatment, control, "all_policy_joint_correct", replicates=replicates))
    return {"resampling_exact_text_groups": len({r["exact_text_group"] for r in rows}),
            "cells": cells, "cross_policy": cross, "reference_rule_flip_strata": strata,
            "paired_effects": contrasts, "cost": costs(rows, pred1, pred2)}, per_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--stage1-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--stage2-predictions", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True, help="Fresh result directory")
    parser.add_argument("--stage", choices=("pilot", "evaluation"), required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists() or args.bootstrap < 1:
        parser.error("Use a fresh output directory and positive bootstrap count")
    cases, pred1, pred2, states, provenance = load_bundle(args.prepared, args.stage1_predictions,
                                                        args.stage2, args.stage2_predictions)
    rows = outcomes(cases, pred1, pred2, states)
    report, case_rows = summarize(rows, pred1, pred2, args.bootstrap)
    prediction_index = {(stage, key): row for stage, source in ((1, pred1), (2, pred2))
                        for key, row in source.items()}
    for row in rows:
        row["logical_cost"] = logical_cost(row["dependencies"], prediction_index)
    report.update(stage=args.stage, provenance=provenance, code_sha256=file_sha(Path(__file__)),
                  evidence_type="existing_native_attribute_judgments_and_derived_rule_labels",
                  new_human_annotations=0, independent_policy_gold=False,
                  parser="Schema-valid JSON substring; each attribute field and decision parsed separately; no repair.",
                  interpretation=["Committed-record stability is mechanical, not learned understanding.",
                                  "Compiler rule consistency is constructed; its joint accuracy equals exact record accuracy.",
                                  "Bootstrap intervals are descriptive pointwise intervals, not a multiplicity-adjusted discovery test."])
    args.output.mkdir(parents=True)
    for filename, records in (("outcomes.jsonl", rows), ("case_outcomes.jsonl", case_rows)):
        with (args.output / filename).open("x", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
    report["outcomes_sha256"] = file_sha(args.output / "outcomes.jsonl")
    report["case_outcomes_sha256"] = file_sha(args.output / "case_outcomes.jsonl")
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "cases": len(cases), "scored_rows": len(rows)}))


if __name__ == "__main__":
    main()
