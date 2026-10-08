#!/usr/bin/env python3
"""Frozen SBIC measurement study; legacy pilot defaults remain unchanged."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys

import evaluate as ev
import prepare as pr
from generate import load_predictions, read_requests, sha

VERSION = "sbic_measurement_v1"
METHODS = ("direct", "guarded_direct", "full_two_stage", "policy_compiler", "neutral_compiler")
ARMS = (*METHODS, "training_constant")
EXPOSED = ("direct", "guarded_direct", "full_record")
REPEAT_N = 64
BOOTSTRAP = 10000
SEED = 20260930
RUNNER_SHA = "f81bc57760070dbef65aaf60fa71ce3c70aa71f345224b7a4932e493f170ef6c"
MODEL_REVISIONS = {"a09a35458c702b33eeacc393d103063234e8bc28",
                   "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"}
HERE = Path(__file__).resolve().parent


def json_write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def source_binding():
    return {name: ev.file_sha(HERE / name) for name in
            ("formal.py", "hatecheck_task.py", "prepare.py", "evaluate.py", "FORMAL_PROTOCOL_V1.md")}


def stage1_requests(cases, bank):
    # Prefix length 64*3*3=576 is divisible by four shards times eight rows/batch.
    repeat_cases = sorted(cases, key=lambda c: pr.digest("paper1-repeat-v1:" + c["group"]))[:REPEAT_N]
    repeated = [pr.request(c, bank, m, p, demo_verdicts=False)
                for c in repeat_cases for p in pr.POLICIES for m in EXPOSED]
    repeat_ids = {r["id"] for r in repeated}
    rest = []
    for case in cases:
        rest.append(pr.request(case, bank, "neutral_record", demo_verdicts=False))
        rest.extend(pr.request(case, bank, m, p, demo_verdicts=False)
                    for p in pr.POLICIES for m in EXPOSED
                    if f"{case['id']}/{p}/{m}" not in repeat_ids)
    return repeated + rest, repeated


def prepare(args):
    if args.output.exists():
        raise ValueError("Use a fresh versioned preparation directory")
    parts, sources = {}, {}
    for partition, suffix in (("train", "trn"), ("dev", "dev"), ("test", "tst")):
        source = args.source / f"SBIC.v2.agg.{suffix}.csv"
        parts[partition], sources[partition] = pr.load_aggregate(source, partition)
        # Released files have no empty texts; refuse rather than shift source row identities.
        if sources[partition]["excluded"].get("empty_text", 0):
            raise ValueError("Unexpected empty aggregate text; audit source record mapping before proceeding")
        for ordinal, row in enumerate(parts[partition], 1):
            row["source_record"] = {"file": source.name, "data_row_1based": ordinal}
    heldout = {r["group"] for r in parts["dev"] + parts["test"]}
    train = [r for r in parts["train"] if r["group"] not in heldout
             and all(v is not None for v in r["attributes"].values())]
    bank = pr.demonstrations(train)
    counts = Counter(tuple(r["attributes"][f] for f in pr.FIELDS) for r in train)
    modal = min(counts, key=lambda a: (-counts[a], a))
    constant = {"attributes": dict(zip(pr.FIELDS, modal)), "eligible_training_records": len(train),
                "pattern_counts": {"".join(map(str, a)): n for a, n in sorted(counts.items())},
                "selection": "most frequent complete training vector; lexicographically smallest tie"}
    dev_groups = {r["group"] for r in parts["dev"]}
    cases, audit = [], []
    for row in parts["test"]:
        resolution = {f: "missing" if row["annotation_means"][f] is None else
                      "tie" if row["attributes"][f] is None else "resolved" for f in pr.FIELDS}
        complete = all(v == "resolved" for v in resolution.values())
        overlap = row["group"] in dev_groups
        audit.append({"id": row["id"], "exact_text_group": row["group"], "partition": "test",
                      "source_record": row["source_record"], "character_length": len(row["text"]),
                      "resolution": resolution, "dev_exact_overlap": overlap, "eligible": complete and not overlap})
        if complete and not overlap:
            cases.append(row)
    cases.sort(key=lambda c: pr.digest("paper1-case-v1:" + c["group"]))
    if len(cases) < REPEAT_N:
        raise ValueError("Formal population cannot supply the frozen 64-group repeat panel")
    save_prepared(args.output, cases, bank, audit, constant, sources, "sbic")


def save_prepared(output, cases, bank, audit, constant, sources, dataset):
    requests, repeated = stage1_requests(cases, bank)
    output.mkdir(parents=True)
    for filename, rows in (("references.jsonl", cases), ("demonstrations.jsonl", bank),
                           ("resolution_log.jsonl", audit), ("stage1_requests.jsonl", requests),
                           ("repeat_requests.jsonl", repeated)):
        pr.write_rows(output / filename, rows)
    json_write(output / "training_constant.json", constant)
    files = {p.name: ev.file_sha(p) for p in output.iterdir() if p.is_file()}
    manifest = {"version": VERSION, "dataset": dataset, "partition": "test", "demo_verdicts": False,
                "selected_cases": len(cases), "sources": sources, "policies": pr.POLICIES,
                "methods": ARMS, "repeat_groups": REPEAT_N,
                "requests": len(requests), "repeat_requests": len(repeated),
                "stage2_request_ceiling": 3 * len(cases),
                "total_request_ceiling_per_model": 13 * len(cases) + len(repeated),
                "demonstration_ids": [r["id"] for r in bank], "files": files,
                "source_binding": source_binding(), "runner_sha256": RUNNER_SHA,
                "new_human_annotations": 0, "actual_design_route": "GPT-6-Astra/xhigh"}
    json_write(output / "preparation.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("selected_cases", "requests", "repeat_requests",
                                               "total_request_ceiling_per_model")}))


def load_prepared(path):
    manifest = json.loads((path / "preparation.json").read_text())
    if (manifest["version"] != VERSION or manifest["source_binding"] != source_binding()
            or manifest["runner_sha256"] != RUNNER_SHA or manifest["policies"] != pr.POLICIES
            or manifest["demo_verdicts"] is not False or manifest["partition"] != "test"):
        raise ValueError("Frozen protocol/source binding changed")
    for name, expected in manifest["files"].items():
        if ev.file_sha(path / name) != expected:
            raise ValueError(f"Frozen prepared artifact changed: {name}")
    cases, bank = [ev.read_jsonl(path / name) for name in ("references.jsonl", "demonstrations.jsonl")]
    for row in cases + bank:
        if (row["group"] != pr.digest(row["text"]) or row["id"] != manifest["dataset"] + ":" + row["group"]
                or set(row["attributes"]) != set(pr.FIELDS)
                or any(type(v) is not int or v not in (0, 1) for v in row["attributes"].values())):
            raise ValueError("Invalid exact original text/reference identity")
    if (len(ev.indexed(cases)) != manifest["selected_cases"] or len(cases) < REPEAT_N
            or any(c["partition"] != "test" for c in cases)
            or any(c["partition"] != "train" for c in bank)
            or {c["group"] for c in cases} & {c["group"] for c in bank}
            or [c["id"] for c in bank] != manifest["demonstration_ids"]
            or [tuple(c["attributes"][f] for f in pr.FIELDS) for c in bank]
            != ([(0, 0, 0), (1, 1, 0), (1, 0, 0), (0, 0, 1)] if manifest["dataset"] == "sbic" else [])):
        raise ValueError("Frozen population/demonstration bank mismatch")
    requests, repeated = stage1_requests(cases, bank)
    if (ev.read_jsonl(path / "stage1_requests.jsonl") != requests
            or ev.read_jsonl(path / "repeat_requests.jsonl") != repeated):
        raise ValueError("Requests do not reconstruct from frozen references and bank")
    return manifest, cases, bank, json.loads((path / "training_constant.json").read_text())


def checked_run(paths, request_file):
    expected = ev.read_jsonl(request_file)
    if not expected:
        if paths:
            raise ValueError("Prediction files supplied for empty request set")
        return {}, None, []
    predictions, config, manifests = load_predictions(paths, request_file)
    fixed = {"code_sha256": RUNNER_SHA, "num_shards": 4, "batch_size": 8, "seed": 17,
             "max_input_tokens": 8192, "max_new_tokens": 128, "do_sample": False,
             "dtype": "bfloat16", "attention": "sdpa", "thinking": False, "tokenizer_use_fast": False}
    if (any(config.get(k) != v for k, v in fixed.items())
            or config.get("model_revision") not in MODEL_REVISIONS
            or {m["config"]["shard_index"] for m in manifests} != set(range(4))
            or len(manifests) != 4 or set(predictions) != {r["id"] for r in expected}):
        raise ValueError("Incomplete or nonfrozen model/decoding/shard configuration")
    by_signature = {m["run_signature"]: m for m in manifests}
    for index, req in enumerate(expected):
        output = predictions[req["id"]]
        if by_signature[output["run_signature"]]["config"]["shard_index"] != index % 4:
            raise ValueError("Output assigned to wrong frozen shard")
    for m in manifests:
        size = len(expected[m["config"]["shard_index"]::4])
        if m.get("total_requests") != size or m.get("completed") != size:
            raise ValueError("Completed manifest counts do not match shard")
    return predictions, config, manifests


def second_artifacts(cases, bank, predictions):
    requests, states = [], []
    for case in cases:
        for policy in pr.POLICIES:
            first_id = f"{case['id']}/{policy}/full_record"
            pred = predictions[first_id]
            parsed = pr.parse_output(pred.get("response")) if pred.get("status") == "ok" else None
            attributes = parsed["attributes"] if parsed else None
            states.append({"id": f"{case['id']}/{policy}/full_two_stage", "first_id": first_id,
                           "first_request_sha256": pred["request_sha256"],
                           "first_run_signature": pred["run_signature"], "attributes": attributes,
                           "first_stage_valid": parsed is not None})
            if parsed:
                requests.append(pr.request(case, bank, "full_two_stage", policy, attributes, demo_verdicts=False))
    return requests, states


def stage2(args):
    if args.output.exists():
        raise ValueError("Use a fresh second-stage directory")
    _, cases, bank, _ = load_prepared(args.prepared)
    pred, config, manifests = checked_run(args.predictions, args.prepared / "stage1_requests.jsonl")
    requests, states = second_artifacts(cases, bank, pred)
    args.output.mkdir(parents=True)
    pr.write_rows(args.output / "requests.jsonl", requests)
    pr.write_rows(args.output / "states.jsonl", states)
    json_write(args.output / "preparation.json", {
        "version": VERSION, "stage": 2, "first_run_config": config,
        "first_run_signatures": sorted(m["run_signature"] for m in manifests),
        "prepared_manifest_sha256": ev.file_sha(args.prepared / "preparation.json"),
        "requests": len(requests), "invalid_first_stage": len(states) - len(requests)})
    print(json.dumps({"requests": len(requests), "invalid_first_stage": len(states) - len(requests)}))


def interval(values, family=1, replicates=BOOTSTRAP):
    """Exact-text group paired bootstrap; family size sets Bonferroni interval tails."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"groups": 0, "difference": None, "ci95_pointwise": None, "ci_familywise": None}
    rng = np.random.default_rng(SEED)
    samples = []
    for start in range(0, replicates, 128):
        samples.extend(values[rng.integers(len(values), size=(min(128, replicates - start), len(values)))].mean(axis=1))
    return {"groups": len(values), "difference": float(values.mean()),
            "ci95_pointwise": np.quantile(samples, [.025, .975]).tolist(),
            "ci_familywise": np.quantile(samples, [.025 / family, 1 - .025 / family]).tolist(),
            "family_size": family, "replicates": replicates, "seed": SEED,
            "interval": "paired exact-original-text percentile bootstrap, Bonferroni tails"}


def constant_rows(rows, attributes):
    output = []
    for row in rows:
        if row["method"] != "neutral_compiler":
            continue
        exact = int(attributes == row["reference_attributes"])
        verdict = pr.decision(attributes, row["policy"])
        correct = int(verdict == row["reference_decision"])
        output.append({**row, "id": row["id"].rsplit("/", 1)[0] + "/training_constant",
                       "method": "training_constant", "predicted_attributes": attributes,
                       "predicted_decision": verdict, "attributes_valid": True, "decision_valid": True,
                       "attributes_exact": exact, "verdict_correct": correct, "joint_correct": exact,
                       "field_correct": {f: int(attributes[f] == row["reference_attributes"][f]) for f in pr.FIELDS},
                       "rule_consistency_eligible": True, "rule_consistent": True,
                       "output_status": "constant_compiled", "first_stage_status": None, "dependencies": []})
    return output


def repeat_comparison(cases, first, repeated, replicates=BOOTSTRAP):
    selected = {r["metadata"]["case_id"] for r in repeated.values()}
    case_rows, report = [], {}
    for method in EXPOSED:
        for case in cases:
            if case["id"] not in selected:
                continue
            values = [[ev.response_fields(source[f"{case['id']}/{policy}/{method}"])
                       for policy in pr.POLICIES] for source in (first, repeated)]
            cross = [pair for realization in values for pair in combinations(realization, 2)]
            within = list(zip(*values))
            row = {"exact_text_group": case["group"], "method": method,
                   "all_six_valid": all(v[1] for realization in values for v in realization)}
            for label, pairs in (("cross_policy", cross), ("same_prompt", within)):
                valid = [(a, b) for a, b in pairs if a[1] and b[1]]
                row[label] = {"possible_pairs": len(pairs), "valid_pairs": len(valid),
                              "attribute_disagreements": sum(a[0] != b[0] for a, b in valid),
                              "correctness_crossings": sum((a[0] == case["attributes"]) !=
                                                            (b[0] == case["attributes"]) for a, b in valid)}
            case_rows.append(row)
        panel = [r for r in case_rows if r["method"] == method]
        complete = [r for r in panel if r["all_six_valid"]]
        summary = {"selected_groups": len(panel), "all_six_valid_groups": len(complete),
                   "complete_panel_coverage": ev.fraction(len(complete), len(panel))}
        for label in ("cross_policy", "same_prompt"):
            totals = {key: sum(r[label][key] for r in panel) for key in panel[0][label]}
            for event in ("attribute_disagreements", "correctness_crossings"):
                totals[event + "_rate_valid_pairs"] = ev.fraction(totals[event], totals["valid_pairs"])
                totals[event + "_rate_all_pairs"] = ev.fraction(totals[event], totals["possible_pairs"])
            summary[label] = totals
        for event in ("attribute_disagreements", "correctness_crossings"):
            summary[event + "_cross_minus_repeat"] = interval(
                [r["cross_policy"][event] / 6 - r["same_prompt"][event] / 3 for r in complete],
                family=6 if event == "correctness_crossings" else 1, replicates=replicates)
        report[method] = summary
    return report, case_rows


def summarize(rows, replicates=BOOTSTRAP):
    per_case = ev.case_metrics(rows)
    lookup = {(r["exact_text_group"], r["method"]): r for r in per_case}
    groups = sorted({r["exact_text_group"] for r in rows})
    cells = {f"{m}/{p}": ev.metrics([r for r in rows if r["method"] == m and r["policy"] == p])
             for m in ARMS for p in pr.POLICIES}
    effects = {}
    policy_effects = {}
    paired_rows = {(r["exact_text_group"], r["method"], r["policy"]): r for r in rows}
    for control in (m for m in ARMS if m != "neutral_compiler"):
        family = 2 if control == "policy_compiler" else 2 * (len(ARMS) - 2)
        key = "neutral_compiler_minus_" + control
        effects[key] = interval([lookup[g, "neutral_compiler"]["mean_policy_joint_correct"] -
                                 lookup[g, control]["mean_policy_joint_correct"] for g in groups],
                                family, replicates)
        effects[key]["role"] = "primary" if control == "policy_compiler" else "remedy_gate"
        effects[key]["minimum_gain_for_remedy"] = .02
        for policy in pr.POLICIES:
            policy_effects[key + "/" + policy] = interval(
                [paired_rows[g, "neutral_compiler", policy]["joint_correct"] -
                 paired_rows[g, control, policy]["joint_correct"] for g in groups],
                replicates=replicates)
    cross = {}
    for method in ARMS:
        members = [r for r in per_case if r["method"] == method]
        grouped_rows = defaultdict(list)
        for row in rows:
            if row["method"] == method:
                grouped_rows[row["exact_text_group"]].append(row)
        crossing, pairs, attr_flip, correct_flip = 0, 0, 0, 0
        for group in groups:
            policies = grouped_rows[group]
            valid = [r for r in policies if r["attributes_valid"]]
            crossing += len(valid) == 3 and len({r["attributes_exact"] for r in valid}) > 1
            for a, b in combinations(valid, 2):
                pairs += 1
                attr_flip += a["predicted_attributes"] != b["predicted_attributes"]
                correct_flip += a["attributes_exact"] != b["attributes_exact"]
        cross[method] = {"groups": len(groups), "all_policy_joint_accuracy":
                         sum(r["all_policy_joint_correct"] for r in members) / len(groups),
                         "mean_policy_joint_accuracy": sum(r["mean_policy_joint_correct"] for r in members) / len(groups),
                         "all_policy_attribute_valid_groups": sum(r["all_policy_attributes_valid"] for r in members),
                         "all_valid_correctness_crossing_groups": crossing,
                         "valid_policy_pairs": pairs, "possible_policy_pairs": 3 * len(groups),
                         "attribute_disagreement_pairs": attr_flip, "exact_correctness_crossing_pairs": correct_flip}
    strata = {}
    for left, right in combinations(pr.POLICIES, 2):
        for flip in (False, True):
            subset = [r for r in rows if r["policy"] in (left, right) and
                      (pr.decision(r["reference_attributes"], left) !=
                       pr.decision(r["reference_attributes"], right)) == flip]
            strata[f"{left}__{right}/reference_flip_{flip}"] = {
                f"{m}/{p}": ev.metrics([r for r in subset if r["method"] == m and r["policy"] == p])
                for m in ARMS for p in (left, right)}
    return {"cells": cells, "cross_policy": cross, "paired_effects": effects,
            "per_policy_joint_effects_descriptive": policy_effects,
            "reference_rule_flip_strata": strata}, per_case


def evaluate(args):
    if args.output.exists():
        raise ValueError("Use a fresh evaluation directory")
    manifest, cases, bank, constant = load_prepared(args.prepared)
    first, config, manifests = checked_run(args.stage1_predictions, args.prepared / "stage1_requests.jsonl")
    repeat, config_r, manifests_r = checked_run(args.repeat_predictions, args.prepared / "repeat_requests.jsonl")
    if ev.fixed_config(config) != ev.fixed_config(config_r):
        raise ValueError("Repeat changed model or decoding")
    for key in repeat:
        if (first[key].get("cached") or repeat[key].get("cached")
                or first[key]["prompt_sha256"] != repeat[key]["prompt_sha256"]
                or first[key]["batch_size"] != repeat[key]["batch_size"]):
            raise ValueError("Repeat is cached, changes messages, or has unmatched effective batch size")
    for a, b in zip(sorted(manifests, key=lambda m: m["config"]["shard_index"]),
                    sorted(manifests_r, key=lambda m: m["config"]["shard_index"])):
        if a.get("process_id") == b.get("process_id") and a.get("start_utc") == b.get("start_utc"):
            raise ValueError("Repeat requires a separate fresh process")
    requests, states = second_artifacts(cases, bank, first)
    second_manifest = json.loads((args.stage2 / "preparation.json").read_text())
    if (second_manifest.get("version") != VERSION or second_manifest.get("stage") != 2
            or second_manifest.get("first_run_config") != config
            or second_manifest.get("first_run_signatures") != sorted(m["run_signature"] for m in manifests)
            or second_manifest.get("prepared_manifest_sha256") != ev.file_sha(args.prepared / "preparation.json")
            or second_manifest.get("requests") != len(requests)
            or second_manifest.get("invalid_first_stage") != len(states) - len(requests)
            or ev.read_jsonl(args.stage2 / "requests.jsonl") != requests
            or ev.read_jsonl(args.stage2 / "states.jsonl") != states):
        raise ValueError("Second stage does not reconstruct from frozen first-stage outputs")
    second, config2, _ = checked_run(args.stage2_predictions, args.stage2 / "requests.jsonl")
    if config2 is not None and ev.fixed_config(config) != ev.fixed_config(config2):
        raise ValueError("Second stage changed model or decoding")
    rows = ev.outcomes(cases, first, second, ev.indexed(states), methods=METHODS)
    if constant is not None:
        rows.extend(constant_rows(rows, constant["attributes"]))
    report, case_rows = summarize(rows)
    report["repeat_background"], repeat_rows = repeat_comparison(cases, first, repeat)
    report["cost"] = ev.costs(rows, first, second, methods=ARMS)
    report["repeat_cost"] = ev.costs([], repeat, {}, methods=())["physical_run"]
    if manifest["dataset"] == "hatecheck":
        from hatecheck_task import functional_summary
        report["functional_probe"] = functional_summary(cases, rows, case_rows, repeat_rows)
    prediction_index = {(stage, key): row for stage, source in ((1, first), (2, second)) for key, row in source.items()}
    for row in rows:
        row["logical_cost"] = ev.logical_cost(row["dependencies"], prediction_index)
    paths = args.stage1_predictions + args.repeat_predictions + args.stage2_predictions
    report.update(version=VERSION, dataset=manifest["dataset"], stage="heldout_formal_measurement", resampling_exact_text_groups=len(cases),
                  model_revision=config["model_revision"], primary_family="two models; two contrasts",
                  repeat_family="two models x three first-stage prompt types; six correctness contrasts",
                  multiplicity_note="Separate named families; all other intervals/readouts descriptive.",
                  preparation_sha256=ev.file_sha(args.prepared / "preparation.json"),
                  source_binding=manifest["source_binding"], training_constant=constant,
                  second_preparation_sha256=ev.file_sha(args.stage2 / "preparation.json"),
                  generation_manifests=[json.loads(p.with_suffix(p.suffix + ".manifest.json").read_text()) for p in paths],
                  prediction_files=[{"path": str(p), "sha256": ev.file_sha(p),
                                     "manifest_sha256": ev.file_sha(p.with_suffix(p.suffix + ".manifest.json"))} for p in paths],
                  new_human_annotations=0, independent_policy_gold=False,
                  interpretation=["Compiler joint accuracy equals exact attribute accuracy by construction.",
                                  "Neutral stability and compiler consistency are mechanical properties.",
                                  "Repeat intervals condition on six valid outputs; report panel coverage.",
                                  "Repeat raw pair rates use six across-rule and three same-prompt opportunities per group."])
    args.output.mkdir(parents=True)
    for name, records in (("outcomes.jsonl", rows), ("case_outcomes.jsonl", case_rows),
                          ("repeat_case_outcomes.jsonl", repeat_rows)):
        pr.write_rows(args.output / name, records)
        report[name + "_sha256"] = ev.file_sha(args.output / name)
    json_write(args.output / "summary.json", report)
    print(json.dumps({"groups": len(cases), "scored_rows": len(rows), "output": str(args.output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("prepare")
    one.add_argument("--source", required=True, type=Path)
    one.add_argument("--output", required=True, type=Path)
    one.add_argument("--dataset", choices=("sbic", "hatecheck"), default="sbic")
    two = commands.add_parser("stage2")
    two.add_argument("--prepared", required=True, type=Path)
    two.add_argument("--predictions", required=True, type=Path, nargs="+")
    two.add_argument("--output", required=True, type=Path)
    three = commands.add_parser("evaluate")
    three.add_argument("--prepared", required=True, type=Path)
    three.add_argument("--stage1-predictions", required=True, type=Path, nargs="+")
    three.add_argument("--repeat-predictions", required=True, type=Path, nargs="+")
    three.add_argument("--stage2", required=True, type=Path)
    three.add_argument("--stage2-predictions", type=Path, nargs="*", default=[])
    three.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    dataset = args.dataset if args.command == "prepare" else json.loads(
        (args.prepared / "preparation.json").read_text())["dataset"]
    prepare_function = prepare
    if dataset == "hatecheck":
        import hatecheck_task
        hatecheck_task.configure(sys.modules[__name__])
        prepare_function = hatecheck_task.prepare
    {"prepare": prepare_function, "stage2": stage2, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
