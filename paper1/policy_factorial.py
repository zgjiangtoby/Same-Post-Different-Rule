#!/usr/bin/env python3
"""Prospective policy-rendering follow-up; frozen original study stays unchanged."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations, product
import json
from pathlib import Path

import evaluate as ev
import prepare as pr
import hatecheck_task as hc
from generate import load_predictions, read_requests, load_local_tokenizer, sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VERSION = "policy_factorial_20261002"
RENDERERS = ("original", "predicate", "truth_table")
GROUPS = 512
SEED = 20261002
RUNNER_SHA = "f81bc57760070dbef65aaf60fa71ce3c70aa71f345224b7a4932e493f170ef6c"
REVISIONS = {"a09a35458c702b33eeacc393d103063234e8bc28", "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"}


def task(dataset):
    if dataset == "sbic":
        return pr.FIELDS, pr.POLICIES, pr.request, pr.decision, {
            "offense": "offensive=1", "sexual": "sexual=1", "intentional": "offensive=1 AND intent=1"}
    if dataset == "hatecheck":
        return hc.FIELDS, hc.POLICIES, hc.request, hc.decision, {
            "all_hate": "hateful=1", "directed_hate": "directed_hate=1", "general_hate": "hateful=1 AND directed_hate=0"}
    raise ValueError("Unknown native task")


def rendered_policy(dataset, policy, renderer):
    fields, policies, _, decision, predicates = task(dataset)
    if renderer == "original":
        return policies[policy]
    if renderer == "predicate":
        return "Flag a post exactly when this Boolean predicate is true: " + predicates[policy] + ". The content-attribute definitions stay fixed."
    if renderer == "truth_table":
        table = [{**dict(zip(fields, values)), "flag": decision(dict(zip(fields, values)), policy)}
                 for values in product((0, 1), repeat=len(fields))]
        return "Use this complete table to map fixed content attributes to flag (0=no, 1=yes): " + json.dumps(table, sort_keys=True, separators=(",", ":"))
    raise ValueError("Unknown renderer")


def request(case, bank, dataset, policy=None, renderer="neutral"):
    _, _, make, _, _ = task(dataset)
    row = make(case, bank, "full_record" if policy else "neutral_record", policy, demo_verdicts=False)
    if policy and renderer != "original":
        payload, instruction = row["messages"][1]["content"].rsplit("\n", 1)
        content = json.loads(payload)
        description = rendered_policy(dataset, policy, renderer)
        content["active_policy"] = description
        for example in content.get("examples", []):
            example["policy"] = description
            if "decision" in example:
                raise ValueError("Demonstration verdict entered extension")
        row["messages"][1]["content"] = json.dumps(content, ensure_ascii=False, sort_keys=True) + "\n" + instruction
    row["id"] = f"{case['id']}/factorial/{renderer}/{policy or 'neutral'}"
    row["metadata"] = {"case_id": case["id"], "exact_text_group": case["group"], "dataset": dataset,
                       "policy": policy, "renderer": renderer, "partition": "test"}
    return row


def construct(cases, bank, dataset):
    output = []
    for case in cases:
        output.extend(request(case, bank, dataset, policy, renderer)
                      for renderer in RENDERERS for policy in task(dataset)[1])
        output.append(request(case, bank, dataset))
    return output


def prepare(args):
    if args.output.exists():
        raise ValueError("Fresh extension directory required")
    all_cases, requests, bases, banks = [], [], {}, {}
    for dataset, base in (("sbic", args.sbic_prepared), ("hatecheck", args.hatecheck_prepared)):
        manifest = json.loads((base / "preparation.json").read_text())
        if manifest.get("dataset") != dataset or manifest.get("demo_verdicts") is not False:
            raise ValueError("Wrong original native preparation")
        for name in ("references.jsonl", "demonstrations.jsonl"):
            if ev.file_sha(base / name) != manifest["files"][name]:
                raise ValueError("Original references/demonstrations changed")
        source_cases = ev.read_jsonl(base / "references.jsonl")
        if len(source_cases) != {"sbic": 4336, "hatecheck": 3728}[dataset]:
            raise ValueError("Original eligible population changed")
        bank = ev.read_jsonl(base / "demonstrations.jsonl")
        cases = sorted(source_cases, key=lambda c: pr.digest("paper1-factorial-20261002:" + c["group"]))[:GROUPS]
        fields = task(dataset)[0]
        if len(cases) != GROUPS or len({c["group"] for c in cases}) != GROUPS:
            raise ValueError("Insufficient/duplicate exact original text groups")
        if ({c["group"] for c in cases} & {c["group"] for c in bank}
                or any(c["partition"] != "train" for c in bank)
                or len(bank) != (4 if dataset == "sbic" else 0)):
            raise ValueError("Original demonstration boundary changed")
        for c in cases:
            if (c["group"] != pr.digest(c["text"]) or c["partition"] != "test"
                    or set(c["attributes"]) != set(fields)
                    or any(type(v) is not int or v not in (0, 1) for v in c["attributes"].values())):
                raise ValueError("Invalid fixed reference")
            c["dataset"] = dataset
        requests.extend(construct(cases, bank, dataset))
        all_cases.extend(cases)
        banks[dataset] = bank
        bases[dataset] = {"preparation_sha256": ev.file_sha(base / "preparation.json"),
                          "reference_sha256": manifest["files"]["references.jsonl"],
                          "demonstration_sha256": manifest["files"]["demonstrations.jsonl"],
                          "eligible_groups": len(source_cases), "selected_groups": len(cases)}
    args.output.mkdir(parents=True)
    pr.write_rows(args.output / "requests.jsonl", requests)
    pr.write_rows(args.output / "references.jsonl", all_cases)
    write_json(args.output / "demonstrations.json", banks)
    write_json(args.output / "preparation.json", {"version": VERSION, "bases": bases, "groups_per_task": GROUPS,
               "requests": len(requests), "source_binding": source_binding(), "runner_sha256": RUNNER_SHA,
               "files": {n: ev.file_sha(args.output / n) for n in ("requests.jsonl", "references.jsonl", "demonstrations.json")},
               "selection": "label-free exact-text hash order after old fixed eligibility", "new_human_annotations": 0})
    print(json.dumps({"groups": len(all_cases), "requests": len(requests)}))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.write("\n")


def source_binding():
    return {n: ev.file_sha(HERE / n) for n in ("policy_factorial.py", "test_policy_factorial.py", "POLICY_FACTORIAL_PLAN_20261002.md", "prepare.py", "hatecheck_task.py")}


def load_prepared(path):
    m = json.loads((path / "preparation.json").read_text())
    if m["version"] != VERSION or m["source_binding"] != source_binding():
        raise ValueError("Extension source/plan changed")
    for n, expected in m["files"].items():
        if ev.file_sha(path / n) != expected:
            raise ValueError("Extension input changed: " + n)
    cases = ev.read_jsonl(path / "references.jsonl")
    banks = json.loads((path / "demonstrations.json").read_text())
    rows = read_requests(path / "requests.jsonl")
    if rows != [r for d in ("sbic", "hatecheck") for r in construct([c for c in cases if c["dataset"] == d], banks[d], d)]:
        raise ValueError("Extension requests do not reconstruct")
    return m, cases, rows


def token_check(args):
    _, cases, rows = load_prepared(args.prepared)
    tokenizer = load_local_tokenizer(args.model)
    counts = {}
    for row in rows:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, thinking=False, enable_thinking=False)
        counts[row["id"]] = len(tokenizer.encode(text, add_special_tokens=False))
    truth = defaultdict(list)
    for row in rows:
        if row["metadata"]["renderer"] == "truth_table":
            truth[row["metadata"]["case_id"]].append(counts[row["id"]])
    mismatch = {k: v for k, v in truth.items() if len(set(v)) != 1}
    report = {"requests": len(rows), "groups": len(cases), "max_input_tokens": max(counts.values()),
              "overlength": sum(v > 8192 for v in counts.values()), "truth_table_length_mismatches": mismatch,
              "tokenizer_class": type(tokenizer).__name__, "model": str(args.model), "input_counts": counts}
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "input_counts"}))
    if mismatch:
        raise ValueError("Truth-table policies are not token-length matched")


def parse_attributes(response, fields):
    obj = pr.parse_json(response)
    a = obj.get("attributes") if obj else None
    if (not isinstance(a, dict) or set(a) != set(fields)
            or any(type(v) not in (int, bool) or v not in (0, 1) for v in a.values())):
        return None
    return {k: int(v) for k, v in a.items()}


def interval(values, clusters=None, family=4):
    import numpy as np
    if not values:
        return {"groups": 0, "mean": None, "ci": None}
    groups = defaultdict(list)
    for i, v in enumerate(values):
        groups[clusters[i] if clusters is not None else i].append(v)
    blocks = list(groups.values())
    totals, sizes = np.array([sum(b) for b in blocks]), np.array([len(b) for b in blocks])
    rng, samples = np.random.default_rng(SEED), []
    for start in range(0, 10000, 128):
        idx = rng.integers(len(blocks), size=(min(128, 10000 - start), len(blocks)))
        samples.extend(totals[idx].sum(axis=1) / sizes[idx].sum(axis=1))
    return {"groups": len(values), "blocks": len(blocks), "mean": float(sum(values) / len(values)),
            "ci": np.quantile(samples, [.025 / family, 1 - .025 / family]).tolist(), "family_size": family,
            "replicates": 10000, "seed": SEED, "scope": "conditional paired percentile case/block resampling; nominal Bonferroni tails"}


def rate(pairs, gold):
    valid = [(a, b) for a, b in pairs if a is not None and b is not None]
    return {"possible": len(pairs), "valid": len(valid), "crossings": sum((a == gold) != (b == gold) for a, b in valid),
            "attribute_changes": sum(a != b for a, b in valid)}


def summarize(cases, predictions):
    reports, outcomes = {}, []
    for dataset in ("sbic", "hatecheck"):
        members = [c for c in cases if c["dataset"] == dataset]
        fields, policies, _, _, _ = task(dataset)
        cells, per_case = defaultdict(list), []
        for case in members:
            matrices = []
            for source in predictions:
                matrix = {}
                for renderer in (*RENDERERS, "neutral"):
                    for policy in policies if renderer != "neutral" else (None,):
                        key = f"{case['id']}/factorial/{renderer}/{policy or 'neutral'}"
                        pred = source[key]
                        attr = parse_attributes(pred.get("response"), fields) if pred.get("status") == "ok" else None
                        matrix[renderer, policy] = attr
                        row = {"id": key, "dataset": dataset, "exact_text_group": case["group"], "renderer": renderer,
                               "policy": policy, "realization": len(matrices), "valid": attr is not None,
                               "correct": int(attr == case["attributes"]), "predicted_attributes": attr,
                               "field_correct": {f: int(attr is not None and attr[f] == case["attributes"][f]) for f in fields}}
                        cells[renderer, policy].append(row)
                        outcomes.append(row)
                matrices.append(matrix)
            row = {"group": case["group"], "template": case.get("native_template_id"), "by_renderer": {},
                   "accuracy": {r: sum(m[r, p] == case["attributes"] for m in matrices for p in policies) / 6 for r in RENDERERS},
                   "neutral_accuracy": sum(m["neutral", None] == case["attributes"] for m in matrices) / 2}
            all_across, all_words = [], []
            for renderer in RENDERERS:
                crossed = [p for matrix in matrices for p in combinations([matrix[renderer, policy] for policy in policies], 2)]
                repeated = [(matrices[0][renderer, p], matrices[1][renderer, p]) for p in policies]
                vals = [matrix[renderer, p] for matrix in matrices for p in policies]
                row["by_renderer"][renderer] = {"complete": all(a is not None for a in vals),
                    "cross": rate(crossed, case["attributes"]), "fresh": rate(repeated, case["attributes"])}
                all_across.extend(crossed)
            for policy in policies:
                all_words.extend(p for matrix in matrices for p in combinations([matrix[r, policy] for r in RENDERERS], 2))
            row.update(all18valid=all(a is not None for m in matrices for a in [m[r, p] for r in RENDERERS for p in policies]),
                       across=rate(all_across, case["attributes"]), wording=rate(all_words, case["attributes"]))
            per_case.append(row)
        renderer_stats = {}
        for renderer in RENDERERS:
            panel = [r for r in per_case if r["by_renderer"][renderer]["complete"]]
            values = [r["by_renderer"][renderer]["cross"]["crossings"] / 6 - r["by_renderer"][renderer]["fresh"]["crossings"] / 3 for r in panel]
            totals = {axis: {k: sum(r["by_renderer"][renderer][axis][k] for r in per_case) for k in ("possible", "valid", "crossings", "attribute_changes")} for axis in ("cross", "fresh")}
            renderer_stats[renderer] = {"selected": len(members), "complete": len(panel), "coverage": len(panel) / len(members),
                "population_coverage_gate": len(panel) / len(members) >= .95, "events": totals,
                "correctness_cross_minus_fresh": interval(values, family=4 if renderer == "truth_table" else 1)}
            if dataset == "hatecheck":
                renderer_stats[renderer]["native_template_sensitivity"] = interval(values, [r["template"] for r in panel], family=1)
        full = [r for r in per_case if r["all18valid"]]
        diff = [r["across"]["crossings"] / 18 - r["wording"]["crossings"] / 18 for r in full]
        metrics = {f"{r}/{p or 'neutral'}": {"opportunities": len(v), "correct": sum(x["correct"] for x in v),
                   "valid": sum(x["valid"] for x in v), "accuracy": sum(x["correct"] for x in v) / len(v),
                   "field_accuracy": {f: sum(x["field_correct"][f] for x in v) / len(v) for f in fields}}
                   for (r, p), v in cells.items()}
        accuracy = {r: {"mean": sum(c["accuracy"][r] for c in per_case) / len(per_case),
                       "minus_original": interval([c["accuracy"][r] - c["accuracy"]["original"] for c in per_case], family=1),
                       "neutral_minus_renderer": interval([c["neutral_accuracy"] - c["accuracy"][r] for c in per_case], family=1)} for r in RENDERERS}
        reports[dataset] = {"groups": len(members), "cells": metrics, "renderers": renderer_stats, "accuracy": accuracy,
                           "balanced_cross_minus_wording": {"complete": len(full), "coverage": len(full) / len(members), "effect": interval(diff),
                               "events": {a: {k: sum(r[a][k] for r in per_case) for k in ("possible", "valid", "crossings", "attribute_changes")} for a in ("across", "wording")}}}
        if dataset == "hatecheck":
            reports[dataset]["balanced_cross_minus_wording"]["native_template_sensitivity"] = interval(diff, [r["template"] for r in full], family=1)
    return reports, outcomes


def evaluate(args):
    if args.output.exists():
        raise ValueError("Fresh evaluation directory required")
    manifest, cases, rows = load_prepared(args.prepared)
    predictions, configs, receipts = [], [], []
    for paths in (args.realization_a, args.realization_b):
        pred, config, metas = load_predictions(paths, args.prepared / "requests.jsonl")
        fixed = {"code_sha256": RUNNER_SHA, "num_shards": 2, "batch_size": 8, "seed": 17,
                 "max_input_tokens": 8192, "max_new_tokens": 128, "do_sample": False,
                 "dtype": "bfloat16", "attention": "sdpa", "thinking": False, "tokenizer_use_fast": False}
        if (set(pred) != {r["id"] for r in rows} or config.get("model_revision") not in REVISIONS
                or any(config.get(k) != v for k, v in fixed.items()) or len(metas) != 2
                or {m["config"]["shard_index"] for m in metas} != {0, 1}):
            raise ValueError("Incomplete or changed extension run")
        for i, row in enumerate(rows):
            shard = next(m for m in metas if m["run_signature"] == pred[row["id"]]["run_signature"])
            if shard["config"]["shard_index"] != i % 2:
                raise ValueError("Wrong logical shard")
        predictions.append(pred)
        configs.append(config)
        receipts.append(metas)
    if configs[0] != configs[1]:
        raise ValueError("Fresh repeat changed decoding/model")
    for row in rows:
        a, b = [p[row["id"]] for p in predictions]
        if a.get("cached") or b.get("cached") or a["batch_size"] != b["batch_size"]:
            raise ValueError("Repeat not uncached/effective-batch matched")
    for a, b in zip(sorted(receipts[0], key=lambda m: m["config"]["shard_index"]), sorted(receipts[1], key=lambda m: m["config"]["shard_index"])):
        if (a["process_id"], a["start_utc"]) == (b["process_id"], b["start_utc"]):
            raise ValueError("Repeat was not a fresh process")
    reports, outcomes = summarize(cases, predictions)
    physical = [p for source in predictions for p in source.values() if not p.get("cached")]
    report = {"version": VERSION, "model_revision": configs[0]["model_revision"], "datasets": reports,
              "preparation_sha256": ev.file_sha(args.prepared / "preparation.json"), "source_binding": manifest["source_binding"],
              "generation_manifests": receipts, "logical_requests": sum(len(p) for p in predictions), "physical_requests": len(physical),
              "charged_input_tokens": sum(p["charged_input_tokens"] for p in physical), "charged_output_tokens": sum(p["charged_output_tokens"] for p in physical),
              "elapsed_batch_seconds": sum(p["elapsed_batch_seconds"] / max(1, p["batch_size"]) for p in physical),
              "new_human_annotations": 0, "scope": "post-original-results prospective follow-up; fixed native labels; no internal causal mechanism identification"}
    args.output.mkdir(parents=True)
    pr.write_rows(args.output / "outcomes.jsonl", outcomes)
    write_json(args.output / "summary.json", report)
    print(json.dumps({"output": str(args.output), "revision": report["model_revision"], "logical_requests": report["logical_requests"]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--sbic-prepared", required=True, type=Path)
    prep.add_argument("--hatecheck-prepared", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    check = sub.add_parser("token-check")
    check.add_argument("--prepared", required=True, type=Path)
    check.add_argument("--model", required=True, type=Path)
    check.add_argument("--output", required=True, type=Path)
    score = sub.add_parser("evaluate")
    score.add_argument("--prepared", required=True, type=Path)
    score.add_argument("--realization-a", required=True, nargs="+", type=Path)
    score.add_argument("--realization-b", required=True, nargs="+", type=Path)
    score.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    {"prepare": prepare, "token-check": token_check, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
