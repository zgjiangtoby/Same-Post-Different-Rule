#!/usr/bin/env python3
"""Constructed CPU tests only: no corpus/model run and no empirical conclusions."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import evaluate as ev
from prepare import FIELDS, POLICIES, decision, digest, request, stage2
from generate import read_requests, sha


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def example(text, pattern, partition="dev"):
    group = digest(text)
    return {"id": f"sbic:{group}", "group": group, "text": text, "partition": partition,
            "attributes": dict(zip(FIELDS, pattern))}


def write_run(path, request_path, responses, *, seed=17, num_shards=1):
    config = {"input_sha256": ev.file_sha(request_path), "code_sha256": "constructed-runner",
              "model_path": "constructed-not-a-model", "model_revision": "constructed",
              "model_config_sha256": "constructed", "versions": {}, "dtype": "bfloat16",
              "attention": "sdpa", "do_sample": False, "thinking": False, "batch_size": 8,
              "max_new_tokens": 128, "max_input_tokens": 8192, "seed": seed,
              "shard_index": 0, "num_shards": num_shards}
    rows = []
    for req in read_requests(request_path):
        status, response = responses[req["id"]]
        rows.append({"id": req["id"], "request_sha256": sha(req), "prompt_sha256": sha(req["messages"]),
                     "run_signature": sha(config), "metadata": req["metadata"], "status": status,
                     "response": response, "input_tokens": 100 if status == "ok" else 9000,
                     "output_tokens": 20 if status == "ok" else 0,
                     "charged_input_tokens": 100 if status == "ok" else 0,
                     "charged_output_tokens": 20 if status == "ok" else 0,
                     "cached": False, "elapsed_batch_seconds": .25 if status == "ok" else 0,
                     "batch_size": 1 if status == "ok" else 0})
    write_jsonl(path, rows)
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps(
        {"status": "COMPLETED", "config": config, "run_signature": sha(config)}))
    return rows


def fixture(root, all_invalid=False, demo_verdicts=True):
    prepared, second = root / "prepared", root / "second"
    prepared.mkdir()
    cases = [example("Constructed case A.", (0, 0, 0)), example("Constructed case B.", (1, 0, 0)),
             example("Constructed case C.", (0, 0, 1))]
    bank = [example(f"Constructed training {index}.", pattern, "train") for index, pattern in enumerate(
        ((0, 0, 0), (1, 1, 0), (1, 0, 0), (0, 0, 1)))]
    requests, responses = [], {}
    for index, case in enumerate(cases):
        pending = [request(case, bank, "neutral_record", demo_verdicts=demo_verdicts)]
        pending.extend(request(case, bank, method, policy, demo_verdicts=demo_verdicts) for policy in POLICIES
                       for method in ("direct", "guarded_direct", "full_record"))
        for req in pending:
            policy, method = req["metadata"]["policy"], req["metadata"]["method"]
            attributes = case["attributes"]
            if method == "neutral_record" and index == 0:
                attributes = dict.fromkeys(FIELDS, 1)  # Stable, intentionally wrong record.
            value = {"attributes": attributes}
            if policy:
                value["decision"] = decision(attributes, policy)
            if index == 0 and method == "direct" and policy == "offense":
                value["attributes"] = {**attributes, "offensive": 2}
            if index == 1 and method == "direct" and policy == "sexual":
                value["decision"] = 9
            invalid = (method == "neutral_record" and index == 2) or (
                all_invalid and method in ("neutral_record", "full_record"))
            responses[req["id"]] = ("input_too_long", None) if invalid else ("ok", json.dumps(value))
        requests.extend(pending)
    write_jsonl(prepared / "stage1_requests.jsonl", requests)
    write_jsonl(prepared / "references.jsonl", cases)
    write_jsonl(prepared / "demonstrations.jsonl", bank)
    manifest = {"version": "sbic_policy_v2", "policies": POLICIES, "partition": "dev", "selected_cases": len(cases),
                "demonstration_ids": [r["id"] for r in bank], "requests": len(requests)}
    if not demo_verdicts:
        manifest.update(version="sbic_policy_v3", demo_verdicts=False)
    (prepared / "preparation.json").write_text(json.dumps(manifest))
    one, two = root / "first.jsonl", root / "last.jsonl"
    write_run(one, prepared / "stage1_requests.jsonl", responses)
    with contextlib.redirect_stdout(io.StringIO()):
        stage2(SimpleNamespace(output=second, predictions=[one], prepared=prepared))
    second_requests = ev.read_jsonl(second / "requests.jsonl")
    if second_requests:
        responses = {}
        by_id = {case["id"]: case for case in cases}
        for req in second_requests:
            case = by_id[req["metadata"]["case_id"]]
            policy, method = req["metadata"]["policy"], req["metadata"]["method"]
            state = next(r for r in ev.read_jsonl(second / "states.jsonl") if r["id"] == req["id"])
            attributes = state["attributes"] if method in ("ap_commit", "freeze_only") else case["attributes"]
            value = {"decision": decision(attributes, policy)}
            if method not in ("ap_commit", "freeze_only"):
                value["attributes"] = attributes
            responses[req["id"]] = "ok", json.dumps(value)
        # Deliberately different shard layout, unchanged model/decoding.
        rows = write_run(two, second / "requests.jsonl", responses, num_shards=2)
        cached = next(r for r in rows if r["id"] == f"{cases[1]['id']}/offense/ap_commit")
        source = next(r for r in rows if r["id"] == f"{cases[1]['id']}/offense/freeze_only")
        assert cached["prompt_sha256"] == source["prompt_sha256"]
        cached.update(cached=True, cache_of=source["id"], charged_input_tokens=0, charged_output_tokens=0,
                      elapsed_batch_seconds=0, batch_size=0)
        write_jsonl(two, rows)
    return prepared, one, second, [two] if second_requests else []


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="paper1_evaluator_constructed_")
        self.root = Path(self.temp.name)
        self.paths = fixture(self.root)
        self.bundle = ev.load_bundle(self.paths[0], [self.paths[1]], self.paths[2], self.paths[3])
        self.cases, self.pred1, self.pred2, self.states, _ = self.bundle
        self.rows = ev.outcomes(self.cases, self.pred1, self.pred2, self.states)

    def tearDown(self):
        self.temp.cleanup()

    def row(self, index, policy, method):
        return next(r for r in self.rows if r["case_id"] == self.cases[index]["id"]
                    and r["policy"] == policy and r["method"] == method)

    def test_independent_marginals_and_missing_denominator(self):
        bad_attrs = self.row(0, "offense", "direct")
        self.assertEqual((bad_attrs["attributes_exact"], bad_attrs["verdict_correct"], bad_attrs["joint_correct"]), (0, 1, 0))
        self.assertEqual(bad_attrs["field_correct"]["sexual"], 1)
        bad_decision = self.row(1, "sexual", "direct")
        self.assertEqual((bad_decision["attributes_exact"], bad_decision["verdict_correct"]), (1, 0))
        key = f"{self.cases[0]['id']}/sexual/direct"
        del self.pred1[key]
        rows = ev.outcomes(self.cases, self.pred1, self.pred2, self.states)
        cell = ev.metrics([r for r in rows if r["method"] == "direct" and r["policy"] == "sexual"])
        self.assertEqual(cell["n"], 3)
        self.assertEqual(cell["output_status_counts"]["missing"], 1)

    def test_first_failure_and_committed_attribute_survives_missing_decision(self):
        failed = self.row(2, "offense", "ap_commit")
        self.assertEqual((failed["joint_correct"], failed["output_status"]), (0, "first_stage_failure"))
        key = f"{self.cases[1]['id']}/offense/ap_commit"
        del self.pred2[key]
        row = next(r for r in ev.outcomes(self.cases, self.pred1, self.pred2, self.states) if r["id"] == key)
        self.assertTrue(row["attributes_valid"])
        self.assertEqual((row["attributes_exact"], row["verdict_correct"], row["joint_correct"]), (1, 0, 0))
        self.assertEqual(len(self.rows), 3 * 3 * len(ev.METHODS))

    def test_stability_and_compiler_consistency_are_not_accuracy(self):
        case = next(r for r in ev.case_metrics(self.rows) if r["case_id"] == self.cases[0]["id"] and r["method"] == "ap_commit")
        self.assertEqual((case["attribute_pair_count"], case["attribute_pair_flips"], case["all_policy_joint_correct"]), (3, 0, 0))
        compiler = self.row(0, "offense", "neutral_compiler")
        self.assertTrue(compiler["rule_consistent"])
        self.assertEqual(compiler["joint_correct"], 0)
        for row in self.rows:
            if row["method"].endswith("_compiler"):
                self.assertEqual(row["joint_correct"], row["attributes_exact"])
        for case in self.cases:
            for policy in POLICIES:
                self.assertEqual(self.states[f"{case['id']}/{policy}/freeze_only"]["first_id"],
                                 f"{case['id']}/{policy}/full_record")

    def test_grouped_bootstrap_keeps_policy_triplets(self):
        rows = []
        for group, sign in (("one", 1), ("two", -1)):
            for policy in POLICIES:
                rows.extend({"exact_text_group": group, "policy": policy, "method": method,
                             "joint_correct": int((sign == 1) == (method == "a"))} for method in ("a", "b"))
        result = ev.paired_effect(rows, "a", "b")
        self.assertEqual((result["resampling_groups"], result["paired_rows"], result["difference"]), (2, 6, 0))
        self.assertEqual(result["ci95"], [-1, 1])
        self.assertEqual(result, ev.paired_effect(rows, "a", "b"))

    def test_cost_reuse_cache_and_no_raw_text(self):
        report, cases = ev.summarize(self.rows, self.pred1, self.pred2, replicates=20)
        neutral = report["cost"]["arms"]["neutral_compiler"]
        self.assertEqual(neutral["standalone_per_policy_sum"]["logical_calls"], 9)
        self.assertEqual(neutral["shared_across_three_policies"]["logical_calls"], 3)
        ap = report["cost"]["arms"]["ap_commit"]
        self.assertEqual(ap["standalone_per_policy_sum"]["logical_calls"], 15)
        self.assertEqual(ap["shared_across_three_policies"]["logical_calls"], 9)
        physical = report["cost"]["physical_run"]
        self.assertEqual(physical["cached_requests"], 1)
        self.assertEqual(physical["charged_input_tokens"], 100 * physical["physical_generated_requests"])
        self.assertEqual(physical["generation_batch_seconds"], .25 * physical["physical_generated_requests"])
        serialized = json.dumps([self.rows, report, cases])
        self.assertNotIn("Constructed case", serialized)
        self.assertEqual(report["cross_policy"]["ap_commit"]["pair_coverage"], 2 / 3)

    def test_provenance_rejects_changed_state_seed_and_missing_required_first(self):
        second = self.paths[2]
        state_path = second / "states.jsonl"
        states = ev.read_jsonl(state_path)
        original = state_path.read_text()
        states[0]["attributes"]["offensive"] ^= 1
        write_jsonl(state_path, states)
        with self.assertRaisesRegex(ValueError, "states/requests"):
            ev.load_bundle(self.paths[0], [self.paths[1]], second, self.paths[3])
        state_path.write_text(original)
        output = self.paths[3][0]
        rows = ev.read_jsonl(output)
        original_rows = output.read_text()
        manifest_path = output.with_suffix(output.suffix + ".manifest.json")
        original_manifest = manifest_path.read_text()
        manifest = json.loads(original_manifest)
        manifest["config"]["seed"] += 1
        manifest["run_signature"] = sha(manifest["config"])
        for row in rows:
            row["run_signature"] = manifest["run_signature"]
        write_jsonl(output, rows)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "between stages"):
            ev.load_bundle(self.paths[0], [self.paths[1]], second, self.paths[3])
        output.write_text(original_rows)
        manifest_path.write_text(original_manifest)
        rows = [r for r in ev.read_jsonl(self.paths[1]) if r["id"] != f"{self.cases[0]['id']}/neutral/neutral_record"]
        write_jsonl(self.paths[1], rows)
        with self.assertRaisesRegex(ValueError, "without its first-stage"):
            ev.load_bundle(self.paths[0], [self.paths[1]], second, self.paths[3])

    def test_all_first_stages_invalid_need_no_second_run(self):
        other = self.root / "all_failed"
        other.mkdir()
        prepared, one, second, paths = fixture(other, all_invalid=True)
        cases, pred1, pred2, states, provenance = ev.load_bundle(prepared, [one], second, paths)
        self.assertEqual(paths, [])
        self.assertTrue(provenance["all_second_stage_requests_suppressed"])
        rows = ev.outcomes(cases, pred1, pred2, states)
        failed = [r for r in rows if r["method"] in ev.SECOND_STAGE]
        self.assertEqual(len(failed), len(cases) * 3 * 4)
        self.assertTrue(all(r["output_status"] == "first_stage_failure" for r in failed))
        self.assertEqual(ev.metrics(failed)["joint_accuracy"], 0)

    def test_verdict_free_control_deletes_only_demonstration_decisions(self):
        other = self.root / "verdict_free"
        other.mkdir()
        prepared, one, second, paths = fixture(other, demo_verdicts=False)
        bundle = ev.load_bundle(prepared, [one], second, paths)
        self.assertFalse(bundle[-1]["demo_verdicts"])
        self.assertEqual(ev.outcomes(*bundle[:4]), self.rows)
        for original, revised in ((self.paths[0] / "stage1_requests.jsonl", prepared / "stage1_requests.jsonl"),
                                  (self.paths[2] / "requests.jsonl", second / "requests.jsonl")):
            old_rows, new_rows = read_requests(original), read_requests(revised)
            self.assertEqual(len(old_rows), len(new_rows))
            for expected, actual in zip(old_rows, new_rows):
                body, instruction = expected["messages"][1]["content"].split("\n", 1)
                body = json.loads(body)
                for demonstration in body["examples"]:
                    demonstration.pop("decision", None)
                expected["messages"][1]["content"] = json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n" + instruction
                self.assertEqual(expected, actual)
        path = second / "preparation.json"
        manifest = json.loads(path.read_text())
        manifest["demo_verdicts"] = True
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "demonstration verdict condition"):
            ev.load_bundle(prepared, [one], second, paths)

    def test_cli_saves_recomputable_text_free_artifacts(self):
        destination = self.root / "evaluated"
        argv = ["evaluate.py", "--prepared", str(self.paths[0]), "--stage1-predictions", str(self.paths[1]),
                "--stage2", str(self.paths[2]), "--stage2-predictions", str(self.paths[3][0]),
                "--output", str(destination), "--stage", "pilot", "--bootstrap", "20"]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            ev.main()
        report = json.loads((destination / "summary.json").read_text())
        rows = ev.read_jsonl(destination / "outcomes.jsonl")
        self.assertEqual(report["resampling_exact_text_groups"], 3)
        self.assertEqual(len(rows), 3 * 3 * len(ev.METHODS))
        self.assertEqual(report["outcomes_sha256"], ev.file_sha(destination / "outcomes.jsonl"))
        self.assertTrue(all("logical_cost" in row for row in rows))
        for path in destination.iterdir():
            self.assertNotIn("Constructed case", path.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
