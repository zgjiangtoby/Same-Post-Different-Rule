"""Constructed checks for the frozen formal estimands and fresh-repeat binding."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import formal as fm
import evaluate as ev
import hatecheck_task as hc
import prepare as pr
from generate import sha


def write_runs(directory, request_file, cases, realization=0, corrupt_repeat=False):
    by_id = {r["id"]: r for r in cases}
    requests = ev.read_jsonl(request_file)
    paths = []
    for shard in range(4):
        config = {"input_sha256": ev.file_sha(request_file), "code_sha256": fm.RUNNER_SHA,
                  "model_path": "constructed", "model_revision": sorted(fm.MODEL_REVISIONS)[0],
                  "model_config_sha256": "constructed", "versions": {}, "dtype": "bfloat16",
                  "attention": "sdpa", "thinking": False, "do_sample": False,
                  "tokenizer_use_fast": False, "batch_size": 8, "max_new_tokens": 128,
                  "max_input_tokens": 8192, "seed": 17, "num_shards": 4, "shard_index": shard}
        rows = []
        for req in requests[shard::4]:
            attrs = by_id[req["metadata"]["case_id"]]["attributes"]
            value = {"attributes": attrs}
            if req["metadata"]["policy"]:
                value["decision"] = pr.decision(attrs, req["metadata"]["policy"])
            rows.append({"id": req["id"], "metadata": req["metadata"], "request_sha256": sha(req),
                         "prompt_sha256": sha(req["messages"]), "run_signature": sha(config),
                         "response": json.dumps(value), "status": "ok", "cached": corrupt_repeat,
                         "input_tokens": 100, "output_tokens": 20, "charged_input_tokens": 100,
                         "charged_output_tokens": 20, "batch_size": 8, "elapsed_batch_seconds": .1})
        path = directory / f"part{realization}_{shard}.jsonl"
        pr.write_rows(path, rows)
        fm.json_write(path.with_suffix(".jsonl.manifest.json"), {
            "status": "COMPLETED", "config": config, "run_signature": sha(config),
            "total_requests": len(rows), "completed": len(rows), "process_id": 100 + realization * 4 + shard,
            "start_utc": f"constructed-{realization}"})
        paths.append(path)
    return paths


class FormalTests(unittest.TestCase):
    def test_hatecheck_native_schema_and_full_path(self):
        saved = [(module, name, getattr(module, name)) for module, names in
                 ((pr, ("FIELDS", "POLICIES", "decision", "request")),
                  (ev, ("FIELDS", "POLICIES", "decision")), (fm, ("VERSION", "ARMS"))) for name in names]
        try:
            hc.configure(fm)
            cases = []
            for i in range(64):
                text = f"Constructed functional case {i}."
                group = pr.digest(text)
                cases.append({"id": "hatecheck:" + group, "group": group, "text": text, "partition": "test",
                              "attributes": {"hateful": int(i % 3 != 0), "directed_hate": int(i % 3 == 1)},
                              "native_case_id": str(i), "native_template_id": str(i // 3),
                              "functionality": "constructed", "source_record": {"data_row_1based": i + 1}})
            attrs = {"hateful": 0, "directed_hate": 0}
            self.assertEqual([hc.decision(attrs, p) for p in hc.POLICIES], [0, 0, 0])
            self.assertIsNotNone(pr.parse_output(json.dumps({"attributes": attrs})))
            self.assertIsNone(pr.parse_output('{"attributes":{"offensive":0,"intent":0,"sexual":0}}'))
            req = hc.request(cases[0], [], "direct", "all_hate")
            self.assertNotIn("functionality", req["messages"][1]["content"])
            self.assertEqual(req, hc.request({**cases[0], "attributes": {"hateful": 1, "directed_hate": 1}},
                                            [], "direct", "all_hate"))
            with tempfile.TemporaryDirectory(prefix="hatecheck_formal_constructed_") as directory:
                root = Path(directory)
                prepared, second = root / "prepared", root / "second"
                with contextlib.redirect_stdout(io.StringIO()):
                    fm.save_prepared(prepared, cases, [], [], None, {}, "hatecheck")
                    self.assertIsNone(fm.load_prepared(prepared)[-1])
                    first = write_runs(root, prepared / "stage1_requests.jsonl", cases)
                    repeat = write_runs(root, prepared / "repeat_requests.jsonl", cases, realization=1)
                    fm.stage2(SimpleNamespace(prepared=prepared, predictions=first, output=second))
                    two = write_runs(root, second / "requests.jsonl", cases, realization=2)
                    fm.evaluate(SimpleNamespace(prepared=prepared, stage1_predictions=first,
                                                repeat_predictions=repeat, stage2=second,
                                                stage2_predictions=two, output=root / "evaluation"))
                report = json.loads((root / "evaluation" / "summary.json").read_text())
                self.assertIsNone(report["training_constant"])
                self.assertNotIn("training_constant", report["cost"]["arms"])
                self.assertEqual(report["functional_probe"]["native_template_sensitivity"]
                                 ["neutral_minus_policy_compiler"]["difference"], 0)
                self.assertEqual(report["functional_probe"]["native_template_sensitivity"]
                                 ["neutral_minus_policy_compiler"]["native_templates"], 22)
        finally:
            for module, name, value in saved:
                setattr(module, name, value)

    def test_constructed_end_to_end_and_provenance(self):
        with tempfile.TemporaryDirectory(prefix="paper1_formal_constructed_") as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            patterns = [(0, 0, 0), (1, 1, 0), (1, 0, 0), (0, 0, 1)]
            for suffix, texts in (("trn", [(f"training-{i}", a) for i, a in enumerate(patterns)]),
                                  ("dev", [("development-only", (0, 0, 0))]),
                                  ("tst", [(f"heldout-{i}", patterns[i % 4]) for i in range(64)] +
                                   [("development-only", (0, 0, 0)), ("unresolved", (1, .5, ""))])):
                with (source / f"SBIC.v2.agg.{suffix}.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["post", *pr.FIELDS.values()])
                    writer.writeheader()
                    for text, attrs in texts:
                        writer.writerow({"post": text, **dict(zip(pr.FIELDS.values(), attrs))})
            prepared, second, output = root / "prepared", root / "second", root / "output"
            with contextlib.redirect_stdout(io.StringIO()):
                fm.prepare(SimpleNamespace(source=source, output=prepared))
            manifest, cases, bank, constant = fm.load_prepared(prepared)
            self.assertEqual((manifest["selected_cases"], manifest["requests"], manifest["repeat_requests"]),
                             (64, 640, 576))
            self.assertEqual(constant["attributes"], dict.fromkeys(pr.FIELDS, 0))
            self.assertEqual(len(ev.read_jsonl(prepared / "resolution_log.jsonl")), 66)
            main_requests = ev.read_jsonl(prepared / "stage1_requests.jsonl")
            repeat_requests = ev.read_jsonl(prepared / "repeat_requests.jsonl")
            self.assertEqual(main_requests[:576], repeat_requests)
            for shard in range(4):
                self.assertEqual(main_requests[shard::4][:144], repeat_requests[shard::4])
            for req in main_requests:
                body = json.loads(req["messages"][1]["content"].split("\n", 1)[0])
                self.assertTrue(all("decision" not in e for e in body["examples"]))
            paths = write_runs(root, prepared / "stage1_requests.jsonl", cases)
            repeats = write_runs(root, prepared / "repeat_requests.jsonl", cases, realization=1)
            with contextlib.redirect_stdout(io.StringIO()):
                fm.stage2(SimpleNamespace(prepared=prepared, predictions=paths, output=second))
            stage2_paths = write_runs(root, second / "requests.jsonl", cases, realization=2)
            args = SimpleNamespace(prepared=prepared, stage1_predictions=paths, repeat_predictions=repeats,
                                   stage2=second, stage2_predictions=stage2_paths, output=output)
            with contextlib.redirect_stdout(io.StringIO()):
                fm.evaluate(args)
            report = json.loads((output / "summary.json").read_text())
            self.assertEqual(report["paired_effects"]["neutral_compiler_minus_policy_compiler"]["difference"], 0)
            self.assertEqual(report["cost"]["arms"]["training_constant"]["shared_across_three_policies"]["logical_calls"], 0)
            self.assertEqual(report["repeat_background"]["guarded_direct"]["same_prompt"]["possible_pairs"], 192)
            self.assertEqual(len(ev.read_jsonl(output / "outcomes.jsonl")), 64 * 3 * 6)
            bad = root / "bad"
            bad.mkdir()
            args.output = root / "rejected"
            args.repeat_predictions = write_runs(bad, prepared / "repeat_requests.jsonl", cases,
                                                realization=3, corrupt_repeat=True)
            with self.assertRaisesRegex(ValueError, "cached"):
                fm.evaluate(args)
            original = (prepared / "references.jsonl").read_text()
            (prepared / "references.jsonl").write_text(original + "\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                fm.load_prepared(prepared)

    def test_crossings_use_matched_pair_rates_not_any_change(self):
        attrs = dict.fromkeys(pr.FIELDS, 0)
        wrong = {**attrs, "intent": 1}
        case = {"id": "fixture", "group": "one", "attributes": attrs}
        first, repeat = {}, {}
        for method in fm.EXPOSED:
            for index, policy in enumerate(pr.POLICIES):
                key = f"fixture/{policy}/{method}"
                for source, value in ((first, attrs if index == 0 else wrong), (repeat, attrs if index == 0 else wrong)):
                    source[key] = {"metadata": {"case_id": "fixture"}, "status": "ok",
                                   "response": json.dumps({"attributes": value})}
        summary, _ = fm.repeat_comparison([case], first, repeat, replicates=20)
        direct = summary["direct"]
        self.assertEqual(direct["cross_policy"]["correctness_crossings"], 4)
        self.assertEqual(direct["same_prompt"]["correctness_crossings"], 0)
        self.assertAlmostEqual(direct["correctness_crossings_cross_minus_repeat"]["difference"], 2 / 3)
        first["fixture/offense/direct"]["response"] = "invalid"
        summary, _ = fm.repeat_comparison([case], first, repeat, replicates=20)
        self.assertEqual(summary["direct"]["complete_panel_coverage"], 0)
        self.assertIsNone(summary["direct"]["correctness_crossings_cross_minus_repeat"]["difference"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
