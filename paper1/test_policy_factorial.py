import json
from itertools import product
import unittest
import policy_factorial as pf


class FactorialChecks(unittest.TestCase):
    def test_truth_table_matches_each_original_function(self):
        for dataset in ("sbic", "hatecheck"):
            fields, policies, _, decide, _ = pf.task(dataset)
            for policy in policies:
                description = pf.rendered_policy(dataset, policy, "truth_table")
                table = json.loads(description.split(": ", 1)[1])
                self.assertEqual(len(table), 2 ** len(fields))
                for row, values in zip(table, product((0, 1), repeat=len(fields))):
                    attributes = dict(zip(fields, values))
                    self.assertEqual({k: row[k] for k in fields}, attributes)
                    self.assertEqual(row["flag"], decide(attributes, policy))

    def test_truth_table_only_flag_bits_change(self):
        for dataset in ("sbic", "hatecheck"):
            texts = [pf.rendered_policy(dataset, p, "truth_table") for p in pf.task(dataset)[1]]
            self.assertEqual(len({len(t) for t in texts}), 1)
            import re
            self.assertEqual(len({re.sub(r'"flag":[01]', '"flag":X', t) for t in texts}), 1)

    def test_prompt_original_and_label_separation(self):
        case = {"id": "sbic:dummy", "group": "dummy", "partition": "test", "text": "A quoted post.",
                "attributes": {"offensive": 1, "intent": 1, "sexual": 1}}
        bank = [{**case, "id": "train:demo", "partition": "train", "text": "Training text."}]
        old = pf.pr.request(case, bank, "full_record", "offense", demo_verdicts=False)
        new = pf.request(case, bank, "sbic", "offense", "original")
        self.assertEqual(old["messages"], new["messages"])
        changed = pf.request(case, bank, "sbic", "intentional", "truth_table")
        content = json.loads(changed["messages"][1]["content"].rsplit("\n", 1)[0])
        self.assertNotIn("attributes", content)
        self.assertNotIn("decision", content["examples"][0])
        self.assertEqual(content["active_policy"], content["examples"][0]["policy"])
        self.assertEqual(len(pf.construct([case], bank, "sbic")), 10)

    def test_failure_is_not_a_semantic_crossing(self):
        gold = {"hateful": 1, "directed_hate": 0}
        self.assertEqual(pf.rate([(gold, None), (gold, {"hateful": 0, "directed_hate": 0})], gold),
                         {"possible": 2, "valid": 1, "crossings": 1, "attribute_changes": 1})
        self.assertIsNone(pf.parse_attributes('{"attributes":{"hateful":1,"directed_hate":2}}', gold))
        self.assertEqual(pf.parse_attributes('{"attributes":{"hateful":1,"directed_hate":0}}', gold), gold)


if __name__ == "__main__":
    unittest.main()
