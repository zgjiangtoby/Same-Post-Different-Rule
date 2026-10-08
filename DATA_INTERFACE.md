# External data interface

This source release does not contain dataset rows, native labels, splits, or derived features. Place legally acquired original files at the locations below, or pass another local directory through `formal.py prepare --source`. No script downloads data.

## SBIC v2

The official [SBIC v2 archive](https://maartensap.com/social-bias-frames/SBIC.v2.tgz) supplies three native aggregate CSV files. Keep their original bytes and partition memberships:

```text
paper1/data/sbic/
  SBIC.v2.agg.trn.csv
  SBIC.v2.agg.dev.csv
  SBIC.v2.agg.tst.csv
```

The loader requires `post`, `offensiveYN`, `intentYN`, and `sexYN`. It rejects duplicate exact text within an aggregate and unexpected empty text. Reference fields are converted from existing aggregate means with the threshold/tie handling in `paper1/prepare.py`; no record is relabeled. Check source hashes against the values in `README.md`, then use:

```bash
export PYTHONPATH=paper1:shared
python paper1/formal.py prepare --source paper1/data/sbic --output paper1/prepared/sbic_test_formal_v1
```

## HateCheck

Acquire `test_suite_cases.csv` from the [pinned official commit](https://raw.githubusercontent.com/paul-rottger/hatecheck-data/3490854f969111a32c55c05f0d9788b471337fa9/test_suite_cases.csv). The frozen loader requires the exact metadata manifest bytes adjacent to that CSV; copy the included metadata file into your local input directory:

```bash
export PYTHONPATH=paper1:shared
mkdir -p paper1/data/hatecheck
cp metadata/hatecheck_source_manifest.json paper1/data/hatecheck/source_manifest.json
sha256sum paper1/data/hatecheck/test_suite_cases.csv paper1/data/hatecheck/source_manifest.json
python paper1/formal.py prepare --dataset hatecheck --source paper1/data/hatecheck --output paper1/prepared/hatecheck_test_formal_v1
```

The CSV SHA-256 must be `ecf0dc1e93fccc90b6f880e632b19e635c0f0a68b3157915a9e20b6938f121cf`. The manifest pins commit `3490854f969111a32c55c05f0d9788b471337fa9`, source URLs, checksums, the released schema, and aggregate intake counts only; it has no case text or per-record judgments. The code uses existing `label_gold` and `direction` to make the two native attributes and carries `templ_id` as a block ID. It checks `case_id`, `test_case`, `label_gold`, `direction`, `templ_id`, and `functionality`, rejects duplicate IDs/exact texts, and requires 3,728 released cases. The official README and LICENSE are linked in the manifest for source terms; neither is distributed here.

Both preparers write references and requests containing post text and labels. Keep `paper1/prepared/`, `paper1/runs/`, and `paper1/results/` outside any public source upload. They are ignored by `.gitignore`.
