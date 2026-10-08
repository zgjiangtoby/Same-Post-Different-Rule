# Same Post, Different Rule — source release

Source for the 2026-10-03 Revised manuscript, *Same Post, Different Rule: Measuring Content-Attribute Stability in Web Moderation*. This is a source-only project: it contains the original formal measurement, the subsequent policy-function × expression study, adjusted native-template intervals, and the descriptive crossing analysis. It contains no corpus records, labels, prepared requests, predictions, results, model weights, or manuscript PDF. The paper's results are **not** reproduced by the included synthetic checks.

Run commands from this directory with Python 3.11. Set `PYTHONPATH=paper1:shared` so sibling modules can import each other. The source Python files and the two protocol documents are copied byte for byte from the frozen reproducibility package; `metadata/hatecheck_source_manifest.json` is the original metadata manifest at a new location. The formal preparation binds the exact `paper1/FORMAL_PROTOCOL_V1.md` bytes. Recreate preparations with this release; do not mix prepared artifacts from another source/protocol binding. `paper1/POLICY_FACTORIAL_PLAN_20261002.md` describes the later follow-up, which was designed after the original results.

## Environment and inputs

The recorded environment is Python 3.11, PyTorch 2.9.1, Transformers 4.57.2, tokenizers 0.22.1, NumPy 1.26.4, and regex 2025.11.3; see `environment/requirements.txt`. Full generation uses CUDA, bfloat16, SDPA, local tokenizers, greedy decoding, seed 17, batch 8, 8,192 input tokens, and 128 new tokens. Install packages and a compatible CUDA build of PyTorch in your own environment. No package, dataset, or model download is needed for the CPU checks below.

Acquire the existing datasets under their own terms; no new annotation is involved:

| Input | Official source and pinned revision | Local files consumed by the code |
|---|---|---|
| SBIC v2 aggregate | [Official archive](https://maartensap.com/social-bias-frames/SBIC.v2.tgz), archive SHA-256 `07cabae1ee0289392ea11b494c0012578bd39a582d58dc5ebd8edc3deda6bc5b`; [data statement](https://maartensap.com/social-bias-frames/DATASTATEMENT.MD) | `paper1/data/sbic/SBIC.v2.agg.{trn,dev,tst}.csv` |
| HateCheck | [Official repository](https://github.com/paul-rottger/hatecheck-data/tree/3490854f969111a32c55c05f0d9788b471337fa9), commit `3490854f969111a32c55c05f0d9788b471337fa9` | `paper1/data/hatecheck/test_suite_cases.csv` and `paper1/data/hatecheck/source_manifest.json` |

The SBIC CSV SHA-256 values are `d60770d7b1e37fdc8260029ee59c3759ae7329de378c7e7986c528c625a6260f` (train), `b8fcf3a992a0de47c41191d00058c659b3e2744e2d051bbbb23d458deb5b35a2` (development), and `12982a534e60a2275f134d07038d1bfd46b252835f269959ec7ed6216f5ee53a` (test). HateCheck's [CSV at the pinned commit](https://raw.githubusercontent.com/paul-rottger/hatecheck-data/3490854f969111a32c55c05f0d9788b471337fa9/test_suite_cases.csv) has SHA-256 `ecf0dc1e93fccc90b6f880e632b19e635c0f0a68b3157915a9e20b6938f121cf`. Copy `metadata/hatecheck_source_manifest.json` to `paper1/data/hatecheck/source_manifest.json` after acquiring the CSV. The manifest is source metadata, not a dataset. See [DATA_INTERFACE.md](DATA_INTERFACE.md) for exact file schemas, placement, and checksum commands. The importer checks the CSV hash, revision, required native columns (`case_id`, `test_case`, `label_gold`, `direction`, `templ_id`, `functionality`), and 3,728 rows. SBIC requires `post`, `offensiveYN`, `intentYN`, and `sexYN`; its original train/dev/test partitions stay fixed. Exact original text defines resampling groups. HateCheck's native `templ_id` defines template blocks. Neither task uses reply graphs, historical extension groups, or new judgments.

Use these local model snapshots (weights are not included):

| Model ID | Exact revision | Example local path |
|---|---|---|
| [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | `a09a35458c702b33eeacc393d103063234e8bc28` | `models/qwen2_5_7b_instruct` |
| [`ibm-granite/granite-3.3-8b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct) | `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b` | `models/granite_3_3_8b_instruct` |

The preparation output contains the post text and existing labels, so keep all generated directories private. Paths below are relative examples and must be fresh. The source scripts reject existing output directories/files where required.

## Offline CPU checks

These use only inline synthetic fixtures. They do not load source datasets or model files, use a GPU, or generate manuscript numbers.

```bash
sha256sum -c SHA256SUMS
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=paper1:shared python -m unittest discover -s paper1 -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=paper1:shared python paper1/test_adjusted_template_intervals.py
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=paper1:shared python paper1/prepare.py self-test
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=paper1:shared python paper1/crossing_structure.py --self-test
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 python shared/generate.py --self-test
```

`SHA256SUMS` is the release checksum manifest. The adjusted-template test is a standalone script and is not invoked by `unittest discover`. `shared/test_tokenizer.py` and `policy_factorial.py token-check` need the pinned local tokenizer snapshots and are separate preflight checks for the full run.

## Original SBIC and HateCheck measurement

The full definitions, prompts, frozen 64-group repeats, endpoints, bootstrap families, and run order are in `paper1/FORMAL_PROTOCOL_V1.md`. The following commands use its actual CLI flags. Prepare both tasks on CPU, then validate their request files:

```bash
export PYTHONPATH=paper1:shared
python paper1/formal.py prepare --source paper1/data/sbic --output paper1/prepared/sbic_test_formal_v1
python paper1/formal.py prepare --dataset hatecheck --source paper1/data/hatecheck --output paper1/prepared/hatecheck_test_formal_v1
for task in sbic_test_formal_v1 hatecheck_test_formal_v1; do
  python shared/generate.py --input "paper1/prepared/$task/stage1_requests.jsonl" --check-input
  python shared/generate.py --input "paper1/prepared/$task/repeat_requests.jsonl" --check-input
done
```

For each task/model pair, set the variables as below and run the subsequent block. Complete all main stage-one shards before starting fresh repeat processes. Choose an available GPU for each shard; `gpu=0` is merely an example. Four logical shards are required even if one GPU executes them sequentially. Keep command receipts, distinct main/repeat runs, and all failures.

```bash
# Choose ONE pair, run the block, then repeat with the other three pairs.
export PYTHONPATH=paper1:shared
task=sbic_test_formal_v1
model_path=models/qwen2_5_7b_instruct
model_revision=a09a35458c702b33eeacc393d103063234e8bc28
run=paper1/runs/qwen_test_formal_v1
stage2=paper1/prepared/sbic_test_formal_qwen_stage2_v1
gpu=0
mkdir -p "$run"
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" python shared/generate.py --input "paper1/prepared/$task/stage1_requests.jsonl" --output "$run/stage1_${shard}.jsonl" --model "$model_path" --model-revision "$model_revision" --batch-size 8 --max-input-tokens 8192 --max-new-tokens 128 --seed 17 --num-shards 4 --shard-index "$shard" || exit 1
done
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" python shared/generate.py --input "paper1/prepared/$task/repeat_requests.jsonl" --output "$run/repeat_${shard}.jsonl" --model "$model_path" --model-revision "$model_revision" --batch-size 8 --max-input-tokens 8192 --max-new-tokens 128 --seed 17 --num-shards 4 --shard-index "$shard" || exit 1
done
python paper1/formal.py stage2 --prepared "paper1/prepared/$task" --predictions "$run"/stage1_{0,1,2,3}.jsonl --output "$stage2"
stage2_predictions=()
if [[ -s "$stage2/requests.jsonl" ]]; then
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$gpu" python shared/generate.py --input "$stage2/requests.jsonl" --output "$run/stage2_${shard}.jsonl" --model "$model_path" --model-revision "$model_revision" --batch-size 8 --max-input-tokens 8192 --max-new-tokens 128 --seed 17 --num-shards 4 --shard-index "$shard" || exit 1
    stage2_predictions+=("$run/stage2_${shard}.jsonl")
  done
fi
python paper1/formal.py evaluate --prepared "paper1/prepared/$task" --stage1-predictions "$run"/stage1_{0,1,2,3}.jsonl --repeat-predictions "$run"/repeat_{0,1,2,3}.jsonl --stage2 "$stage2" --stage2-predictions "${stage2_predictions[@]}" --output "$run/evaluation"
```

For the other pairs, change variables only: SBIC/Granite uses `task=sbic_test_formal_v1`, `run=paper1/runs/granite_test_formal_v1`, `stage2=paper1/prepared/sbic_test_formal_granite_stage2_v1`; HateCheck/Qwen uses `task=hatecheck_test_formal_v1`, `run=paper1/runs/qwen_hatecheck_formal_v1`, `stage2=paper1/prepared/hatecheck_formal_qwen_stage2_v1`; HateCheck/Granite uses `task=hatecheck_test_formal_v1`, `run=paper1/runs/granite_hatecheck_formal_v1`, `stage2=paper1/prepared/hatecheck_formal_granite_stage2_v1`. For Granite set `model_path=models/granite_3_3_8b_instruct` and `model_revision=51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`. The formal evaluator writes scored cases, repeats, and summaries to each `evaluation/` directory.

## Function × expression follow-up

This uses the already prepared SBIC/HateCheck references, three rules, three renderings, one neutral record, and fresh identical-message realizations A and B. The source-defined selection is 512 exact-text groups per task. See `paper1/POLICY_FACTORIAL_PLAN_20261002.md` for its statistical families and protocol. Token checks require the local model snapshots:

```bash
export PYTHONPATH=paper1:shared
PYTHONPATH=paper1:shared python paper1/policy_factorial.py prepare --sbic-prepared paper1/prepared/sbic_test_formal_v1 --hatecheck-prepared paper1/prepared/hatecheck_test_formal_v1 --output paper1/prepared/policy_factorial_v1
PYTHONPATH=paper1:shared python paper1/policy_factorial.py token-check --prepared paper1/prepared/policy_factorial_v1 --model models/qwen2_5_7b_instruct --output paper1/prepared/policy_factorial_v1/token_audit_qwen.json
PYTHONPATH=paper1:shared python paper1/policy_factorial.py token-check --prepared paper1/prepared/policy_factorial_v1 --model models/granite_3_3_8b_instruct --output paper1/prepared/policy_factorial_v1/token_audit_granite.json
```

Complete all eight generation jobs before evaluating either model. Each A/B job is a fresh process and output file. The shown `gpu` value can be changed to an available device; keep the same two-shard layout and settings.

```bash
export PYTHONPATH=paper1:shared
gpu=0
for model_key in qwen granite; do
  case "$model_key" in
    qwen) model_path=models/qwen2_5_7b_instruct; model_revision=a09a35458c702b33eeacc393d103063234e8bc28 ;;
    granite) model_path=models/granite_3_3_8b_instruct; model_revision=51dd4bc2ade4059a6bd87649d68aa11e4fb2529b ;;
  esac
  run="paper1/runs/${model_key}_policy_factorial_v1"
  mkdir -p "$run"
  for realization in a b; do
    for shard in 0 1; do
      CUDA_VISIBLE_DEVICES="$gpu" python shared/generate.py --input paper1/prepared/policy_factorial_v1/requests.jsonl --output "$run/${realization}_${shard}.jsonl" --model "$model_path" --model-revision "$model_revision" --batch-size 8 --max-input-tokens 8192 --max-new-tokens 128 --num-shards 2 --shard-index "$shard" --seed 17 || exit 1
    done
  done
done
for model_key in qwen granite; do
  run="paper1/runs/${model_key}_policy_factorial_v1"
  PYTHONPATH=paper1:shared python paper1/policy_factorial.py evaluate --prepared paper1/prepared/policy_factorial_v1 --realization-a "$run/a_0.jsonl" "$run/a_1.jsonl" --realization-b "$run/b_0.jsonl" "$run/b_1.jsonl" --output "paper1/results/${model_key}_policy_factorial_20261002_v1_evaluation" || exit 1
done
```

## Post-analysis modules

After all original and factorial evaluations, reconstruct the adjusted HateCheck template-block intervals and the descriptive erroneous-field/rule crossing structure. These scripts consume the exact relative evaluation paths above, validate their input summaries, and write fresh output directories. They use existing outcomes and native template IDs only.

```bash
export PYTHONPATH=paper1:shared
PYTHONPATH=paper1:shared python paper1/adjusted_template_intervals.py --root . --output paper1/results/adjusted_template_intervals_reproduced
PYTHONPATH=paper1:shared python paper1/crossing_structure.py --root . --output paper1/results/crossing_structure_reproduced
```

The original formal intervals use exact-text groups. The adjusted script additionally resamples whole HateCheck native templates while preserving group weights. Its family-adjusted intervals and the crossing analysis are described in the Revised manuscript. CPU checks alone only test code behavior on synthetic fixtures; full results require the pinned data, model snapshots, generation, and all scored outputs.
