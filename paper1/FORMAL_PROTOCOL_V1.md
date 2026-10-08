# SBIC policy-associated measurement study, formal protocol v1

Design date: 2026-09-30, Asia/Shanghai. Actual scientific design and implementation route: GPT-6-Astra/xhigh. Bounded execution/checks use GPT-6-Sol/medium. Status: locked design candidate for independent pre-inference review; no formal test model outcome has been consulted in designing this protocol. A preparation binds the exact bytes of this document, `formal.py`, `hatecheck_task.py`, `prepare.py`, and `evaluate.py`. Correct defects before preparation/inference and record them; never silently reinterpret a completed run.

## Question, scope, and claim map

The question is whether a changed downstream Boolean decision rule changes agreement with an unchanged content-attribute reference scheme, beyond observed same-prompt fresh-generation disagreement, and what accuracy/cost tradeoff follows from reusing a neutral measurement. This is a measurement study, not a new prompting or compilation algorithm. The four 64-development-post pilots are exploratory antecedents. Their unfavorable outcomes remain in `PILOT_FINDINGS.md`.

| Claim to test | Essential evidence | Failure interpretation |
|---|---|---|
| C1: Exposing a fixed-codebook measurement to different rules produces correctness-changing differences beyond measured runtime disagreement | Matched repeated first-stage direct, guarded-direct, and active-record outputs; valid-pair exact-correctness crossings and paired uncertainty, in both fixed models | Equal or smaller cross-rule disagreement, low coverage, or imprecise intervals do not establish a controlled effect; report the bound |
| C2: Neutral reuse has a signed accuracy and cost tradeoff relative to rule-exposed measurement | Primary mean joint-accuracy difference, neutral minus active compilation; strong direct, guarded, full-information, and zero-call constant controls | A zero or negative accuracy difference blocks a neutral-remedy claim, even if reuse is cheaper or perfectly stable |

At most these two claims are primary. An accuracy improvement is not required for an honest measurement result. The AP joint-superiority branch is withdrawn: the same-record deterministic compiler pointwise upper-bounds the extra model decision on joint accuracy. No AP, mask-only, or freeze-only expansion is included. No new annotation, reply/parent inputs, conversational tasks, historical relationship data, private data upload, or test-based prompt search is permitted.

## Population and fixed references

Use the official released `SBIC.v2.agg.{trn,dev,tst}.csv` bytes already acquired, and retain their native partition membership. Hash exact original text without normalization; preserve each selected original text, native aggregate-file name, and 1-based source data-row position. Duplicate text inside an aggregate is an error. A novel source schema or empty text requiring row-identity reconstruction stops preparation for an explicit source audit.

Use `offensiveYN`, `intentYN`, `sexYN`; retain the pilot's definitions and projection: means greater than .5 positive, less than .5 negative, ties within `math.isclose(value,.5,abs_tol=1e-10)` and missing/nonfinite means unresolved. Values outside [0,1] are errors. These are existing dataset judgments, not newly independent/type-blind labels or access to private intent. Policy references are deterministic functions of those judgments, not independently adjudicated moderation outcomes.

The main population is **all** complete-reference official test exact-text groups after excluding exact matches in the official development split. Training eligibility for both demonstrations and the constant baseline excludes every exact development/test text, then requires complete references. Thus no selected test text is supplied as a training demonstration or counted in the fitted constant even if a native train duplicate exists. No partition is reassigned. Preparation retains per-field tie/missing/resolved status, length, eligibility, original row identity, and development overlap for every nonempty test row. Ambiguous/missing references are outside the three-field primary estimand and must appear in coverage; input-length rejection, parse failure, or malformed model outputs remain inside selected-case denominators as failures.

The evaluation order is fixed by SHA256(`paper1-case-v1:` + exact_text_group). There is no outcome-adaptive subsampling or favorable-stratum selection. Source and output hashes, eligibility counts, training-only modal-vector counts, and request budgets are recorded before inference. Test reference distributions must not be used to tune prompts, thresholds, policies, or selected cases.

## Frozen prompts, arms, and models

Reuse the exact verdict-free v3 prompt construction, fixed codebook, four training demonstration patterns `(O,I,S)=(0,0,0),(1,1,0),(1,0,0),(0,0,1)`, deterministic bank order, and policy strings in `prepare.py`. Every demonstration retains its post and attributes; exposed demonstrations retain policy strings but no `decision` key. Query references never enter requests. The three rules are `O`, `S`, and `O AND I`, in the existing order offense, sexual, intentional. Existing guard wording and provisional-record wording remain unchanged.

| Arm | What is generated | Logical calls/post across three policies |
|---|---|---:|
| Direct | Attributes and verdict under each active rule | 3 |
| Guarded direct | Same with the existing invariant-codebook reminder | 3 |
| Full-information two-stage | Active-rule attribute extraction, then same-rule provisional-record recheck plus verdict | 6 |
| Active compiler (`policy_compiler`) | Known rule applied to each active-rule extracted record; shares extraction with full-information | 3 standalone; shared in the study |
| Neutral compiler | One policy-free extracted record, compiled under all three rules | 1 amortized; 3 if independently re-extracted per rule |
| Training constant | Most frequent whole `(O,I,S)` vector in eligible training records; ties resolve to lexicographically smallest vector | 0 |

The bank selection and constant use training labels only. The constant is not the vector with the highest held-out score. Active and neutral compilers' joint correctness equals exact record correctness; these are one fact, not independent confirmations. Their rule consistency, and neutral across-rule stability, are construction properties.

Use both already acquired checkpoints, without family pooling:

- `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`, local `models/qwen2_5_7b_instruct`.
- `ibm-granite/granite-3.3-8b-instruct`, revision `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`, local `models/granite_3_3_8b_instruct`.

The frozen generator is `shared/generate.py`, SHA256 `f81bc57760070dbef65aaf60fa71ce3c70aa71f345224b7a4932e493f170ef6c`. Use the validated sunproj environment, bfloat16, SDPA, slow tokenizer, greedy decoding, thinking disabled, seed 17, batch size 8, max input 8192, max new tokens 128, four interleaved shards indexed 0–3. Freeze and retain library versions, model config hashes, effective batch sizes, GPU model, process starts, and complete command/log receipts. Equal ceilings are not equal token costs.

## Same-prompt repeats with matched opportunities

Select exactly 64 eligible test groups by SHA256(`paper1-repeat-v1:` + exact_text_group), without looking at labels or outcomes after the eligibility gate. For each selected group, include the direct, guarded-direct, and full-record first-stage prompts under all three rules: 576 requests. Place these requests first in the main stage-one file. Write the identical prefix to `repeat_requests.jsonl` with unchanged IDs/messages/metadata. The rest of main stage one follows afterward.

Since 576 is divisible by 4 shards × batch 8, the prefix comprises exactly the first 18 complete batches in each main shard. Fresh repeat runs use the identical 18 batches in the same shard positions. Run them in new processes and new output files with the same seed/model/decoding after main stage one, without `--resume`. There is no cross-process cache and no duplicate prompt inside the repeated file. Never selectively regenerate disagreements, invalid outputs, or long inputs. If interrupted execution changes repeat-prefix batch composition, retain it as a failed runtime check and rerun a whole predeclared realization in a fresh versioned directory, before reading disagreements; do not pool or choose favorable realizations. The executor must retain fresh-start command receipts because the immutable runner does not itself encode every resume history.

For each selected group and prompt type, call the two three-policy realizations A and B. Across-rule comparison uses the three unordered policy pairs within A and the three within B: six opportunities. Same-prompt comparison uses A versus B under each of the three policies: three opportunities. For each pair, exact-correctness crossing is 1 precisely when one schema-valid complete attribute vector equals the reference and the other schema-valid vector does not. Attribute disagreement is a separate endpoint; both-wrong different records count as disagreement but not exact-correctness crossing.

The matched group-level difference is **crossing_count_across/6 − crossing_count_repeat/3**, on groups with all six attribute outputs valid. This is a difference of per-opportunity rates, never any-of-three versus one opportunity. Report the all-six-valid panel size/coverage, raw possible/valid pair counts, event rates on valid pairs, and conservative observed-event rates on all pairs. Invalid outputs are not relabeled as semantic crossings. If all-six-valid coverage is below 95%, restrict inference explicitly to the valid panel and withhold a population-wide controlled-effect claim. The repeat is an empirical runtime background under one frozen layout, not a guarantee of universal determinism or identification of an internal mechanism. Full-information stage-two drift is descriptive; it has no independent matched stage-two repeat.

## Endpoints, uncertainty, and decision rules

The primary accuracy estimand per model is `mean_i [joint(neutral,i) − mean_policy joint(active,i,policy)]`, signed neutral minus active. The full denominator is every selected test group and every rule, including failed outputs. Compiler joint accuracy is identical to exact three-field accuracy. The primary two-model family has two contrasts; report 10,000-replicate paired exact-original-text-group percentile bootstrap pointwise 95% and Bonferroni 97.5% intervals (seed 20260930). Preserve all three policies and all methods for a group on every draw. No test-data-dependent stopping or seed selection.

The repeat correctness-crossing family has six contrasts: two models × three exposed first-stage prompt types. Report pointwise 95% and Bonferroni 99.1667% intervals using the same group resampling. A positive family-adjusted lower bound supports excess crossing for that specific model/interface. Require that criterion for guarded direct in both models, with at least 95% complete-panel coverage, before claiming persistence beyond the guard across both families. A positive unguarded-only result must be described that way. Attribute-disagreement intervals, per-policy signed accuracy intervals, full-information crossings, reference-rule-flip strata, and all-rule success are descriptive, with no discovery claims from selecting their favorable cells.

The remedy threshold is a **2 percentage point lower-bound gain** in mean joint accuracy. A general neutral-remedy claim requires the primary adjusted lower bound to exceed .02 in both models and the same lower-bound criterion against direct, guarded direct, full-information, and training constant in both models. The latter is a separate eight-contrast family with Bonferroni 99.375% intervals. Passing one model, one policy, or all-rule success cannot replace these gates. These are prespecified practical and inferential criteria, not an assertion that the study is powered to meet them. Families answer separate questions; no omnibus familywise claim across all readouts is made.

Also report per-field accuracy/coverage; schema-valid whole-vector coverage; verdict-only and joint accuracy; record/verdict consistency with valid denominators; input-too-long counts; exact-correctness crossings and attribute disagreements; all-rule success; positive/negative reference rule-flip strata; and actual logical/physical calls, cache reuse, input/output tokens, and measured inference batch seconds. Standalone costs and amortized policy reuse are separate. Compile control and training constant cost zero extra model calls. Repeat cost is separately charged, not hidden in a system arm.

The repeat panel of 64 groups may be imprecise. An interval including zero is inconclusive, not evidence of no effect. All-test accuracy precision depends on the paired difference variance, not just N: a worst-case normal approximation has adjusted half-width at most about `2.24/sqrt(N)` (3.4 pp if N≈4,345), often smaller with paired agreement. A 1,024-group shortcut would have a roughly 7.0 pp worst-case half-width, so it cannot be asserted adequate for a 2 pp remedy threshold without an independently justified variance/power model. Retain all eligible test groups; report precision honestly if no remedy gate is cleared.

## Compute budget and run order

The official-source metadata audit gives N=4,336 eligible SBIC groups: 4,691 test records, 4,345 complete before overlap removal, and 9 complete exact development overlaps removed (10 overlaps total). Eligible training has 32,324 records and its modal vector is `(0,0,0)` with 16,058 records. These are source eligibility/training receipts, not formal model outcomes.

For N eligible groups: stage one `10N`, stage two at most `3N`, one fresh repeat `576` requests/model. SBIC requires stage one 43,360, stage two at most 13,008, and repeat 576: **56,944/model; 113,888 across both models**. Hard token ceilings are `8192 × requests` input tokens and `128 × requests` output tokens; actual counts and long-input rejection are recorded. Estimate elapsed time from each model's completed pilot generation batch seconds divided by physical requests, then apply it to formal request counts; record estimates in `../ops/PAPER1_FORMAL_EXECUTION_ESTIMATE.md` before launch. Use the formal 13-request design rather than the abandoned pilot's 22-request grid. Model loading/host work, padding, and longer inputs make this an estimate, not an SLA. The four local GPUs are preferred after approved occupancy checks; do not stop unrelated jobs.

| Milestone | Required work | Gate |
|---|---|---|
| M0 | Synthetic CPU regression, old-pilot reproduction, formal request/hash/batch-prefix checks | No output leakage; all failures retain denominator |
| M1 | Independent fresh-agent protocol/code review, official-source eligibility receipt and token-length preflight | Resolve substantive defects before unused-test inference |
| M2 | Both frozen model main stage-one runs and whole fresh prefix repeat | Complete bound manifests and matched fresh-start receipts |
| M3 | Build only full-information stage two, run both models, evaluate once complete | No partial outcome inspection/tuning |
| M4 | Independent result-to-claim and statistics review | Keep unfavorable signs, nulls, failures, and model differences |

Must-run: these milestones. Cut: AP expansions, extra prompt searches, new human labels, new model families, extra policy families, automatic favorable-subgroup follow-ups. No threshold or prompt changes after M2 starts. A narrow or negative result does not trigger more experiments automatically.

## Exact executor commands

Run from the project root. CPU-only construction and validation:

```bash
python -m unittest discover -s paper1 -p 'test_*.py' -v
python paper1/formal.py prepare --source paper1/data/sbic --output paper1/prepared/sbic_test_formal_v1
python shared/generate.py --input paper1/prepared/sbic_test_formal_v1/stage1_requests.jsonl --check-input
python shared/generate.py --input paper1/prepared/sbic_test_formal_v1/repeat_requests.jsonl --check-input
```

For each Qwen shard `s=0,1,2,3`, after review and approved GPU occupancy check, substitute s literally or use a shell loop. Use a distinct output/log per shard and retain the actual command. Main and repeat phases must finish separately; never place both on the same GPU concurrently:

```bash
CUDA_VISIBLE_DEVICES="$s" python shared/generate.py --input paper1/prepared/sbic_test_formal_v1/stage1_requests.jsonl --output "paper1/runs/qwen_test_formal_v1/stage1_${s}.jsonl" --model models/qwen2_5_7b_instruct --model-revision a09a35458c702b33eeacc393d103063234e8bc28 --batch-size 8 --max-input-tokens 8192 --max-new-tokens 128 --seed 17 --num-shards 4 --shard-index "$s"
CUDA_VISIBLE_DEVICES="$s" python shared/generate.py --input paper1/prepared/sbic_test_formal_v1/repeat_requests.jsonl --output "paper1/runs/qwen_test_formal_v1/repeat_${s}.jsonl" --model models/qwen2_5_7b_instruct --model-revision a09a35458c702b33eeacc393d103063234e8bc28 --batch-size 8 --max-input-tokens 8192 --max-new-tokens 128 --seed 17 --num-shards 4 --shard-index "$s"
python paper1/formal.py stage2 --prepared paper1/prepared/sbic_test_formal_v1 --predictions paper1/runs/qwen_test_formal_v1/stage1_{0,1,2,3}.jsonl --output paper1/prepared/sbic_test_formal_qwen_stage2_v1
CUDA_VISIBLE_DEVICES="$s" python shared/generate.py --input paper1/prepared/sbic_test_formal_qwen_stage2_v1/requests.jsonl --output "paper1/runs/qwen_test_formal_v1/stage2_${s}.jsonl" --model models/qwen2_5_7b_instruct --model-revision a09a35458c702b33eeacc393d103063234e8bc28 --batch-size 8 --max-input-tokens 8192 --max-new-tokens 128 --seed 17 --num-shards 4 --shard-index "$s"
python paper1/formal.py evaluate --prepared paper1/prepared/sbic_test_formal_v1 --stage1-predictions paper1/runs/qwen_test_formal_v1/stage1_{0,1,2,3}.jsonl --repeat-predictions paper1/runs/qwen_test_formal_v1/repeat_{0,1,2,3}.jsonl --stage2 paper1/prepared/sbic_test_formal_qwen_stage2_v1 --stage2-predictions paper1/runs/qwen_test_formal_v1/stage2_{0,1,2,3}.jsonl --output paper1/runs/qwen_test_formal_v1/evaluation
```

Granite uses exactly those commands with run/preparation `qwen` replaced by `granite`, model path `models/granite_3_3_8b_instruct`, revision `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`. If all first-stage records fail and stage two is empty, omit generation and omit `--stage2-predictions`; keep all failures in evaluation. Do not edit the runner or reuse cached predictions across fresh realizations.

## HateCheck native-label functional probe, separately scoped

The approved official release is `paul-rottger/hatecheck-data` commit `3490854f969111a32c55c05f0d9788b471337fa9`; local `source_manifest.json` binds the CSV and official README. This is a template-based functional suite, not a second natural-post sample and not a source of SBIC intention/sexual labels. Official `label_gold` gives hateful/non-hateful; official `direction` is directed/general for hateful cases and `-` for non-hateful cases. The acquired schema contains 945 directed hateful, 1,618 general hateful, and 1,165 non-hateful cases. These are intake label counts, not model results.

A separate task uses **H = existing hateful label** and **D = hateful content directed at an individual as a protected-group member**. On hateful cases D follows the official directed/general label. On non-hateful cases D=0 follows the explicit conjunction in its definition, not an imputed direction judgment. Its frozen rules are H, D, and H AND NOT D. The fixed codebook asks those two fields and the verdict; its hate definition follows the original paper's protected-group abuse definition, and its direction distinction follows the official README. See [Röttger et al., ACL 2021, §1](https://aclanthology.org/2021.acl-long.4.pdf) and the locally pinned README. This supports only a cross-codebook functional measurement claim, not transfer of SBIC intent/sexual accuracy. No SBIC field is fabricated from functionality names.

The release has no native train split. The probe is therefore zero-shot, with no demonstration bank or training-constant arm. Use all 3,728 original released cases, each of which has unique exact text and complete compatible H/D references. Preserve `case_id`, `templ_id`, functionality, exact original test-case text, native labels and source-row membership. There are 845 native templates; related templates can remain dependent even after template blocking. No case or template is reassigned as training/development data. Do not feed `ref_case_id`, `ref_templ_id`, related cases, template identity, target metadata, or functionality metadata to the model. Functionality-stratified reporting uses existing official labels. No relationship graph or conversational context is constructed.

`hatecheck_task.py` contains the exact frozen two-field prompts and task mapping. At process startup `formal.py --dataset hatecheck` installs this one native schema into the reused parsing/scoring helpers; it does not rewrite their source or change SBIC defaults. Reuse the same five essential arms, model checkpoints, decoding, prefix ordering, label-free 64-group repeat selection, and 576 fresh repeated prompts per model. There are 37,280 stage-one requests and at most 11,184 second-stage requests, plus 576 repeats: **49,040/model; 98,080 across both models**. Both datasets together require at most **211,968 logical requests**. HateCheck's two-field joint correctness is not numerically pooled with SBIC's three-field endpoint.

Apply the same within-task two-model primary and six-comparison correctness-repeat families. The remedy comparison against direct, guarded direct and full-information has six contrasts across the two models (Bonferroni 99.1667% intervals), with the same .02 lower-bound threshold; no native training-constant claim is made. Neither task's favorable cells can substitute for the other's failures. The paper does not claim omnibus familywise control over both distinct tasks. The HateCheck task is a functional corroboration; its native labels/template suite do not make it a natural-post prevalence estimate or an independent SBIC-codebook replication.

Primary uncertainty still resamples exact-text groups. Additionally, resample whole native `templ_id` blocks, preserving all member groups and recomputing the group-weighted mean (not an unweighted template average), for neutral-versus-control mean joint differences and all-valid matched repeat crossing differences. Report these as descriptive 95% dependence sensitivities, alongside all 29 functionality-specific tables. Do not claim a precise generalizable functional effect if the native-template sensitivity is inconclusive. Keep incoherent predicted `(H,D)=(0,1)` vectors as errors; report their frequency and do not silently repair them.

CPU preparation command:

```bash
python paper1/formal.py prepare --dataset hatecheck --source paper1/data/hatecheck --output paper1/prepared/hatecheck_formal_v1
```

Then use the exact SBIC run/stage2/evaluate command sequence above with prepared directory `hatecheck_formal_v1`, model runs `qwen_hatecheck_formal_v1` and `granite_hatecheck_formal_v1`, and second-stage directories `hatecheck_formal_qwen_stage2_v1` and `hatecheck_formal_granite_stage2_v1`. Stage2/evaluate read the frozen dataset tag automatically. Main and repeat input checks use the corresponding HateCheck request paths. Complete the same independent protocol/code review and CPU checks before inference; do not retune prompts on early functional outputs. The source intake and this protocol are not completed model evidence.
