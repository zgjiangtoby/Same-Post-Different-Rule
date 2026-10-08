#!/usr/bin/env python3
"""Offline, greedy batched generation; gold-free request JSONL in, raw JSONL out."""
import argparse
import atexit
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_requests(path):
    rows, ids = [], set()
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) - {"id", "messages", "metadata"}:
            raise ValueError(f"Invalid request keys on line {line_no}")
        key, messages = row.get("id"), row.get("messages")
        if not isinstance(key, str) or not key or key in ids:
            raise ValueError(f"Missing/duplicate request ID on line {line_no}")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Missing messages for {key}")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in {"system", "user", "assistant"}
                    or not isinstance(message["content"], str)):
                raise ValueError(f"Invalid chat message for {key}")
        if "metadata" in row and not isinstance(row["metadata"], dict):
            raise ValueError(f"Invalid metadata for {key}")
        ids.add(key)
        rows.append(row)
    if not rows:
        raise ValueError("No requests")
    return rows


def load_predictions(paths, request_path):
    """Bind completed model outputs to exact requests and a single decoding setup."""
    requests = {r["id"]: r for r in read_requests(request_path)}
    input_sha = hashlib.sha256(request_path.read_bytes()).hexdigest()
    predictions, configs, manifests = {}, [], []
    for path in paths:
        manifest = json.loads(path.with_suffix(path.suffix + ".manifest.json").read_text())
        config = manifest["config"]
        if (manifest["status"] != "COMPLETED" or config["input_sha256"] != input_sha
                or manifest["run_signature"] != sha(config)):
            raise ValueError("Outputs require a completed, consistent manifest for these exact requests")
        configs.append({k: v for k, v in config.items() if k != "shard_index"})
        manifests.append(manifest)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["id"]
            if key in predictions or key not in requests or row["request_sha256"] != sha(requests[key]):
                raise ValueError("Duplicate, unknown, or stale inference result")
            if (row["run_signature"] != manifest["run_signature"]
                    or row["prompt_sha256"] != sha(requests[key]["messages"])
                    or row.get("metadata", {}) != requests[key].get("metadata", {})):
                raise ValueError("Inference result does not match its request and run manifest")
            predictions[key] = row
    if not configs or any(config != configs[0] for config in configs[1:]):
        raise ValueError("Combine only shards with the same model and decoding configuration")
    return predictions, configs[0], manifests


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_local_tokenizer(model):
    from transformers import AutoTokenizer
    # Transformers 4.57.2's local fast-loader has upstream bug #42395; keep official files unchanged.
    return AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False,
                                         padding_side="left", use_fast=False)


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory(prefix="extra_generate_test_") as directory:
        source = Path(directory) / "requests.jsonl"
        row = {"id": "unit", "messages": [{"role": "user", "content": "literal test"}]}
        source.write_text(json.dumps(row) + "\n")
        assert read_requests(source) == [row]
        assert sha(row) == sha(json.loads(json.dumps(row)))
        source.write_text((json.dumps(row) + "\n") * 2)
        try:
            read_requests(source)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate request IDs accepted")
        source.write_text(json.dumps({**row, "gold": 1}) + "\n")
        try:
            read_requests(source)
        except ValueError:
            pass
        else:
            raise AssertionError("Unexpected request fields accepted")
        atomic_json(Path(directory) / "manifest.json", {"passed": True})
        assert json.loads((Path(directory) / "manifest.json").read_text())["passed"]
        source.write_text(json.dumps(row) + "\n")
        output = Path(directory) / "output.jsonl"
        config = {"input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "shard_index": 0}
        result = {"id": row["id"], "request_sha256": sha(row), "prompt_sha256": sha(row["messages"]),
                  "run_signature": sha(config), "response": "fixture", "status": "ok"}
        output.write_text(json.dumps(result) + "\n")
        atomic_json(output.with_suffix(".jsonl.manifest.json"),
                    {"status": "COMPLETED", "config": config, "run_signature": sha(config)})
        assert load_predictions([output], source)[0][row["id"]]["response"] == "fixture"
        output.write_text(json.dumps({**result, "request_sha256": "stale"}) + "\n")
        try:
            load_predictions([output], source)
        except ValueError:
            pass
        else:
            raise AssertionError("Stale request result accepted")
    print(json.dumps({"status": "PASS", "scope": "request validation, output provenance and manifest write; no model inference"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-revision")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-input", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.input:
        parser.error("--input required")
    rows = read_requests(args.input)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid shard specification")
    rows = rows[args.shard_index::args.num_shards]
    if args.check_input:
        print(json.dumps({"requests": len(rows), "unique_prompts": len({sha(r['messages']) for r in rows})}))
        return
    if (not args.output or not args.model or not args.model.is_dir()
            or not args.model_revision or min(args.batch_size, args.max_input_tokens, args.max_new_tokens) < 1):
        parser.error("Local --model, --model-revision, --output and positive budgets required")
    if args.output.resolve() == args.input.resolve():
        parser.error("Output may not overwrite input")
    if args.output.exists() and not args.resume:
        parser.error("Output exists; use a new path or explicit --resume")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoModelForCausalLM, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("A visible CUDA device is required; use approved device access")
    torch.set_num_threads(4)
    set_seed(args.seed)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers")}
    config = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model_path": str(args.model.resolve()), "model_revision": args.model_revision,
              "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
              "versions": versions, "tokenizer_use_fast": False,
              "dtype": "bfloat16", "attention": "sdpa", "do_sample": False,
              "thinking": False, "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
              "max_input_tokens": args.max_input_tokens, "seed": args.seed,
              "shard_index": args.shard_index, "num_shards": args.num_shards}
    signature = sha(config)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    done, cache = {}, {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            prior = json.loads(line)
            if prior["run_signature"] != signature or prior["id"] in done:
                raise ValueError("Resume signature mismatch or duplicate output ID")
            done[prior["id"]] = prior
            cache[prior["prompt_sha256"]] = prior
        expected = {r["id"]: sha(r) for r in rows}
        if any(key not in expected or expected[key] != value["request_sha256"] for key, value in done.items()):
            raise ValueError("Existing outputs are not this request shard")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"run_signature": signature, "config": config, "status": "RUNNING",
                "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "process_id": os.getpid(), "device": torch.cuda.get_device_name(0),
                "total_requests": len(rows), "completed": len(done)}
    atomic_json(manifest_path, manifest)

    def mark_interrupted():
        if manifest["status"] != "COMPLETED":
            manifest.update(status="INCOMPLETE", completed=len(done),
                            end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            atomic_json(manifest_path, manifest)

    # A killed OS process may not run this handler; its recorded PID/log remain authoritative.
    atexit.register(mark_interrupted)
    tokenizer = load_local_tokenizer(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True, dtype=torch.bfloat16,
                                               attn_implementation="sdpa").to("cuda").eval()
    print(json.dumps({"event": "loaded", "signature": signature, "requests": len(rows), "done": len(done)}), flush=True)
    pending = [row for row in rows if row["id"] not in done]
    with args.output.open("a", buffering=1) as stream:
        def emit(row, result):
            record = {**result, "id": row["id"], "request_sha256": sha(row),
                      "prompt_sha256": sha(row["messages"]), "run_signature": signature,
                      "metadata": row.get("metadata", {})}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            done[row["id"]] = record
            cache[record["prompt_sha256"]] = record

        for start in range(0, len(pending), args.batch_size):
            group = pending[start:start + args.batch_size]
            unique, rendered, delayed = [], [], []
            in_batch = set()
            for row in group:
                key = sha(row["messages"])
                if key in cache or key in in_batch:
                    delayed.append(row)
                    continue
                text = tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                                     add_generation_prompt=True, thinking=False,
                                                     enable_thinking=False)
                count = len(tokenizer.encode(text, add_special_tokens=False))
                if count > args.max_input_tokens:
                    emit(row, {"status": "input_too_long", "response": None, "input_tokens": count,
                               "output_tokens": 0, "charged_input_tokens": 0, "charged_output_tokens": 0,
                               "elapsed_batch_seconds": 0, "batch_size": 0, "cached": False})
                    continue
                in_batch.add(key)
                unique.append(row)
                rendered.append(text)
            if unique:
                encoded = tokenizer(rendered, return_tensors="pt", padding=True,
                                    add_special_tokens=False, truncation=False).to("cuda")
                torch.cuda.synchronize()
                began = time.monotonic()
                with torch.inference_mode():
                    output = model.generate(**encoded, do_sample=False, max_new_tokens=args.max_new_tokens,
                                            pad_token_id=tokenizer.pad_token_id)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - began
                eos = model.generation_config.eos_token_id
                eos = set(eos if isinstance(eos, list) else [eos])
                for index, row in enumerate(unique):
                    tokens = output[index, encoded["input_ids"].shape[1]:].tolist()
                    end = next((j + 1 for j, token in enumerate(tokens) if token in eos), len(tokens))
                    tokens = tokens[:end]
                    length = int(encoded["attention_mask"][index].sum().item())
                    emit(row, {"status": "ok", "response": tokenizer.decode(tokens, skip_special_tokens=True),
                               "input_tokens": length, "output_tokens": len(tokens),
                               "charged_input_tokens": length, "charged_output_tokens": len(tokens),
                               "elapsed_batch_seconds": elapsed, "batch_size": len(unique), "cached": False})
            for row in delayed:
                source = cache[sha(row["messages"])]
                emit(row, {**source, "cache_of": source.get("cache_of", source["id"]), "cached": True,
                           "charged_input_tokens": 0, "charged_output_tokens": 0,
                           "elapsed_batch_seconds": 0, "batch_size": 0})
            manifest["completed"] = len(done)
            atomic_json(manifest_path, manifest)
            print(json.dumps({"event": "progress", "completed": len(done), "total": len(rows)}), flush=True)
    manifest.update(status="COMPLETED", end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    atomic_json(manifest_path, manifest)
    atexit.unregister(mark_interrupted)


if __name__ == "__main__":
    main()
