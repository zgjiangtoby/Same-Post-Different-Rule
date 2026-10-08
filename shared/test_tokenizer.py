"""Offline CPU regression: shared loader and token IDs match the official tokenizer JSON."""
import argparse
import json
import os
from pathlib import Path

from generate import load_local_tokenizer, read_requests, sha


def check(model, inputs):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from tokenizers import Tokenizer

    tokenizer = load_local_tokenizer(model)
    native = Tokenizer.from_file(str(model / "tokenizer.json"))
    assert tokenizer.padding_side == "left"
    fixtures = ["Return only {\"ok\": 1}.", "Olá, ação e café!", "हा एक साधा मजकूर आहे.",
                "同一段文字，三个规则。", "First line.\n  Second\tline 🙂"]
    count = 0
    for value in fixtures:
        assert tokenizer.encode(value, add_special_tokens=False) == native.encode(value, add_special_tokens=False).ids
        count += 1
    seen = set()
    chats = [[{"role": "user", "content": value}] for value in fixtures]
    chats.extend(row["messages"] for path in inputs for row in read_requests(path))
    for messages in chats:
        key = sha(messages)
        if key in seen:
            continue
        seen.add(key)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 thinking=False, enable_thinking=False)
        expected = native.encode(rendered, add_special_tokens=False).ids
        assert tokenizer.encode(rendered, add_special_tokens=False) == expected, "Token IDs differ from official JSON"
        assert tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              thinking=False, enable_thinking=False) == expected
    print(json.dumps({"status": "PASS", "tokenizer_class": type(tokenizer).__name__,
                      "is_fast": tokenizer.is_fast, "literal_fixtures": count,
                      "distinct_chat_fixtures": len(seen), "model_inference": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", default=[])
    args = parser.parse_args()
    check(args.model, args.input)
