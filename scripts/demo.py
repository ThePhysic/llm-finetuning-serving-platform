"""Small reviewer-facing terminal client for the actual triage inference API."""
from __future__ import annotations

import argparse
import json
import os
import sys

import httpx

from llm_platform.prompts import ZERO_SHOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--report", default=(
        "Order 18422 remains unfulfilled. Warehouse service reports insufficient stock, "
        "but the inventory database shows 12 available units. "
        "The latest inventory synchronization failed with a timeout."))
    args = parser.parse_args()
    key = os.environ.get("LLM_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    with httpx.Client(base_url=args.url.rstrip("/"), headers=headers, timeout=120) as client:
        response = client.get("/v1/models")
        response.raise_for_status()
        loaded = [item["id"] for item in response.json()["data"] if item.get("loaded")]
        if args.model is None and len(loaded) != 1:
            raise SystemExit("A single loaded model is required; select --model explicitly")
        model = args.model or loaded[0]
        print(f"Served artifact: {model}")
        print("Advisory model output; review all evidence and proposed actions.\n")
        payload = {"model": model, "messages": [{"role": "system", "content": ZERO_SHOT},
            {"role": "user", "content": args.report}], "max_tokens": 320,
            "temperature": 0, "seed": 42, "stream": args.stream}
        if args.stream:
            done = False
            finish = None
            with client.stream("POST", "/v1/chat/completions", json=payload) as completion:
                completion.raise_for_status()
                for line in completion.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        print("\nGeneration failed:", chunk["error"].get("code"), file=sys.stderr)
                        raise SystemExit(1)
                    if chunk.get("model") != model:
                        raise SystemExit("Generation failed: served model identity mismatch")
                    for choice in chunk.get("choices", []):
                        if choice.get("finish_reason") is not None:
                            finish = choice["finish_reason"]
                        print(choice.get("delta", {}).get("content", ""), end="", flush=True)
                print()
            if not done or finish not in {"stop", "length"}:
                raise SystemExit("Generation failed: incomplete stream")
        else:
            completion = client.post("/v1/chat/completions", json=payload)
            completion.raise_for_status()
            result = completion.json()
            if result.get("model") != model:
                raise SystemExit("Generation failed: served model identity mismatch")
            if (not result.get("choices")
                    or result["choices"][0].get("finish_reason") not in {"stop", "length"}):
                raise SystemExit("Generation failed: incomplete completion")
            print(result["choices"][0]["message"]["content"])
            print("\nMeasured token usage:", result["usage"])


if __name__ == "__main__":
    main()
