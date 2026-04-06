#!/usr/bin/env python3
"""Comprehensive LLM Benchmark Suite for Model Comparison.

Runs a standardized set of tests against an OpenAI-compatible API endpoint
and produces a detailed report covering:
  - Model info & capabilities
  - TTFT (time to first token)
  - Throughput (tokens/sec)
  - Multi-turn reasoning
  - Code generation
  - Math / logic
  - Summarization
  - Instruction following
  - Context handling
"""

import argparse
import json
import time
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

# ============================================================================
# Test prompts — identical across models for fair comparison
# ============================================================================

BENCHMARK_PROMPTS = {
    "reasoning": {
        "label": "Multi-step Reasoning",
        "messages": [
            {"role": "user", "content": (
                "A farmer has 17 sheep. All but 9 run away. How many sheep does "
                "the farmer have left? Explain your reasoning step by step."
            )},
        ],
        "max_tokens": 300,
        "expect_contains": ["9"],
    },
    "code_python": {
        "label": "Python Code Generation",
        "messages": [
            {"role": "user", "content": (
                "Write a Python function `merge_sorted(a, b)` that merges two "
                "sorted lists into one sorted list without using built-in sort. "
                "Include type hints and a docstring."
            )},
        ],
        "max_tokens": 500,
        "expect_contains": ["def merge_sorted"],
    },
    "code_debug": {
        "label": "Code Debugging",
        "messages": [
            {"role": "user", "content": (
                "Find and fix the bug in this code:\n\n"
                "def fibonacci(n):\n"
                "    if n <= 1:\n"
                "        return n\n"
                "    a, b = 0, 1\n"
                "    for i in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n\n"
                "print(fibonacci(0))  # Expected: 0\n"
                "print(fibonacci(1))  # Expected: 1\n"
                "print(fibonacci(6))  # Expected: 8\n"
                "print(fibonacci(10)) # Expected: 55\n\n"
                "What does this function actually return for fibonacci(6) and why? "
                "Fix it if needed."
            )},
        ],
        "max_tokens": 500,
        "expect_contains": ["8"],
    },
    "math": {
        "label": "Mathematical Reasoning",
        "messages": [
            {"role": "user", "content": (
                "Solve: If 3x + 7 = 22, what is the value of 5x - 3? "
                "Show your work."
            )},
        ],
        "max_tokens": 300,
        "expect_contains": ["x = 5", "22"],
    },
    "summarization": {
        "label": "Text Summarization",
        "messages": [
            {"role": "user", "content": (
                "Summarize the following in exactly 2 sentences:\n\n"
                "The transformer architecture, introduced in the 2017 paper "
                "'Attention Is All You Need' by Vaswani et al., revolutionized "
                "natural language processing by replacing recurrent neural networks "
                "with self-attention mechanisms. This allowed for much greater "
                "parallelization during training, leading to significant speedups. "
                "The architecture consists of an encoder-decoder structure, where "
                "both components use multi-head attention layers. Transformers "
                "became the foundation for models like BERT, GPT, and T5, which "
                "achieved state-of-the-art results across numerous NLP benchmarks. "
                "The key innovation was the attention mechanism that allows the "
                "model to weigh the importance of different parts of the input "
                "when generating each part of the output."
            )},
        ],
        "max_tokens": 200,
        "expect_contains": ["transformer", "attention"],
    },
    "instruction_follow": {
        "label": "Instruction Following",
        "messages": [
            {"role": "user", "content": (
                "List exactly 5 programming languages that start with the letter "
                "'P'. Format as a numbered list. Do not include any other text."
            )},
        ],
        "max_tokens": 150,
        "expect_contains": ["Python"],
    },
    "creative": {
        "label": "Creative Writing",
        "messages": [
            {"role": "user", "content": (
                "Write a haiku about GPU computing."
            )},
        ],
        "max_tokens": 100,
    },
    "multi_turn": {
        "label": "Multi-turn Conversation",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": (
                "What is the population of that city? And what is the most "
                "famous landmark there?"
            )},
        ],
        "max_tokens": 300,
        "expect_contains": ["Eiffel"],
    },
    "json_output": {
        "label": "Structured JSON Output",
        "messages": [
            {"role": "user", "content": (
                "Return a JSON object with exactly these fields:\n"
                "- name: \"benchmark_test\"\n"
                "- score: 95.5\n"
                "- tags: [\"fast\", \"accurate\"]\n"
                "- passed: true\n\n"
                "Return ONLY the JSON, no markdown fences, no explanation."
            )},
        ],
        "max_tokens": 150,
        "expect_json": True,
    },
    "long_context": {
        "label": "Long Context Understanding",
        "messages": [
            {"role": "user", "content": (
                "I will give you a list of items. Remember them all.\n\n"
                "apple, banana, cherry, dragonfruit, elderberry, fig, grape, "
                "honeydew, imbe, jackfruit, kiwi, lemon, mango, nectarine, "
                "orange, papaya, quince, raspberry, strawberry, tangerine, "
                "ugli fruit, vanilla bean, watermelon, ximenia, yam berry, zucchini\n\n"
                "Now answer: What is the 15th item in the list? What comes "
                "after 'kiwi'? What is the last item?"
            )},
        ],
        "max_tokens": 200,
        "expect_contains": ["orange", "lemon", "zucchini"],
    },
}


def call_api(base_url: str, api_key: str, model: str, messages: list,
             max_tokens: int = 300, temperature: float = 0.1,
             stream: bool = False) -> dict:
    """Call the OpenAI-compatible chat completions API."""
    url = f"{base_url}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    if stream:
        start = time.perf_counter()
        first_token_time = None
        full_text = ""
        token_count = 0  # counts SSE chunks with content (approximates tokens)

        resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=120)
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        full_text += content
                        token_count += 1
                except json.JSONDecodeError:
                    continue

        end = time.perf_counter()
        ttft = (first_token_time - start) if first_token_time else (end - start)
        total_time = end - start

        return {
            "text": full_text,
            "ttft": ttft,
            "total_time": total_time,
            "completion_tokens": token_count,
            "tps": token_count / total_time if total_time > 0 else 0,
        }
    else:
        start = time.perf_counter()
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        end = time.perf_counter()
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", len(text.split()))
        total_time = end - start

        return {
            "text": text,
            "ttft": None,
            "total_time": total_time,
            "completion_tokens": completion_tokens,
            "tps": completion_tokens / total_time if total_time > 0 else 0,
        }


def run_benchmark(base_url: str, api_key: str, model: str) -> dict:
    """Run full benchmark suite against a model."""

    # 1. Model info
    try:
        resp = requests.get(f"{base_url}/v1/models",
                            headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
        model_info = resp.json()["data"][0] if resp.ok else {}
    except Exception:
        model_info = {}

    max_ctx = model_info.get("max_model_len", "unknown")

    results = {
        "model_name": model,
        "model_root": model_info.get("root", "unknown"),
        "max_context_length": max_ctx,
        "endpoint": base_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": {},
        "summary": {},
    }

    # 2. TTFT benchmark (3 runs, streaming)
    print(f"\n{'='*60}")
    print(f"  Benchmarking: {model}")
    print(f"  Endpoint: {base_url}")
    print(f"  Max context: {max_ctx}")
    print(f"{'='*60}\n")

    ttft_times = []
    print("[TTFT] Running 3 streaming warmup/measurement calls...")
    for i in range(3):
        try:
            r = call_api(base_url, api_key, model,
                         [{"role": "user", "content": "Say hello."}],
                         max_tokens=20, stream=True)
            ttft_times.append(r["ttft"])
            print(f"  Run {i+1}: TTFT={r['ttft']*1000:.0f}ms, TPS={r['tps']:.1f}")
        except Exception as e:
            print(f"  Run {i+1}: FAILED - {e}")

    if ttft_times:
        results["summary"]["ttft_avg_ms"] = round(statistics.mean(ttft_times) * 1000, 1)
        results["summary"]["ttft_min_ms"] = round(min(ttft_times) * 1000, 1)
        results["summary"]["ttft_p50_ms"] = round(statistics.median(ttft_times) * 1000, 1)

    # 3. Run each benchmark test
    all_tps = []
    total_pass = 0
    total_tests = len(BENCHMARK_PROMPTS)

    for test_id, test in BENCHMARK_PROMPTS.items():
        print(f"\n[{test['label']}] Running...")
        try:
            r = call_api(base_url, api_key, model,
                         test["messages"], max_tokens=test["max_tokens"],
                         stream=True)

            text = r["text"]
            passed = True
            checks = []

            # Check expected content
            if "expect_contains" in test:
                for keyword in test["expect_contains"]:
                    found = keyword.lower() in text.lower()
                    checks.append({"keyword": keyword, "found": found})
                    if not found:
                        passed = False

            # Check JSON output
            if test.get("expect_json"):
                try:
                    # Try to extract JSON from response
                    clean = text.strip()
                    if clean.startswith("```"):
                        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
                    json.loads(clean)
                    checks.append({"json_valid": True})
                except Exception:
                    checks.append({"json_valid": False})
                    passed = False

            if passed:
                total_pass += 1

            all_tps.append(r["tps"])

            result_entry = {
                "label": test["label"],
                "passed": passed,
                "checks": checks,
                "ttft_ms": round(r["ttft"] * 1000, 1) if r["ttft"] else None,
                "total_time_s": round(r["total_time"], 2),
                "completion_tokens": r["completion_tokens"],
                "tps": round(r["tps"], 1),
                "response_preview": text[:300],
            }
            results["tests"][test_id] = result_entry

            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {r['tps']:.1f} tok/s | {r['total_time']:.2f}s | "
                  f"{r['completion_tokens']} tokens")
            if not passed:
                print(f"  Checks: {checks}")

        except Exception as e:
            results["tests"][test_id] = {
                "label": test["label"],
                "passed": False,
                "error": str(e),
            }
            print(f"  [ERROR] {e}")

    # 4. Summary
    results["summary"]["tests_passed"] = total_pass
    results["summary"]["tests_total"] = total_tests
    results["summary"]["pass_rate"] = f"{total_pass}/{total_tests} ({100*total_pass/total_tests:.0f}%)"
    if all_tps:
        results["summary"]["avg_tps"] = round(statistics.mean(all_tps), 1)
        results["summary"]["median_tps"] = round(statistics.median(all_tps), 1)
        results["summary"]["max_tps"] = round(max(all_tps), 1)

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {model}")
    print(f"{'='*60}")
    for k, v in results["summary"].items():
        print(f"  {k}: {v}")
    print()

    return results


def save_report(results: dict, output_dir: str = ".") -> str:
    """Save benchmark results to JSON file."""
    model_safe = results["model_name"].replace("/", "_").replace(" ", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{model_safe}_{ts}.json"
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved: {path}")
    return str(path)


def print_comparison(reports: list[dict]):
    """Print side-by-side comparison of two benchmark reports."""
    if len(reports) < 2:
        return

    a, b = reports[0], reports[1]
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<30} {'Model A':>18} {'Model B':>18}")
    print(f"  {'':->30} {'':->18} {'':->18}")
    print(f"  {'Model':<30} {a['model_name']:>18} {b['model_name']:>18}")
    print(f"  {'Max Context':<30} {str(a['summary'].get('max_context_length',a.get('max_context_length','?'))):>18} {str(b['summary'].get('max_context_length',b.get('max_context_length','?'))):>18}")

    for key in ["ttft_avg_ms", "avg_tps", "median_tps", "max_tps", "pass_rate"]:
        va = a["summary"].get(key, "—")
        vb = b["summary"].get(key, "—")
        unit = "ms" if "ttft" in key else ("tok/s" if "tps" in key else "")
        print(f"  {key:<30} {str(va)+unit:>18} {str(vb)+unit:>18}")

    print(f"\n  {'Test':<30} {'A':>8} {'B':>8} {'Winner':>10}")
    print(f"  {'':->30} {'':->8} {'':->8} {'':->10}")

    a_wins = b_wins = 0
    all_tests = set(list(a["tests"].keys()) + list(b["tests"].keys()))
    for tid in sorted(all_tests):
        ta = a["tests"].get(tid, {})
        tb = b["tests"].get(tid, {})
        pa = "PASS" if ta.get("passed") else "FAIL"
        pb = "PASS" if tb.get("passed") else "FAIL"
        tps_a = ta.get("tps", 0)
        tps_b = tb.get("tps", 0)
        if ta.get("passed") and not tb.get("passed"):
            winner = "A"
            a_wins += 1
        elif tb.get("passed") and not ta.get("passed"):
            winner = "B"
            b_wins += 1
        elif tps_a > tps_b:
            winner = "A"
            a_wins += 1
        elif tps_b > tps_a:
            winner = "B"
            b_wins += 1
        else:
            winner = "TIE"
        label = ta.get("label") or tb.get("label", tid)
        print(f"  {label:<30} {pa:>8} {pb:>8} {winner:>10}")

    print(f"\n  Final Score: {a['model_name']} {a_wins} — {b_wins} {b['model_name']}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Benchmark Suite")
    parser.add_argument("--url", required=True, help="Base URL (e.g. http://192.168.29.252:8000)")
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--compare", help="Path to previous report JSON for comparison")
    args = parser.parse_args()

    results = run_benchmark(args.url, args.api_key, args.model)
    report_path = save_report(results, args.output_dir)

    if args.compare:
        with open(args.compare) as f:
            prev = json.load(f)
        print_comparison([prev, results])
