#!/usr/bin/env python3
"""LLM evaluation script for CUE-Bench.

The script supports OpenAI-compatible chat completion APIs and writes one JSONL
prediction record per sample, so interrupted runs can be resumed.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config

try:
    from openai import OpenAI
except ImportError:
    print("请安装 openai: pip install openai")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = ROOT_DIR / "data" / "paper_experimental_split"
PROMPTS_DIR = ROOT_DIR / "prompts"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── 数据加载 ──

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records


def load_prompt_template(task, method, model_key=None):
    task_cfg = config.TASKS[task]
    fname = task_cfg["prompt_files"][method]
    with open(PROMPTS_DIR / fname, "r", encoding="utf-8") as f:
        return f.read()


# ── 断点续跑 ──

def get_output_path(model_key, method, task, shot):
    result_tag = os.environ.get("RESULT_TAG", "").strip()
    tag_suffix = f"_{result_tag}" if result_tag else ""
    return RESULTS_DIR / f"{model_key}_{method}_{task}_{shot}shot{tag_suffix}.jsonl"


def load_completed_ids(out_path):
    """读取已完成的 sample_id 集合，用于断点续跑"""
    completed = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line.strip())
                        completed.add(r["sample_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return completed


def append_result(out_path, result):
    """逐条追加写入结果"""
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ── Few-shot 样本选择 ──

def select_few_shot_examples(dev_data, task, n):
    label_field = config.TASKS[task]["label_field"]
    labels = config.TASKS[task]["labels"]

    by_label = {}
    for r in dev_data:
        lab = r.get(label_field)
        if lab and lab in labels:
            by_label.setdefault(lab, []).append(r)

    for lab in by_label:
        by_label[lab].sort(key=lambda x: 0 if x.get("quality_tier") == "human" else 1)

    examples = []
    if n >= len(labels):
        for lab in labels:
            if lab in by_label and by_label[lab]:
                examples.append(by_label[lab][0])
    else:
        for lab in labels:
            if lab in by_label and by_label[lab]:
                examples.append(by_label[lab][0])
        examples = examples[:n]

    return examples


def format_few_shot_block(examples, task, method):
    label_field = config.TASKS[task]["label_field"]
    output_key = config.TASKS[task]["output_key"].get(method, label_field)

    blocks = []
    for i, ex in enumerate(examples, 1):
        ctx = ex.get("context", "（无上下文）")
        text = ex["text"]
        label = ex[label_field]
        scenario = ex.get("scenario_type") or ex.get("scenario_type_desc", "")

        if method in ("chain", "gold_p") and task == "intent":
            block = f"示例{i}:\n【场景类型】{scenario}\n【上下文】{ctx}\n【目标句】{text}\n【已知情感立场】{ex['affective_stance']}\n答案: {{\"{output_key}\": \"{label}\"}}"
        elif method in ("chain", "gold_pi") and task == "emotion":
            block = f"示例{i}:\n【场景类型】{scenario}\n【上下文】{ctx}\n【目标句】{text}\n【已知情感立场】{ex['affective_stance']}\n【已知语用意图】{ex['pragmatic_intent']}\n答案: {{\"{output_key}\": \"{label}\"}}"
        elif method == "gold_p" and task == "emotion":
            block = f"示例{i}:\n【场景类型】{scenario}\n【上下文】{ctx}\n【目标句】{text}\n【已知情感立场】{ex['affective_stance']}\n答案: {{\"{output_key}\": \"{label}\"}}"
        elif method == "chain" and task == "stance":
            block = f"示例{i}:\n【场景类型】{scenario}\n【上下文】{ctx}\n【目标句】{text}\n答案: {{\"explicit_polarity\": \"{ex['explicit_polarity']}\", \"implicit_polarity\": \"{ex['implicit_polarity']}\", \"affective_stance\": \"{label}\"}}"
        else:
            block = f"示例{i}:\n【场景类型】{scenario}\n【上下文】{ctx}\n【目标句】{text}\n答案: {{\"{output_key}\": \"{label}\"}}"

        blocks.append(block)

    return "\n\n".join(blocks)


# ── API 调用 ──

def get_client(model_key):
    model_cfg = config.MODELS[model_key]
    api_key = os.environ.get(model_cfg["api_key_var"], "")
    base_url = os.environ.get(model_cfg["base_url_var"], config.DEFAULT_BASE_URL)
    if not api_key:
        raise ValueError(f"API key not set for {model_key} ({model_cfg['api_key_var']})")
    timeout = float(os.environ.get("OPENAI_TIMEOUT", model_cfg.get("timeout", 60)))
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def call_api(client, model_name, system_prompt, user_prompt,
             max_retries=config.MAX_RETRIES, max_tokens=None, disable_thinking=None):
    tokens = max_tokens or config.MAX_TOKENS
    retries = max_retries

    actual_user = user_prompt
    extra = {}
    if disable_thinking == "no_think_tag":
        actual_user = user_prompt + "\n/no_think"
    elif disable_thinking == "reasoning_limit":
        extra = {"reasoning": {"max_tokens": 1}}
    elif disable_thinking == "enable_thinking_false":
        extra = {"enable_thinking": False}

    for attempt in range(retries):
        try:
            kwargs = dict(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": actual_user},
                ],
                temperature=config.TEMPERATURE,
                max_tokens=tokens,
            )
            if extra:
                kwargs["extra_body"] = extra
            response = client.chat.completions.create(**kwargs)
            if not response.choices:
                return None
            message = response.choices[0].message
            content = message.content
            if not content:
                content = getattr(message, "reasoning_content", None)
            if not content and hasattr(message, "model_dump_json"):
                content = message.model_dump_json()
            if not content:
                return None
            return content.strip()
        except json.JSONDecodeError:
            # API 返回的 HTTP body 损坏（大响应传输问题），值得重试
            if attempt < retries - 1:
                wait = config.RETRY_DELAY * (attempt + 1)
                print(f"  Retry {attempt+1}/{retries} (transport error) after {wait}s")
                time.sleep(wait)
            else:
                print(f"  Transport error persisted after {retries} attempts, skipping")
                return None
        except Exception as e:
            if extra and "enable_thinking" in str(e):
                print("  Provider rejected enable_thinking=False; retrying without it")
                return call_api(client, model_name, system_prompt, user_prompt,
                                max_retries=max_retries, max_tokens=max_tokens,
                                disable_thinking=None)
            if attempt < retries - 1:
                wait = config.RETRY_DELAY * (attempt + 1)
                print(f"  Retry {attempt+1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"  Failed after {retries} attempts: {e}")
                return None


def parse_json_response(text, output_key):
    if not text or len(text) < 3:
        return None

    # 1. 尝试直接解析整段
    try:
        obj = json.loads(text)
        return obj.get(output_key)
    except json.JSONDecodeError:
        pass

    # 2. 从 markdown code block 中提取
    m = re.search(r'```(?:json)?\s*(\{.+?\})\s*```', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            return obj.get(output_key)
        except json.JSONDecodeError:
            pass

    # 3. 匹配最外层 { ... }（支持多行/嵌套字段）
    m = re.search(r'\{[^{}]*"' + re.escape(output_key) + r'"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj.get(output_key)
        except json.JSONDecodeError:
            pass

    # 4. 匹配任意 JSON object（多行）
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj.get(output_key)
        except json.JSONDecodeError:
            pass

    # 5. 直接匹配 "key": "value"（处理截断的 JSON）
    m = re.search(rf'"{output_key}"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1)

    return None


# ── 主评估逻辑 ──

def evaluate(model_key, task, method, shot, max_samples=None, dev_data=None, test_data=None):
    """
    评估单个 (model, task, method, shot) 组合。
    支持断点续跑：自动跳过已完成的 sample_id。
    """
    out_path = get_output_path(model_key, method, task, shot)
    completed_ids = load_completed_ids(out_path)

    print(f"\n{'='*60}")
    print(f"Model: {model_key} | Task: {task} | Method: {method} | Shot: {shot}")
    if completed_ids:
        print(f"  Resuming: {len(completed_ids)} already done")
    print(f"{'='*60}")

    model_cfg = config.MODELS[model_key]
    task_cfg = config.TASKS[task]
    output_key = task_cfg["output_key"][method]

    client = get_client(model_key)
    template = load_prompt_template(task, method, model_key)

    # Few-shot
    few_shot_block = ""
    if shot == "few" and dev_data:
        n_shot = config.FEW_SHOT_N.get(task, 5)
        examples = select_few_shot_examples(dev_data, task, n_shot)
        few_shot_block = "\n\n## 参考示例\n\n" + format_few_shot_block(examples, task, method) + "\n\n---\n\n"

    # Split template: system (framework desc) and user (per-sample)
    system_lines = []
    user_template_lines = []
    in_user = False
    for line in template.split("\n"):
        if line.startswith("【场景类型】") or line.startswith("【上下文】") or line.startswith("【目标句】") or in_user:
            in_user = True
            user_template_lines.append(line)
        else:
            system_lines.append(line)

    system_prompt = "\n".join(system_lines).strip()
    if few_shot_block:
        system_prompt += few_shot_block
    user_template = "\n".join(user_template_lines).strip()

    data = test_data
    if max_samples:
        data = data[:max_samples]

    correct = 0
    total = 0
    parse_failures = 0
    skipped = 0
    all_results = []

    for sample in data:
        sid = sample["sample_id"]

        # 断点续跑：跳过已完成
        if sid in completed_ids:
            skipped += 1
            continue

        ctx = sample.get("context", "（无上下文）")
        text = sample["text"]
        scenario_desc = sample.get("scenario_type") or sample.get("scenario_type_desc", "")
        user_prompt = (user_template
                       .replace("{scenario_type_desc}", scenario_desc)
                       .replace("{scenario_type}", scenario_desc)
                       .replace("{context}", ctx)
                       .replace("{text}", text))

        if method == "gold_p":
            user_prompt = user_prompt.replace("{affective_stance}", sample.get("affective_stance", ""))
            if "{pragmatic_intent}" in user_prompt:
                user_prompt = user_prompt.replace("{pragmatic_intent}", sample.get("pragmatic_intent", ""))
        elif method == "gold_pi":
            user_prompt = user_prompt.replace("{affective_stance}", sample.get("affective_stance", ""))
            user_prompt = user_prompt.replace("{pragmatic_intent}", sample.get("pragmatic_intent", ""))

        thinking_override = None if method in ("cot", "structure_only", "chain") else model_cfg.get("disable_thinking")
        if model_cfg.get("force_disable_thinking"):
            thinking_override = model_cfg.get("disable_thinking")

        raw = call_api(client, model_cfg["model_name"], system_prompt, user_prompt,
                       max_tokens=model_cfg.get("max_tokens"),
                       max_retries=model_cfg.get("max_retries", config.MAX_RETRIES),
                       disable_thinking=thinking_override)
        pred = parse_json_response(raw, output_key)

        gold = sample.get(task_cfg["label_field"])
        is_correct = pred == gold if pred else False
        if is_correct:
            correct += 1
        if pred is None:
            parse_failures += 1
        total += 1

        result = {
            "sample_id": sid,
            "gold": gold,
            "predicted": pred,
            "correct": is_correct,
            "raw_response": raw,
        }
        all_results.append(result)

        # 逐条写入（断点安全）
        append_result(out_path, result)

        done = total + skipped
        if total % 50 == 0:
            acc = correct / total if total else 0
            print(f"  [{done}/{len(data)}] acc={acc:.3f}, parse_fail={parse_failures}")

    acc = correct / total if total else 0
    print(f"\n  Final: {correct}/{total} = {acc:.3f} accuracy, {parse_failures} parse failures")
    if skipped:
        print(f"  Skipped (already done): {skipped}")
    print(f"  Results at: {out_path}")

    return all_results


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="LLM 评估脚本（支持断点续跑）")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick-test", metavar="MODEL",
                       help="快速测试：50条，direct+cot 全部")
    group.add_argument("--test-cot", metavar="MODEL",
                       help="测试 CoT：50条 × 3任务 × few-shot")
    group.add_argument("--run-model", metavar="MODEL",
                       help="单模型全量：该模型所有 few-shot 任务（可多窗口并行）")
    group.add_argument("--run-priority", metavar="MODEL",
                       help="优先跑：direct×3(zero+few) + cot×3(few only)")
    group.add_argument("--all", action="store_true",
                       help="全量评估（few-shot 优先，zero-shot 最后）")

    parser.add_argument("--model", choices=list(config.MODELS.keys()), help="模型（单任务模式）")
    parser.add_argument("--task", choices=list(config.TASKS.keys()), help="任务")
    parser.add_argument("--method", choices=config.METHODS, help="方法")
    parser.add_argument("--shot", choices=["zero", "few"], default="few", help="shot 设置（默认 few）")
    parser.add_argument("--max-samples", type=int, help="最大样本数")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help="Directory containing dev.jsonl and test.jsonl")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if data_dir.resolve() != DATA_DIR.resolve() and not os.environ.get("RESULT_TAG"):
        os.environ["RESULT_TAG"] = data_dir.name

    dev_data = load_jsonl(data_dir / "dev.jsonl")
    test_data = load_jsonl(data_dir / "test.jsonl")
    print(f"Loaded data from {data_dir}: dev={len(dev_data)}, test={len(test_data)}")
    if os.environ.get("RESULT_TAG"):
        print(f"Result tag: {os.environ['RESULT_TAG']}")

    # 优先任务组合: (method, task, shot)
    PRIORITY_JOBS = [
        # Phase 1: direct zero-shot
        ("direct", "stance", "zero"),
        ("direct", "intent",  "zero"),
        ("direct", "emotion", "zero"),
        # Phase 2: direct few-shot
        ("direct", "stance", "few"),
        ("direct", "intent",  "few"),
        ("direct", "emotion", "few"),
        # Phase 3: cot few-shot only
        ("cot",    "stance", "few"),
        ("cot",    "intent",  "few"),
        ("cot",    "emotion", "few"),
    ]

    if args.quick_test:
        model = args.quick_test
        print(f"\n*** Quick Test: {model}, 50 samples ***\n")
        for method, task, shot in PRIORITY_JOBS:
            try:
                evaluate(model, task, method, shot, 50, dev_data, test_data)
            except Exception as e:
                print(f"  ERROR [{shot}/{method}/{task}]: {e}")

    elif args.test_cot:
        model = args.test_cot
        print(f"\n*** Test CoT: {model}, 50 samples, few-shot ***\n")
        for task in ["stance", "intent", "emotion"]:
            try:
                evaluate(model, task, "cot", "few", 50, dev_data, test_data)
            except Exception as e:
                print(f"  ERROR [cot/{task}]: {e}")

    elif args.run_model:
        # Single-model full few-shot evaluation over public methods.
        model = args.run_model
        print(f"\n*** Run Model: {model}, all tasks, few-shot ***\n")
        for task in config.TASKS:
            for method in config.METHODS:
                if method not in config.TASKS[task]["prompt_files"]:
                    continue
                try:
                    evaluate(model, task, method, "few", args.max_samples, dev_data, test_data)
                except Exception as e:
                    print(f"  ERROR [{task}/{method}]: {e}")

    elif args.run_priority:
        # direct(zero+few) + cot(few only)
        model = args.run_priority
        print(f"\n*** Priority Run: {model} ***")
        print(f"  Phase 1: direct × 3 tasks × zero-shot")
        print(f"  Phase 2: direct × 3 tasks × few-shot")
        print(f"  Phase 3: cot × 3 tasks × few-shot")
        print(f"  Total: {len(PRIORITY_JOBS)} jobs\n")

        for i, (method, task, shot) in enumerate(PRIORITY_JOBS, 1):
            print(f"\n--- [{i}/{len(PRIORITY_JOBS)}] {method} / {task} / {shot}-shot ---")
            try:
                evaluate(model, task, method, shot, args.max_samples, dev_data, test_data)
            except Exception as e:
                print(f"  ERROR [{method}/{task}/{shot}]: {e}")

    elif args.all:
        # 全量：few-shot 优先，zero-shot 最后
        for shot in ["few", "zero"]:
            for model_key in config.MODELS:
                for task in config.TASKS:
                    for method in config.METHODS:
                        if method not in config.TASKS[task]["prompt_files"]:
                            continue
                        try:
                            evaluate(model_key, task, method, shot,
                                     args.max_samples, dev_data, test_data)
                        except Exception as e:
                            print(f"  ERROR: {e}")
    elif args.model and args.task and args.method:
        evaluate(args.model, args.task, args.method, args.shot,
                 args.max_samples, dev_data, test_data)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
