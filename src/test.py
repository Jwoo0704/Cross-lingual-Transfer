import os
import json
import time
import re
import random
import torch
import multiprocessing as mp
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-12B-it"
SAMPLES_PER_LANG = 300
RANDOM_SEED = 777
CANONICAL_LANG = "en"

FULL_LANGS = {
    'ko': 'Korean',
    'en': 'English',

    'am': 'Amharic',
    'vi': 'Vietnamese',
    'fa': 'Persian',
    'uk': 'Ukrainian',
    'yo': 'Yoruba',
    'ru': 'Russian',
    'de': 'German',

    'el': 'Greek',
    'fil': 'Filipino',
    'id': 'Indonesian',
    'es': 'Spanish',
    'bn': 'Bengali',
    'pt': 'Portuguese',

    'ja': 'Japanese',
    'hi': 'Hindi',
    'zh': 'Chinese',
    'si': 'Sinhala',
    'te': 'Telugu',
    'tr': 'Turkish',
    'so': 'Somali',

    'pl': 'Polish',
    'fr': 'French',
    'ar': 'Arabic',
    'ig': 'Igbo',
    'it': 'Italian',
    'ny': 'Chichewa',
}


def prepare_dataset(samples_per_lang=SAMPLES_PER_LANG, random_seed=RANDOM_SEED):
    en_ds = load_dataset("CohereLabs/Global-MMLU", CANONICAL_LANG)['test']
    pool = [
        x for x in en_ds
        if x.get("is_annotated") is True
        and x.get("cultural_sensitivity_label") == "CA"
    ]
    pool_ids = sorted(x["sample_id"] for x in pool)
    print(f"[*] English canonical CA&annotated pool: {len(pool_ids)}")

    rng = random.Random(random_seed)
    rng.shuffle(pool_ids)
    chosen_ids = pool_ids[:min(samples_per_lang, len(pool_ids))]
    print(f"[*] Selected common items: {len(chosen_ids)} (same per language)")

    chosen_set = set(chosen_ids)

    test_queries, ground_truth = [], {}
    for lang_code, lang_name in FULL_LANGS.items():
        ds = load_dataset("CohereLabs/Global-MMLU", lang_code)['test']
        by_id = {x["sample_id"]: x for x in ds if x["sample_id"] in chosen_set}

        missing = chosen_set - set(by_id.keys())
        if missing:
            print(f"[!] {lang_name}({lang_code}): missing items {len(missing)} -> excluded")

        for sid in chosen_ids:
            if sid not in by_id:
                continue
            item = by_id[sid]
            question_id = f"{lang_code}_{sid}"
            opts = (item.get("option_a", ""), item.get("option_b", ""),
                    item.get("option_c", ""), item.get("option_d", ""))
            formatted = (f"{item.get('question','')} "
                         f"A) {opts[0]} B) {opts[1]} C) {opts[2]} D) {opts[3]}")

            al = str(item.get("answer")).upper().strip()
            if al not in ("A", "B", "C", "D"):
                print(f"[!] [{question_id}] invalid label ({item.get('answer')!r}) excluded")
                continue

            test_queries.append({
                "id": question_id,
                "lang": lang_name,
                "sample_id": sid,
                "subject_category": item.get("subject_category", ""),
                "subject": item.get("subject", ""),
                "query": formatted,
            })
            ground_truth[question_id] = al

    print(f"[*] Total generated queries: {len(test_queries)}")
    return test_queries, ground_truth


def strip_thought_channel(text):
    text = re.sub(r'<\|channel>.*?<channel\|>', '', text, flags=re.DOTALL)
    return text.strip()


def extract_answer(model_output):
    m = list(re.finditer(r'answer\s*is\s*\**([A-D])\b', model_output, re.IGNORECASE))
    if m:
        return m[-1].group(1).upper()
    m2 = list(re.finditer(r'정답\s*(?:은|는|:)?\s*\**([A-D])\b', model_output))
    if m2:
        return m2[-1].group(1).upper()
    return None


def run_experiment(test_queries, ground_truth, output_filename, summary_filename,
                   prompt_template, device="cuda:0", max_new_tokens=3072):
    print(f"[*] [{device}] Preparing {len(test_queries)} prompts...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()

    MAX_CONTEXT_LENGTH = getattr(model.config, "max_position_embeddings", 8192)
    SAFETY_MARGIN = 8
    MAX_INPUT_TOKENS = MAX_CONTEXT_LENGTH - max_new_tokens - SAFETY_MARGIN
    print(f"[*] [{device}] Max context: {MAX_CONTEXT_LENGTH} / Input allowed: {MAX_INPUT_TOKENS}")

    correct_counts = {lang: 0 for lang in FULL_LANGS.values()}
    total_counts = {lang: 0 for lang in FULL_LANGS.values()}
    failed_counts = {lang: 0 for lang in FULL_LANGS.values()}
    skipped_counts = {lang: 0 for lang in FULL_LANGS.values()}
    overall_correct = overall_total = overall_failed = overall_skipped = 0

    with open(output_filename, 'w', encoding='utf-8') as f:
        for item in test_queries:
            target_lang = item["lang"]
            actual_ans = ground_truth.get(item["id"])

            try:
                messages = [
                    {"role": "user", "content": prompt_template + "\n\n" + item['query']},
                ]

                chat_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                encoded = tokenizer(chat_text, return_tensors="pt")
                input_len = encoded["input_ids"].shape[-1]

                if input_len > MAX_INPUT_TOKENS:
                    result_data = {
                        "id": item["id"], "target_lang": target_lang,
                        "sample_id": item["sample_id"],
                        "subject_category": item["subject_category"],
                        "model_output": None, "extracted_answer": None,
                        "ground_truth": actual_ans, "is_correct": False,
                        "status": "skipped_too_long", "input_len": input_len,
                    }
                    skipped_counts[target_lang] += 1
                    overall_skipped += 1
                    print(f"[!] [{item['id']}] input {input_len} > allowed {MAX_INPUT_TOKENS} -> skipped")
                    f.write(json.dumps(result_data, ensure_ascii=False) + '\n'); f.flush()
                    continue

                inputs = {k: v.to(device) for k, v in encoded.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                gen = outputs[0][inputs['input_ids'].shape[-1]:]
                was_truncated = (len(gen) >= max_new_tokens) and (gen[-1].item() != tokenizer.eos_token_id)
                raw_output = tokenizer.decode(gen, skip_special_tokens=False)
                decoded = tokenizer.decode(gen, skip_special_tokens=True)
                model_output = strip_thought_channel(decoded)

                extracted_answer = extract_answer(model_output)
                if extracted_answer is None:
                    status = "truncated_or_malformed"
                    is_correct = False
                    failed_counts[target_lang] += 1
                    overall_failed += 1
                else:
                    status = "success"
                    is_correct = (extracted_answer == actual_ans)
                    total_counts[target_lang] += 1
                    overall_total += 1
                    if is_correct:
                        correct_counts[target_lang] += 1
                        overall_correct += 1

                result_data = {
                    "id": item["id"], "target_lang": target_lang,
                    "sample_id": item["sample_id"],
                    "subject_category": item["subject_category"],
                    "model_output": model_output,
                    "extracted_answer": extracted_answer,
                    "ground_truth": actual_ans, "is_correct": is_correct,
                    "status": status,
                    "was_truncated": was_truncated,
                }

                print(f"[{item['id']}] GT:{actual_ans} / Pred:{extracted_answer} -> "
                      f"{'O' if is_correct else 'X'}")

            except Exception as e:
                result_data = {
                    "id": item["id"], "target_lang": target_lang,
                    "sample_id": item.get("sample_id"),
                    "subject_category": item.get("subject_category"),
                    "model_output": None, "extracted_answer": None,
                    "ground_truth": actual_ans, "is_correct": False,
                    "status": "failed", "error": f"{type(e).__name__}: {e}",
                }
                failed_counts[target_lang] += 1
                overall_failed += 1
                print(f"[-] [{item['id']}] failed ({type(e).__name__}): {e}")

            f.write(json.dumps(result_data, ensure_ascii=False) + '\n'); f.flush()

    summary_lines = ["=== Final Experiment Summary ==="]
    for lang in sorted(FULL_LANGS.values()):
        cnt = total_counts.get(lang, 0)
        fail = failed_counts.get(lang, 0)
        skip = skipped_counts.get(lang, 0)
        if cnt > 0 or fail > 0 or skip > 0:
            acc = (correct_counts[lang] / cnt) * 100 if cnt > 0 else 0.0
            summary_lines.append(
                f"{lang:15s} | Accuracy: {acc:5.1f}% ({correct_counts[lang]}/{cnt})"
                f"  | Failed: {fail} | Skipped: {skip}"
            )
    if overall_total > 0:
        overall_acc = (overall_correct / overall_total) * 100
        summary_lines.append("-" * 60)
        summary_lines.append(
            f"{'OVERALL':15s} | Accuracy: {overall_acc:5.1f}% "
            f"({overall_correct}/{overall_total})  | Failed: {overall_failed} "
            f"| Skipped: {overall_skipped}"
        )
    summary_text = "\n".join(summary_lines)
    print("\n\n" + summary_text)
    with open(summary_filename, 'w', encoding='utf-8') as sf:
        sf.write(summary_text + "\n")


def main():
    PROMPT_FILENAME = "prompts_files/translation.txt"
    try:
        with open(PROMPT_FILENAME, 'r', encoding='utf-8') as pf:
            prompt_template = pf.read()
    except Exception as e:
        print(f"[-] Failed to load prompt file '{PROMPT_FILENAME}': {e}")
        return

    prompt_key = os.path.basename(PROMPT_FILENAME).split('.')[0].lower()
    max_new = 1024 if prompt_key in ("zeroshot","zs") else (1536 if prompt_key=="translation" else 3072)
    print(f"[*] max_new_tokens = {max_new} (prompt='{prompt_key}')")

    test_queries, ground_truth = prepare_dataset()

    lang_names = list(FULL_LANGS.values())
    half = len(lang_names) // 2
    langs_gpu0 = set(lang_names[:half])
    langs_gpu1 = set(lang_names[half:])
    queries_gpu0 = [q for q in test_queries if q['lang'] in langs_gpu0]
    queries_gpu1 = [q for q in test_queries if q['lang'] in langs_gpu1]
    print(f"[*] GPU0: {len(queries_gpu0)} / GPU1: {len(queries_gpu1)}")

    t = int(time.time())
    base = os.path.basename(PROMPT_FILENAME).split('.')[0]
    out0, out1 = f"{base}_results_gpu0_{t}.jsonl", f"{base}_results_gpu1_{t}.jsonl"
    sum0, sum1 = f"{base}_summary_gpu0_{t}.txt", f"{base}_summary_gpu1_{t}.txt"

    p0 = mp.Process(target=run_experiment,
                    args=(queries_gpu0, ground_truth, out0, sum0, prompt_template, "cuda:0", max_new))
    p1 = mp.Process(target=run_experiment,
                    args=(queries_gpu1, ground_truth, out1, sum1, prompt_template, "cuda:1", max_new))
    p0.start(); p1.start(); p0.join(); p1.join()

    merged = f"{base}_results_merged_{t}.jsonl"
    with open(merged, 'w', encoding='utf-8') as mf:
        for fname in (out0, out1):
            if os.path.exists(fname):
                with open(fname, 'r', encoding='utf-8') as sf:
                    mf.write(sf.read())
    print(f"\n[*] Merge complete → {merged}")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
