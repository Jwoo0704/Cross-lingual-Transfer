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
BATCH_SIZE = 5

FULL_LANGS = {
    'ko': 'Korean', # 1st Anchor
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

    'ja': 'Japanese', # 3rd Anchor
    'hi': 'Hindi',
    'zh': 'Chinese',
    'si': 'Sinhala',
    'te': 'Telugu',
    'tr': 'Turkish',
    'so': 'Somali',

    'pl': 'Polish',
    'fr': 'French', # 2nd Anchor
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
    print(f"[*] Eng canonical CA&annotated pool: {len(pool_ids)}")

    rng = random.Random(random_seed)
    rng.shuffle(pool_ids)
    chosen_ids = pool_ids[:min(samples_per_lang, len(pool_ids))]
    print(f"[*] common problem number: {len(chosen_ids)}")

    chosen_set = set(chosen_ids)

    test_queries, ground_truth = [], {}
    for lang_code, lang_name in FULL_LANGS.items():
        ds = load_dataset("CohereLabs/Global-MMLU", lang_code)['test']
        by_id = {x["sample_id"]: x for x in ds if x["sample_id"] in chosen_set}

        missing = chosen_set - set(by_id.keys())
        if missing:
            print(f"[!] {lang_name}({lang_code}): {len(missing)} -> None")

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
                print(f"[!] [{question_id}] Abnormal labels ({item.get('answer')!r}) Exclusion")
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

    print(f"[*] queries: {len(test_queries)}")
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


def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def run_experiment(test_queries, ground_truth, output_filename, summary_filename,
                    prompt_template, device="cuda:0", max_new_tokens=3072):
    print(f"[*] [{device}] total {len(test_queries)} prompts preparing")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    MAX_CONTEXT_LENGTH = getattr(model.config, "max_position_embeddings", 8192)
    SAFETY_MARGIN = 8
    MAX_INPUT_TOKENS = MAX_CONTEXT_LENGTH - max_new_tokens - SAFETY_MARGIN
    print(f"[*] [{device}] MAX CONTEXT: {MAX_CONTEXT_LENGTH} / MAX TOKENS: {MAX_INPUT_TOKENS}")

    correct_counts = {lang: 0 for lang in FULL_LANGS.values()}
    total_counts = {lang: 0 for lang in FULL_LANGS.values()}
    failed_counts = {lang: 0 for lang in FULL_LANGS.values()}
    skipped_counts = {lang: 0 for lang in FULL_LANGS.values()}
    overall_correct = overall_total = overall_failed = overall_skipped = 0

    chat_texts_all = []
    for item in test_queries:
        messages = [{"role": "user", "content": prompt_template + "\n\n" + item['query']}]
        chat_texts_all.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ))

    lengths = [len(tokenizer(t)["input_ids"]) for t in chat_texts_all]
    order = sorted(range(len(test_queries)), key=lambda i: lengths[i])
    sorted_queries = [test_queries[i] for i in order]
    sorted_texts = [chat_texts_all[i] for i in order]

    with open(output_filename, 'w', encoding='utf-8') as f:
        for batch_items, batch_texts in zip(batched(sorted_queries, BATCH_SIZE),
                                             batched(sorted_texts, BATCH_SIZE)):
            encoded = tokenizer(batch_texts, return_tensors="pt", padding=True)
            input_lens = encoded["attention_mask"].sum(dim=1)

            keep_idx = [i for i, l in enumerate(input_lens) if l <= MAX_INPUT_TOKENS]
            skip_idx = [i for i in range(len(batch_items)) if i not in keep_idx]

            for i in skip_idx:
                item = batch_items[i]
                actual_ans = ground_truth.get(item["id"])
                target_lang = item["lang"]
                result_data = {
                    "id": item["id"], "target_lang": target_lang,
                    "sample_id": item["sample_id"],
                    "subject_category": item["subject_category"],
                    "model_output": None, "extracted_answer": None,
                    "ground_truth": actual_ans, "is_correct": False,
                    "status": "skipped_too_long", "input_len": int(input_lens[i]),
                }
                skipped_counts[target_lang] += 1
                overall_skipped += 1
                print(f"[!] [{item['id']}] Input {int(input_lens[i])} > approval {MAX_INPUT_TOKENS} -> skip")
                f.write(json.dumps(result_data, ensure_ascii=False) + '\n'); f.flush()

            if not keep_idx:
                continue

            kept_items = [batch_items[i] for i in keep_idx]
            kept_texts = [batch_texts[i] for i in keep_idx]
            encoded = tokenizer(kept_texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in encoded.items()}

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        stop_strings=["The answer is A", "The answer is B", "The answer is C", "The answer is D"],
                        tokenizer=tokenizer,
                    )

                input_len_batch = inputs['input_ids'].shape[-1]

                for i, item in enumerate(kept_items):
                    target_lang = item["lang"]
                    actual_ans = ground_truth.get(item["id"])

                    gen = outputs[i][input_len_batch:]
                    was_truncated = (len(gen) >= max_new_tokens) and (gen[-1].item() != tokenizer.eos_token_id)
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
                    f.write(json.dumps(result_data, ensure_ascii=False) + '\n'); f.flush()

            except Exception as e:
                for item in kept_items:
                    target_lang = item["lang"]
                    actual_ans = ground_truth.get(item["id"])
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
                    f.write(json.dumps(result_data, ensure_ascii=False) + '\n'); f.flush()
                print(f"[-] batch failed ({type(e).__name__}): {e}")

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
    PROMPT_FILENAME = "prompts_files/translation_ja_new.txt"
    try:
        with open(PROMPT_FILENAME, 'r', encoding='utf-8') as pf:
            prompt_template = pf.read()
    except Exception as e:
        print(f"[-] Failed to load prompt file '{PROMPT_FILENAME}': {e}")
        return

    prompt_key = os.path.basename(PROMPT_FILENAME).split('.')[0].lower()
    if prompt_key in ("zeroshot", "zs"):
        max_new = 128
    elif prompt_key.startswith("translation"):
        max_new = 1024
    else:
        max_new = 2048
    print(f"[*] max_new_tokens = {max_new} (prompt='{prompt_key}')")

    test_queries, ground_truth = prepare_dataset()

    N_GPU = 3
    GPU_IDS = [5, 6, 7]

    lang_names = list(FULL_LANGS.values())
    raw_groups = [lang_names[i::N_GPU] for i in range(N_GPU)]
    raw_groups.sort(key=len)   
    groups = [set(g) for g in raw_groups]

    queries_by_gpu = [
        [q for q in test_queries if q['lang'] in groups[i]]
        for i in range(N_GPU)
    ]
    
    for i in range(N_GPU):
        print(f"[*] GPU{GPU_IDS[i]}: {len(queries_by_gpu[i])} queries")

    t = int(time.time())
    base = os.path.basename(PROMPT_FILENAME).split('.')[0]
    outs = [f"{base}_results_gpu{GPU_IDS[i]}_{t}.jsonl" for i in range(N_GPU)]
    sums = [f"{base}_summary_gpu{GPU_IDS[i]}_{t}.txt" for i in range(N_GPU)]

    procs = []
    for i in range(N_GPU):
        p = mp.Process(
            target=run_experiment,
            args=(queries_by_gpu[i], ground_truth, outs[i], sums[i],
                  prompt_template, f"cuda:{GPU_IDS[i]}", max_new)
        )
        procs.append(p)

    for p in procs:
        p.start()
    for p in procs:
        p.join()

    merged = f"{base}_results_merged_{t}.jsonl"
    with open(merged, 'w', encoding='utf-8') as mf:
        for fname in outs:
            if os.path.exists(fname):
                with open(fname, 'r', encoding='utf-8') as sf:
                    mf.write(sf.read())
    print(f"\n[*] Merged -> {merged}")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
