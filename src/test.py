import os
import json
import time
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "google/gemma-4-E2B-it"

FULL_LANGS = {
    'am': 'Amharic', 'ar': 'Arabic', 'bn': 'Bengali', 'cs': 'Czech', 'de': 'German', 
    'el': 'Greek', 'en': 'English', 'es': 'Spanish', 'fa': 'Persian', 'fil': 'Filipino', 
    'fr': 'French', 'ha': 'Hausa', 'he': 'Hebrew', 'hi': 'Hindi', 'id': 'Indonesian', 
    'ig': 'Igbo', 'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean', 'ky': 'Kyrgyz', 
    'lt': 'Lithuanian', 'mg': 'Malagasy', 'ms': 'Malay', 'ne': 'Nepali', 'nl': 'Dutch', 
    'ny': 'Chichewa', 'pl': 'Polish', 'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 
    'si': 'Sinhala', 'sn': 'Shona', 'so': 'Somali', 'sr': 'Serbian', 'sv': 'Swedish', 
    'sw': 'Swahili', 'te': 'Telugu', 'tr': 'Turkish', 'uk': 'Ukrainian', 'vi': 'Vietnamese', 
    'yo': 'Yoruba', 'zh': 'Chinese'
}

def prepare_dataset(samples_per_lang=500, random_seed=777): # SAMPLES_PER_LANG: 자유롭게 수정
    test_queries = []
    ground_truth = {}
    options_map = {0: "A", 1: "B", 2: "C", 3: "D", "0": "A", "1": "B", "2": "C", "3": "D"}

    for lang_code, lang_name in FULL_LANGS.items():
        try:
            global_mmlu = load_dataset("CohereLabs/Global-MMLU", lang_code)
            actual_samples = min(samples_per_lang, len(global_mmlu['test']))
            sampled_data = global_mmlu['test'].shuffle(seed=random_seed).select(range(actual_samples))

            for i, item in enumerate(sampled_data):
                question_id = f"{lang_code}_mmlu_{i+1}"
                opt_a, opt_b = item.get("option_a", ""), item.get("option_b", "")
                opt_c, opt_d = item.get("option_c", ""), item.get("option_d", "")
                
                formatted_query = f"{item.get('question', '')} A) {opt_a} B) {opt_b} C) {opt_c} D) {opt_d}"
                ans = item.get("answer")
                answer_letter = options_map.get(int(ans), "A") if isinstance(ans, int) or str(ans).isdigit() else str(ans).upper().strip()

                test_queries.append({
                    "id": question_id,
                    "lang": lang_name,
                    "query": formatted_query
                })
                ground_truth[question_id] = answer_letter
        except Exception as e:
            print(f"[-] {lang_name}({lang_code}) 데이터셋 로드 실패: {e}")

    return test_queries, ground_truth

def run_experiment(test_queries, ground_truth, output_filename, summary_filename, prompt_template):
    print(f"[*] 총 {len(test_queries)}개의 프롬프트를 준비합니다...")
    
    print("[*] HuggingFace 엔진을 초기화합니다. (GPU 1개 할당 중...)")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        device_map="cuda:0", 
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"  
    )
    model.eval()

    MAX_NEW_TOKENS = 10
    MAX_CONTEXT_LENGTH = getattr(model.config, "max_position_embeddings", 8192)
    SAFETY_MARGIN = 8 
    MAX_INPUT_TOKENS = MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS - SAFETY_MARGIN
    print(f"[*] 모델 최대 컨텍스트: {MAX_CONTEXT_LENGTH} / 입력 허용 토큰: {MAX_INPUT_TOKENS}")

    correct_counts = {lang: 0 for lang in FULL_LANGS.values()}
    total_counts = {lang: 0 for lang in FULL_LANGS.values()}
    failed_counts = {lang: 0 for lang in FULL_LANGS.values()}
    overall_correct = 0
    overall_total = 0
    overall_failed = 0

    print("[*] 모델 추론 시작 (순차 처리)...")
    with open(output_filename, 'w', encoding='utf-8') as f:
        
        for item in test_queries:
            target_lang = item["lang"]
            actual_ans = ground_truth.get(item["id"])

            try:
                messages = [
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": item['query']},
                ]

                chat_text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )

                encoded = processor(text=chat_text, return_tensors="pt")
                input_len = encoded["input_ids"].shape[-1]
                was_truncated = input_len > MAX_INPUT_TOKENS

                if was_truncated:
                    encoded = {k: v[:, -MAX_INPUT_TOKENS:] for k, v in encoded.items()}
                    print(f"[!] [{item['id']}] 입력 {input_len}토큰 -> {MAX_INPUT_TOKENS}토큰으로 절단됨 (앞쪽 절단)")

                inputs = {k: v.to("cuda:0") for k, v in encoded.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )

                generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
                model_output = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                clean_output = model_output.strip().upper()

                extracted_answer = None
                if clean_output in ["A", "B", "C", "D"]:
                    extracted_answer = clean_output
                else:
                    matches = re.findall(r"\b([A-D])\b", clean_output)
                    if matches:
                        extracted_answer = matches[-1]

                is_correct = (extracted_answer == actual_ans)

                result_data = {
                    "id": item["id"],
                    "target_lang": target_lang,
                    "model_output": model_output,
                    "extracted_answer": extracted_answer,
                    "ground_truth": actual_ans,
                    "is_correct": is_correct,
                    "input_truncated": was_truncated,
                    "status": "success"
                }

                total_counts[target_lang] += 1
                overall_total += 1
                if is_correct:
                    correct_counts[target_lang] += 1
                    overall_correct += 1

                print(f"[{item['id']}] 정답: {actual_ans} / 예측: {extracted_answer} -> {'정답' if is_correct else '오답'}")

            except Exception as e:
                result_data = {
                    "id": item["id"],
                    "target_lang": target_lang,
                    "model_output": None,
                    "extracted_answer": None,
                    "ground_truth": actual_ans,
                    "is_correct": False,
                    "status": "failed",
                    "error": f"{type(e).__name__}: {e}"
                }
                failed_counts[target_lang] += 1
                overall_failed += 1
                print(f"[-] [{item['id']}] 처리 실패 ({type(e).__name__}): {e}")

            f.write(json.dumps(result_data, ensure_ascii=False) + '\n')
            f.flush()

            # time.sleep(1) -> 유사시 부활

    summary_lines = ["=== Final Experiment Summary ==="]
    for lang in sorted(FULL_LANGS.values()):
        cnt = total_counts.get(lang, 0)
        fail = failed_counts.get(lang, 0)
        if cnt > 0 or fail > 0:
            acc = (correct_counts[lang] / cnt) * 100 if cnt > 0 else 0.0
            summary_lines.append(
                f"{lang:15s} | Accuracy: {acc:5.1f}% ({correct_counts[lang]}/{cnt})  | Failed: {fail}"
            )

    if overall_total > 0:
        overall_acc = (overall_correct / overall_total) * 100
        summary_lines.append("-" * 60)
        summary_lines.append(
            f"{'OVERALL':15s} | Accuracy: {overall_acc:5.1f}% ({overall_correct}/{overall_total})  | Failed: {overall_failed}"
        )

    summary_text = "\n".join(summary_lines)
    print("\n\n" + summary_text)

    with open(summary_filename, 'w', encoding='utf-8') as sf:
        sf.write(summary_text + "\n")

def main():
    PROMPT_FILENAME = "prompts_files/five_shot.txt"

    try:
        with open(PROMPT_FILENAME, 'r', encoding='utf-8') as pf:
            prompt_template = pf.read()
    except Exception as e:
        print(f"[-] 프롬프트 파일 '{PROMPT_FILENAME}' 로드 실패: {e}")
        return
    
    SAMPLES_PER_LANG = 500 # 여기를 수정해야 함.
    
    test_queries, ground_truth = prepare_dataset(samples_per_lang=SAMPLES_PER_LANG)
    """ 
    Is this okay that 'ground_truth' is used in this situation?
    In this situation, 'ground_truth' doesn't mean that the prompt pair (lang, prompt).
    """
    
    current_time = int(time.time())
    prompt_filename = PROMPT_FILENAME.split('.')[0] 
    output_filename = f"{prompt_filename}_results_test_{current_time}.jsonl"
    summary_filename = f"{prompt_filename}_summary_test_{current_time}.txt"

    run_experiment(test_queries, ground_truth, output_filename, summary_filename, prompt_template)

if __name__ == "__main__":
    main()