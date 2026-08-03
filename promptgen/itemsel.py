"""
문항 선별 — 기존 실험 코드(src/test.py)의 prepare_dataset 과 동일한 규칙.

    pool     = 영어 split 에서 is_annotated==True 이고 CA 인 문항
    pool_ids = sorted(sample_id)          # 정렬로 결정론 확보
    Random(RANDOM_SEED).shuffle(pool_ids) # 시드 고정 셔플
    eval     = pool_ids[:300]             # 앞에서 평가 문항
    demo_pool= pool_ids[-20:]             # 뒤에서 데모 추출용 (평가와 절대 안 겹침)

앞 300 / 뒤 20 이므로 pool 이 320보다 크면 교집합은 항상 공집합이다.
seed 와 개수는 anchors.yaml 의 item_selection 에서 바꾼다.
"""

from __future__ import annotations

import random
import sys

TRUEY = {True, 1, "True", "true", "TRUE", "1"}


def load_split(lang_code: str, cache_dir: str | None = None):
    """Global-MMLU 의 한 언어 split 을 그대로 반환한다."""
    from datasets import load_dataset
    return load_dataset("CohereLabs/Global-MMLU", lang_code, cache_dir=cache_dir)["test"]


def build_pool(cfg: dict, cache_dir: str | None = None) -> list[str]:
    """기준 언어(영어)에서 필터를 걸고 정렬+셔플한 sample_id 목록."""
    sel = cfg["item_selection"]
    ref = sel.get("reference_lang", "en")

    ds = load_split(ref, cache_dir)
    pool = [x for x in ds
            if x.get("is_annotated") in TRUEY
            and str(x.get("cultural_sensitivity_label", "")).strip() == "CA"]
    ids = sorted(x["sample_id"] for x in pool)
    print(f"[sel] {ref} CA & annotated pool: {len(ids)}개")

    random.Random(sel["seed"]).shuffle(ids)
    return ids


def split_pool(ids: list[str], cfg: dict) -> tuple[list[str], list[str]]:
    """(데모 풀, 평가 세트). 앞에서 eval_n, 뒤에서 demo_pool_size."""
    sel = cfg["item_selection"]
    n, k = sel["eval_n"], sel["demo_pool_size"]

    if len(ids) < n + k:
        sys.exit(f"[error] pool 이 {len(ids)}개뿐이라 평가 {n} + 데모 {k} 를 못 뗍니다.")

    eval_ids = ids[:n]
    demo_pool = ids[-k:]

    overlap = set(eval_ids) & set(demo_pool)
    if overlap:
        sys.exit(f"[error] 평가와 데모 풀이 겹칩니다: {sorted(overlap)[:5]}")

    print(f"[sel] 평가 {len(eval_ids)}문항 (앞) / 데모 풀 {len(demo_pool)}문항 (뒤)")
    return demo_pool, eval_ids


def resolve_demos(cfg: dict, demo_pool: list[str]) -> list[str]:
    """anchors.yaml 의 demos 를 확정한다. 데모 풀 밖이면 중단."""
    demos = cfg.get("demos") or []
    if not demos:
        demos = demo_pool[:5]
        print(f"[sel] demos 미지정 → 데모 풀 앞에서 자동 선택: {demos}")
        return demos

    outside = [d for d in demos if d not in set(demo_pool)]
    if outside:
        sys.exit(
            f"[error] demos 가 데모 풀(뒤 {len(demo_pool)}개) 밖에 있습니다:\n"
            f"    {outside}\n"
            "    → 평가 세트에 섞였는지 확인하십시오. anchors.yaml 의 demos 를 비우면\n"
            "      데모 풀에서 자동 선택합니다.")
    return demos


def format_query(item: dict) -> str:
    """기존 코드와 동일한 문항 렌더링: '질문 A) .. B) .. C) .. D) ..' 한 줄."""
    return (f"{item.get('question','')} "
            f"A) {item.get('option_a','')} B) {item.get('option_b','')} "
            f"C) {item.get('option_c','')} D) {item.get('option_d','')}")


def make_id(lang_code: str, sample_id: str) -> str:
    """기존 코드와 동일한 id: '{lang_code}_{sample_id}'."""
    return f"{lang_code}_{sample_id}"
