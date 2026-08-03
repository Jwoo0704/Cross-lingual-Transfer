from __future__ import annotations

import random
import sys

TRUEY = {True, 1, "True", "true", "TRUE", "1"}


def load_split(lang_code: str, cache_dir: str | None = None):
    from datasets import load_dataset
    return load_dataset("CohereLabs/Global-MMLU", lang_code, cache_dir=cache_dir)["test"]


def build_pool(cfg: dict, cache_dir: str | None = None) -> list[str]:
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
    return (f"{item.get('question','')} "
            f"A) {item.get('option_a','')} B) {item.get('option_b','')} "
            f"C) {item.get('option_c','')} D) {item.get('option_d','')}")


def make_id(lang_code: str, sample_id: str) -> str:
    return f"{lang_code}_{sample_id}"
