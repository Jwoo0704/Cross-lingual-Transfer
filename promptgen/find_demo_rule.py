#!/usr/bin/env python3
"""
"마지막 20문항" 이 어떤 정렬 기준이었는지 역추적한다.

기록된 데모 5문항이 어느 (풀 정의 × 정렬 기준) 조합에서 마지막 20개 안에 들어가는지
전부 돌려보고, 각 문항이 실제로 끝에서 몇 번째인지 위치까지 찍는다.
어느 조합에서도 안 들어가면 "마지막 20"이 아닌 다른 규칙으로 뽑힌 것이다.

사용
  pip install datasets pandas
  python find_demo_rule.py
  python find_demo_rule.py --local path/to/global_mmlu_en.csv   # 이미 받아둔 파일이 있으면
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

TARGETS = [
    "clinical_knowledge/test/65",
    "high_school_mathematics/test/48",
    "professional_medicine/test/169",
    "medical_genetics/test/15",
    "college_physics/test/20",
]
TRUEY = {True, 1, "True", "true", "TRUE", "1"}


def load(local: str | None) -> pd.DataFrame:
    if local:
        return pd.read_csv(local, dtype=str, keep_default_na=False)
    from datasets import load_dataset
    ds = load_dataset("CohereLabs/Global-MMLU", "en", split="test")
    return ds.to_pandas()


def parts(sid: str) -> tuple[str, str, int]:
    m = re.match(r"(.+)/([^/]+)/(\d+)$", str(sid))
    return (m.group(1), m.group(2), int(m.group(3))) if m else (str(sid), "", -1)


POOLS = {
    "CA + is_annotated": lambda d: d[d.is_annotated.isin(TRUEY)
                                     & (d.cultural_sensitivity_label.astype(str).str.strip() == "CA")],
    "is_annotated only": lambda d: d[d.is_annotated.isin(TRUEY)],
    "필터 없음 (전체)":   lambda d: d,
}

ORDERS = {
    "원본 행 순서":            lambda s: list(s),
    "sample_id 문자열 정렬":   lambda s: sorted(s),
    "과목 + 번호(숫자) 정렬":  lambda s: sorted(s, key=parts),
    "번호(숫자)만 정렬":       lambda s: sorted(s, key=lambda x: parts(x)[2]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local")
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    df = load(args.local)
    print(f"영어 test split: {len(df):,}행\n")

    missing = [t for t in TARGETS if t not in set(df.sample_id.astype(str))]
    if missing:
        print(f"[!] 데이터셋에 아예 없는 sample_id: {missing}\n")

    best = []
    for pname, pfn in POOLS.items():
        pool_df = pfn(df)
        ids = pool_df.sample_id.astype(str).tolist()
        print(f"── 풀: {pname}  ({len(ids):,}문항)")
        inside = [t for t in TARGETS if t in set(ids)]
        if len(inside) < len(TARGETS):
            print(f"     이 풀에 없는 데모: {[t for t in TARGETS if t not in set(ids)]}")

        for oname, ofn in ORDERS.items():
            ordered = ofn(ids)
            n = len(ordered)
            pos = {}
            for t in TARGETS:
                pos[t] = n - ordered.index(t) if t in ordered else None
            hit = sum(1 for v in pos.values() if v is not None and v <= args.k)
            mark = "  ★★★ 전부 포함" if hit == len(TARGETS) else (f"  ({hit}/{len(TARGETS)})" if hit else "")
            print(f"     {oname:22s} 끝에서의 위치: "
                  + ", ".join(f"{t.split('/')[0][:18]}={v if v else '없음'}" for t, v in pos.items())
                  + mark)
            if hit == len(TARGETS):
                best.append((pname, oname, ordered[-args.k:]))
        print()

    if best:
        print("=" * 70)
        for pname, oname, last in best:
            print(f"\n조합 확정: 풀 = {pname} / 정렬 = {oname}")
            print(f"마지막 {args.k}개:")
            for x in last:
                print("   ", x, "  ← 데모" if x in TARGETS else "")
    else:
        print("=" * 70)
        print("어느 조합에서도 5개가 모두 마지막 20 안에 들지 않습니다.")
        print("'마지막 20' 이 아닌 다른 규칙으로 뽑혔을 가능성이 높습니다.")
        print("위 표의 '끝에서의 위치' 값들이 비슷한 구간에 몰려 있으면 그 구간이 실제 풀입니다.")
        print("\n실험을 새로 시작하시는 참이면, anchors.yaml 의 demos 를 비워")
        print("데모 풀에서 자동 선택하게 하는 편이 규칙과 데모가 항상 일치해서 안전합니다.")


if __name__ == "__main__":
    main()
