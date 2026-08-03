from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


THINK_BLOCK = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)
UNCLOSED_THINK = re.compile(r"<(think|thinking)>.*\Z", re.DOTALL | re.IGNORECASE)


PATTERNS = [
    ("answer_is",  re.compile(r"answer\s+is\s*[:\-]?\s*\(?\*{0,2}([A-D])\*{0,2}\)?", re.IGNORECASE)),
    ("boxed",      re.compile(r"\\boxed\s*\{\s*\(?([A-D])\)?\s*\}", re.IGNORECASE)),
    ("bold_final", re.compile(r"\*\*\s*\(?([A-D])\)?\s*\*\*")),
    ("trailing",   re.compile(r"(?:^|\n)\s*\(?([A-D])\)?[.):]?\s*\Z")),
]


def strip_thinking(text: str) -> str:
    text = THINK_BLOCK.sub("", text)
    return UNCLOSED_THINK.sub("", text)


def extract(text: str) -> tuple[str | None, str | None]:
    if not isinstance(text, str) or not text.strip():
        return None, None
    body = strip_thinking(text)
    for name, pat in PATTERNS:
        matches = pat.findall(body)
        if matches:
            return matches[-1].upper(), name
    return None, None


def load_predictions(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("id", "output"):
        if col not in df.columns:
            sys.exit(f"[error] 결과 파일에 '{col}' 열/키가 없습니다. 현재 열: {list(df.columns)}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--out", default=Path("out"), type=Path)
    ap.add_argument("--answer-key", type=Path, help="주면 정확도까지 계산")
    ap.add_argument("--max-chars", type=int, default=None,
                    help="finish_reason 이 없는 결과 파일에서만 사용. 출력 길이가 이 값 이상이면 잘린 것으로 간주")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_predictions(args.pred)
    has_fr = "finish_reason" in df.columns
    if not has_fr and args.max_chars is None:
        print("[warn] finish_reason 열이 없고 --max-chars 도 없어, 잘림/미검출을 구분하지 않고 "
              "전부 no_match 로 표시합니다. vLLM 결과에 finish_reason 을 포함해 달라고 요청하는 편이 낫습니다.")

    preds, matched, status = [], [], []
    for _, r in df.iterrows():
        text = str(r["output"])
        p, m = extract(text)
        preds.append(p)
        matched.append(m)
        if p is not None:
            status.append("ok")
        elif has_fr:

            status.append("truncated" if str(r["finish_reason"]).lower() == "length" else "no_match")
        elif args.max_chars is not None and len(text) >= args.max_chars:
            status.append("truncated")
        else:
            status.append("no_match")

    out = pd.DataFrame({"id": df["id"], "pred": preds, "status": status, "matched_by": matched})
    out.to_csv(args.out / "extracted.csv", index=False, encoding="utf-8")

    retry = out.loc[out.status != "ok", "id"]
    (args.out / "retry_ids.txt").write_text("\n".join(map(str, retry)), encoding="utf-8")

    n = len(out)
    print(f"총 {n}건")
    for s, c in out.status.value_counts().items():
        print(f"  {s:10s} {c:6d}  ({c / n:.2%})")
    print(f"\n[ok] {args.out / 'extracted.csv'}")
    print(f"[ok] {args.out / 'retry_ids.txt'}  ({len(retry)}건 재실행 대상)")

    if args.answer_key:
        key = pd.read_csv(args.answer_key, dtype=str, keep_default_na=False)
        m = out.merge(key, on="id", how="left")
        ok = m[m.status == "ok"]
        print(f"\n정확도 (파싱 성공분 {len(ok)}건 기준): {(ok.pred == ok.answer.str.upper()).mean():.4f}")
        print(f"정확도 (전체 {n}건, 파싱 실패=오답):   {(m.pred.fillna('') == m.answer.str.upper()).mean():.4f}")


if __name__ == "__main__":
    main()
