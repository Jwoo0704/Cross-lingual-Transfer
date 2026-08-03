from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

import itemsel

HERE = Path(__file__).parent
TEMPLATE_DIR = HERE / "templates"
SHOT_DIR = HERE / "shots"

PLACEHOLDERS = ("{language}", "{query}")
SHOT_SLOT = "<<SHOTS>>"


def sha8(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:8]


def build_input(cfg: dict, eval_ids: list, cache_dir: str | None) -> pd.DataFrame:
    chosen = set(eval_ids)
    rank_of = {sid: i + 1 for i, sid in enumerate(eval_ids)}
    rows = []
    for a in cfg["anchors"]:
        code, name = a["code"], a["name"]
        ds = itemsel.load_split(code, cache_dir)
        by_id = {x["sample_id"]: x for x in ds if x["sample_id"] in chosen}
        missing = chosen - set(by_id)
        if missing:
            print(f"[!] {name}({code}): 누락 {len(missing)}개 → 제외")
        for sid in eval_ids:
            item = by_id.get(sid)
            if item is None:
                continue
            ans = str(item.get("answer", "")).upper().strip()
            if ans not in ("A", "B", "C", "D"):
                print(f"[!] {code}_{sid}: 비정상 레이블 {item.get('answer')!r} → 제외")
                continue
            rows.append({"rank": rank_of[sid], "id": itemsel.make_id(code, sid),
                         "language": name, "query": itemsel.format_query(item),
                         "answer": ans})
    df = pd.DataFrame(rows)
    if df["id"].duplicated().any():
        sys.exit(f"[error] id 중복: {df.loc[df['id'].duplicated(),'id'].head(3).tolist()}")
    return df


def write_input_csv(df: pd.DataFrame, out: Path) -> None:
    p = out / "input.csv"
    df = df.sort_values(["rank", "language"]).reset_index(drop=True)
    df[["rank", "id", "language", "query"]].to_csv(p, index=False, quoting=csv.QUOTE_ALL,
                                                   encoding="utf-8", lineterminator="\n")
    back = pd.read_csv(p, dtype=str, keep_default_na=False)
    if not back["query"].equals(df["query"].reset_index(drop=True).astype(str)):
        sys.exit("[error] input.csv 왕복 검증 실패")
    print(f"[ok] {p}  ({len(df)} rows, {df['language'].nunique()} languages)")
    df[["id", "answer"]].to_csv(out / "answer_key.csv", index=False, encoding="utf-8")
    print(f"[ok] {out/'answer_key.csv'}  (전달용 아님)")


def render_shots(items: list, condition: str, stages: list) -> str:
    blocks = []
    for i, s in enumerate(items, 1):
        if condition == "translation":
            body = f"{s['anchor_text']}\n\n{s['english_text']}"
        else:
            body = "\n\n".join(f"({p}%)\n{s['stages'][str(p)]}" for p in stages)
        blocks.append(f"### Example {i}\n{body}\n\nThe answer is {s['answer']}")
    return "\n\n".join(blocks)


def build_one(condition: str, anchor, cfg: dict, out: Path) -> dict:
    tpl = TEMPLATE_DIR / f"{condition}.txt"
    if not tpl.exists():
        sys.exit(f"[error] 템플릿 없음: {tpl}")
    template = tpl.read_text(encoding="utf-8")

    n_shots, incomplete = 0, []
    if anchor is None:
        prompt, job = template.replace(SHOT_SLOT, ""), condition
    else:
        job = f"{condition}_{anchor['code']}"
        sp = SHOT_DIR / f"{anchor['code']}.json"
        if not sp.exists():
            sys.exit(f"[error] {sp} 없음 — make_demos.py extract 를 먼저 실행하십시오.")
        items = json.loads(sp.read_text(encoding="utf-8"))["items"]
        n_shots = len(items)
        if condition == "csicl":
            incomplete = [f"{s['sample_id']}@{p}%" for s in items for p in cfg["stages"]
                          if not s["stages"].get(str(p), "").strip()]
        if SHOT_SLOT not in template:
            sys.exit(f"[error] {tpl.name} 에 {SHOT_SLOT} 가 없습니다.")
        prompt = template.replace(SHOT_SLOT, render_shots(items, condition, cfg["stages"]))

    d = out / job
    d.mkdir(parents=True, exist_ok=True)
    (d / "prompt.txt").write_text(prompt, encoding="utf-8")
    return {"job": job, "condition": condition,
            "anchor": anchor["code"] if anchor else None,
            "template_sha8": sha8(template), "prompt_sha8": sha8(prompt),
            "n_shots": n_shots, "chars": len(prompt),
            "incomplete_stages": incomplete, "path": str(d / "prompt.txt")}


def validate(path, df: pd.DataFrame) -> list:
    text = Path(path).read_text(encoding="utf-8")
    problems = []
    for ph in PLACEHOLDERS:
        n = text.count(ph)
        if n == 0:
            problems.append(f"{ph} 없음")
        elif n > 1:
            problems.append(f"{ph} 가 {n}회")

    stray = []
    for ln, line in enumerate(text.splitlines(), 1):
        probe = line
        for ph in PLACEHOLDERS:
            probe = probe.replace(ph, "")
        probe = probe.replace("{{", "").replace("}}", "")
        if "{" in probe or "}" in probe:
            stray.append(ln)
    if stray:
        problems.append(f"자리표시자 외 중괄호 {len(stray)}줄 (예: {stray[:5]}행)")

    lengths = []
    for _, r in df.sample(min(len(df), 200), random_state=0).iterrows():
        rendered = text.replace("{language}", str(r["language"])).replace("{query}", str(r["query"]))
        if any(ph in rendered for ph in PLACEHOLDERS):
            problems.append("치환 후에도 자리표시자가 남음")
            break
        lengths.append(len(rendered))
    if lengths:
        lengths.sort()
        print(f"      렌더 길이 chars  min {lengths[0]} / median {lengths[len(lengths)//2]} / max {lengths[-1]}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path("out"), type=Path)
    ap.add_argument("--anchors", nargs="*")
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--cache-dir", help="HuggingFace 캐시 위치")
    args = ap.parse_args()

    cfg = yaml.safe_load((HERE / "anchors.yaml").read_text(encoding="utf-8"))
    anchors = [a for a in cfg["anchors"] if not args.anchors or a["code"] in args.anchors]
    conditions = args.conditions or cfg["conditions"]
    args.out.mkdir(parents=True, exist_ok=True)

    pool = itemsel.build_pool(cfg, args.cache_dir)
    demo_pool, eval_ids = itemsel.split_pool(pool, cfg)
    demo_ids = itemsel.resolve_demos(cfg, demo_pool)

    df = build_input(cfg, eval_ids, args.cache_dir)
    leak = set(x.split("_", 1)[1] for x in df["id"]) & set(demo_ids)
    if leak:
        sys.exit(f"[error] 데모 문항이 평가 세트에 있습니다: {sorted(leak)}")
    write_input_csv(df, args.out)

    manifest, failed = [], False
    for cond in conditions:
        targets = ([None] if cond == "zero_shot"
                   else [a for a in anchors if cond not in a.get("exclude_from", [])])
        for a in targets:
            meta = build_one(cond, a, cfg, args.out)
            print(f"[ok] {meta['path']}  (shots={meta['n_shots']}, {meta['chars']} chars)")
            problems = validate(meta["path"], df)
            if meta["incomplete_stages"]:
                problems.append(f"미완성 단계 {len(meta['incomplete_stages'])}개: "
                                f"{meta['incomplete_stages'][:3]}")
            meta["problems"] = problems
            for p in problems:
                print(f"      [!] {p}")
                failed = True
            manifest.append(meta)

    (args.out / "manifest.json").write_text(json.dumps(
        {"rows": len(df), "languages": sorted(df["language"].unique()),
         "item_selection": cfg["item_selection"], "demo_ids": demo_ids,
         "demo_pool": demo_pool, "eval_ids": eval_ids, "jobs": manifest},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[ok] {args.out/'manifest.json'}  — prompt.txt {len(manifest)}개")
    if failed:
        print("\n검증 경고가 있습니다. 전달 전에 확인하십시오.")
        sys.exit(1)


if __name__ == "__main__":
    main()