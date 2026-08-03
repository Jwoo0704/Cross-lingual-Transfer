from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

import itemsel
from mixing import (anchor_retention, check_monotonic, check_progression,
                    check_stage, english_ratio)

HERE = Path(__file__).parent
SHOT_DIR = HERE / "shots"
QUERY_FORMAT = "{question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}"

SWITCH_SYSTEM = """You produce intermediate code-switched versions of a multiple-choice question.

You are given the same question in two languages: the ANCHOR version and the ENGLISH version.
Produce THREE versions, switched 25%, 50% and 75% into English.

Rules:
- The ANCHOR language provides the morphosyntactic frame. Keep its word order, its case
  markers, particles, agreement and other function morphemes intact in every version.
- Switch only content morphemes (nouns, verbs, adjectives, adverbs) into English.
  At 25% switch a few; at 50% switch about half; at 75% switch most.
- Each version must switch strictly more than the previous one. Do not repeat a version.
- Keep the four options labelled A) B) C) D) in the same order.
- Do not translate fully, explain, comment, or include the answer.

Output format — reproduce these markers EXACTLY, nothing before or after:

<<<25>>>
...the 25% version...
<<<50>>>
...the 50% version...
<<<75>>>
...the 75% version...
<<<END>>>
"""

STAGE_RE = re.compile(r"<<<\s*(25|50|75)\s*>>>\s*\n(.*?)(?=<<<)", re.DOTALL)


MARKUP_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__|`(.+?)`", re.DOTALL)


def strip_markup(t: str) -> str:
    return MARKUP_RE.sub(lambda m: next(g for g in m.groups() if g is not None), t).strip()


def parse_stages(text: str) -> dict:
    if not text.rstrip().endswith("<<<END>>>"):
        text = text.rstrip() + "\n<<<END>>>"
    return {pct: strip_markup(body) for pct, body in STAGE_RE.findall(text)}


def load_cfg() -> dict:
    return yaml.safe_load((HERE / "anchors.yaml").read_text(encoding="utf-8"))


def render_query(row: pd.Series) -> str:
    cols = ["question", "option_a", "option_b", "option_c", "option_d"]
    return QUERY_FORMAT.format(**{c: str(row[c]) for c in cols})


def cmd_extract(args, cfg) -> None:
    pool = itemsel.build_pool(cfg, args.cache_dir)
    demo_pool, _ = itemsel.split_pool(pool, cfg)
    demo_ids = itemsel.resolve_demos(cfg, demo_pool)

    ref = cfg["item_selection"].get("reference_lang", "en")
    en_ds = itemsel.load_split(ref, args.cache_dir)
    en_by_id = {x["sample_id"]: x for x in en_ds if x["sample_id"] in set(demo_ids)}

    SHOT_DIR.mkdir(exist_ok=True)
    anchored = [c for c in cfg["conditions"] if c != "zero_shot"]
    stages = cfg["stages"]

    for a in cfg["anchors"]:
        code, name = a["code"], a["name"]
        if all(c in a.get("exclude_from", []) for c in anchored):
            print(f"[skip] {code}: anchor 조건에서 제외됨")
            continue

        ds = itemsel.load_split(code, args.cache_dir)
        by_id = {x["sample_id"]: x for x in ds if x["sample_id"] in set(demo_ids)}

        items = []
        for sid in demo_ids:
            anc, eng = by_id.get(sid), en_by_id.get(sid)
            if anc is None or eng is None:
                print(f"[skip] {code}/{sid}: 병렬 문항 결손")
                continue
            at, et = itemsel.format_query(anc), itemsel.format_query(eng)
            items.append({
                "sample_id": sid,
                "answer": str(anc.get("answer", "")).upper().strip(),
                "anchor_text": at,
                "english_text": et,
                "stages": {str(p): (at if p == 0 else et if p == 100 else "") for p in stages},
                "stage_ratio": {},
            })

        out = SHOT_DIR / f"{code}.json"
        out.write_text(json.dumps({"anchor": a, "items": items},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] {out}  ({len(items)} items, 0%/100% 확보)")


_LAST_CALL = [0.0]


def _throttle() -> None:
    rpm = float(os.environ.get("RPM", "5"))
    gap = 60.0 / max(rpm, 0.1)
    wait = gap - (time.monotonic() - _LAST_CALL[0])
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL[0] = time.monotonic()


def call_llm(system: str, user: str) -> str:
    from google import genai
    from google.genai import types

    model = os.environ.get("SWITCH_MODEL", "gemini-3.5-flash")
    level = os.environ.get("THINKING_LEVEL", "high").lower()

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _call(with_thinking: bool):
        kwargs = {"system_instruction": system,
                  "max_output_tokens": int(os.environ.get("MAX_OUTPUT_TOKENS", "16000"))}
        if with_thinking:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level)
        return client.models.generate_content(
            model=model, contents=user,
            config=types.GenerateContentConfig(**kwargs),
        )

    use_thinking = True
    for attempt in range(6):
        _throttle()
        try:
            r = _call(use_thinking)
            return (r.text or "").strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                delay = min(60, 5 * 2 ** attempt)
                print(f"      [i] 요청 한도 초과 → {delay}초 대기 후 재시도")
                time.sleep(delay)
                continue
            if use_thinking and "thinking" in msg.lower():
                print(f"      [i] thinking_level 미지원 → 기본 설정으로 호출")
                use_thinking = False
                continue
            raise
    raise RuntimeError("요청 한도 초과가 계속됩니다. RPM 을 낮추거나 결제를 활성화하십시오.")


def cmd_switch(args, cfg) -> None:
    tol = {int(k): v for k, v in cfg["stage_tolerance"].items()}
    prog = dict(cfg.get("progression", {"min_step": 0.04, "min_span": 0.25}))
    for k in ("min_step", "min_span"):
        v = os.environ.get(k.upper())
        if v:
            prog[k] = float(v)
            print(f"[i] {k} = {prog[k]} (환경변수로 덮어씀)")
    codes = args.anchors or [a["code"] for a in cfg["anchors"]]

    if not args.dry_run and "GEMINI_API_KEY" not in os.environ:
        sys.exit("[error] GEMINI_API_KEY 가 없습니다. --dry-run 으로 빈 칸만 만들 수 있습니다.")

    for code in codes:
        path = SHOT_DIR / f"{code}.json"
        if not path.exists():
            print(f"[skip] {path} 없음 — extract 를 먼저 실행하십시오.")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))

        def save() -> None:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        for item in data["items"]:
            anc, eng = item["anchor_text"], item["english_text"]
            if all(item["stages"].get(str(p), "").strip() for p in (25, 50, 75)) and not args.force:
                continue
            if args.dry_run:
                continue

            user = f"ANCHOR ({data['anchor']['name']}):\n{anc}\n\nENGLISH:\n{eng}"
            ok = False
            try:
              for attempt in range(1, args.retries + 1):
                cand = parse_stages(call_llm(SWITCH_SYSTEM, user))
                missing = [p for p in ("25", "50", "75") if p not in cand]
                if missing:
                    print(f"  [retry {attempt}/{args.retries}] {code}/{item['sample_id']}: "
                          f"단계 {missing} 파싱 실패")
                    continue

                dirty = [p for p, t in cand.items()
                         if any(x in t for x in ("**", "```", "<<<"))]
                if dirty:
                    print(f"  [retry {attempt}/{args.retries}] {code}/{item['sample_id']}: "
                          f"단계 {dirty} 에 마크업 잔재")
                    continue

                ratios = {"0": english_ratio(anc, anc, eng),
                          "100": english_ratio(eng, anc, eng)}
                keeps = {pct: anchor_retention(t, anc, eng) for pct, t in cand.items()}
                for pct, text in cand.items():
                    passed, r, why = check_stage(text, int(pct), anc, eng, tol)
                    ratios[pct] = r
                    if not passed:
                        print(f"      [!] {code}/{item['sample_id']} {why}")

                ok, why = check_progression({k: round(v, 3) for k, v in ratios.items()},
                                            retention=keeps, **prog)
                if ok:
                    item["stages"].update(cand)
                    item["stage_ratio"] = {k: round(v, 3) for k, v in ratios.items()}
                    item["retention"] = {k: round(v, 3) for k, v in keeps.items()}
                    shown = " → ".join(f"{p}%:{ratios[str(p)]:.2f}" for p in (0, 25, 50, 75, 100))
                    print(f"  [ok] {code}/{item['sample_id']}  {shown}  (시도 {attempt})")
                    break
                print(f"  [retry {attempt}/{args.retries}] {code}/{item['sample_id']}: {why}")
            except (KeyboardInterrupt, Exception) as e:

                save()
                print(f"\n[중단] {code}/{item['sample_id']} 처리 중 {type(e).__name__}: {e}")
                print(f"[ok] {path} — 여기까지 저장됨. 다시 실행하면 이어서 진행합니다.")
                raise

            save()
            if not ok:
                from mixing import tokenize
                A, E = set(tokenize(anc)), set(tokenize(eng))
                print(f"  [FAIL] {code}/{item['sample_id']} — 구분 가능한 토큰: "
                      f"anchor 고유 {len(A - E)}개 / 영어 고유 {len(E - A)}개, "
                      f"공유 {len(A & E)}개")
                print(f"         구분 토큰이 적으면 지표 해상도가 낮아 폭이 좁게 나옵니다. "
                      f"MIN_SPAN 을 낮춰 재시도한 뒤 생성물을 눈으로 확인하십시오.")

        done = sum(1 for i in data["items"]
                   if all(i["stages"].get(str(p), "").strip() for p in (25, 50, 75)))
        print(f"[ok] {path}  ({done}/{len(data['items'])} 항목 완료)")


def cmd_batch(args, cfg) -> None:
    outdir = HERE / "batch"
    outdir.mkdir(exist_ok=True)
    codes = args.anchors or [a["code"] for a in cfg["anchors"]]

    made = 0
    for a in cfg["anchors"]:
        if a["code"] not in codes:
            continue
        path = SHOT_DIR / f"{a['code']}.json"
        if not path.exists():
            continue
        items = json.loads(path.read_text(encoding="utf-8"))["items"]
        todo = [i for i in items
                if any(not i["stages"].get(p, "").strip() for p in ("25", "50", "75"))]
        if not todo and not args.force:
            print(f"[skip] {a['code']}: 이미 전부 채워짐")
            continue
        todo = todo or items

        body = [BATCH_HEADER.format(n=len(todo), lang=a["name"], first_id=todo[0]["sample_id"])]
        for i in todo:
            body.append(f"\n=== {i['sample_id']} ===\nANCHOR ({a['name']}):\n{i['anchor_text']}\n"
                        f"\nENGLISH:\n{i['english_text']}")
        p = outdir / f"{a['code']}_request.txt"
        p.write_text("\n".join(body), encoding="utf-8")
        print(f"[ok] {p}  ({len(todo)}문항 × 3단계 = {len(todo)*3}개 요청)")
        made += 1

    if made:
        print(f"\n{made}개 파일을 채팅창에 하나씩 붙여넣고, 답변 전체를 복사해\n"
              f"  {outdir}/<code>_response.txt 로 저장한 뒤\n"
              f"  python make_demos.py ingest 를 실행하십시오.")


def cmd_ingest(args, cfg) -> None:
    outdir = HERE / "batch"
    tol = {int(k): v for k, v in cfg["stage_tolerance"].items()}
    codes = args.anchors or [a["code"] for a in cfg["anchors"]]

    for a in cfg["anchors"]:
        code = a["code"]
        if code not in codes:
            continue
        resp = outdir / f"{code}_response.txt"
        shot = SHOT_DIR / f"{code}.json"
        if not resp.exists() or not shot.exists():
            continue

        text = resp.read_text(encoding="utf-8")
        parsed = {(sid.strip(), pct): body.strip() for sid, pct, body in ITEM_RE.findall(text)}
        if not parsed:
            print(f"[!] {code}: <<<ITEM ...>>> 마커를 하나도 못 찾았습니다. "
                  "답변을 마커째 그대로 복사했는지 확인하십시오.")
            continue

        data = json.loads(shot.read_text(encoding="utf-8"))
        ok = bad = miss = 0
        for item in data["items"]:
            anc, eng = item["anchor_text"], item["english_text"]
            for pct in (25, 50, 75):
                body = parsed.get((item["sample_id"], str(pct)))
                if body is None:
                    miss += 1
                    continue
                passed, ratio, why = check_stage(body, pct, anc, eng, tol)
                if passed:
                    item["stages"][str(pct)] = body
                    item["stage_ratio"][str(pct)] = round(ratio, 3)
                    ok += 1
                else:
                    print(f"  [reject] {code}/{item['sample_id']} {pct}%  {why}")
                    bad += 1
            for pct in (0, 100):
                item["stage_ratio"][str(pct)] = round(
                    english_ratio(item["stages"][str(pct)], anc, eng), 3)
            note = check_monotonic([(int(p), r) for p, r in item["stage_ratio"].items()])
            if note:
                print(f"  [!] {code}/{item['sample_id']}: {note}")

        shot.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] {code}: 채택 {ok} / 반려 {bad} / 응답없음 {miss}")
        if bad or miss:
            print(f"      해당 항목만 다시 요청하려면 batch 를 다시 실행하십시오 "
                  f"(채워진 것은 건너뜁니다).")


def cmd_report(args, cfg) -> None:
    rows = []
    for a in cfg["anchors"]:
        path = SHOT_DIR / f"{a['code']}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data["items"]:
            for pct, r in item.get("stage_ratio", {}).items():
                rows.append({"anchor": a["code"], "morphology": a["morphology"],
                             "sample_id": item["sample_id"], "stage": int(pct), "ratio": r})
    if not rows:
        sys.exit("측정된 비율이 없습니다. switch 를 먼저 실행하십시오.")

    df = pd.DataFrame(rows)
    print("\n실측 영어 혼입 비율 (anchor × 단계, 데모 문항 평균)\n")
    print(df.pivot_table(index="anchor", columns="stage", values="ratio", aggfunc="mean").round(3).to_string())
    print("\n단계 내 anchor 간 표준편차 — 클수록 생성기가 anchor를 고르게 다루지 못했다는 뜻\n")
    print(df.pivot_table(index="stage", values="ratio", aggfunc="std").round(3).to_string())

    missing = [(a["code"], i["sample_id"], p)
               for a in cfg["anchors"] if (SHOT_DIR / f"{a['code']}.json").exists()
               for i in json.loads((SHOT_DIR / f"{a['code']}.json").read_text(encoding="utf-8"))["items"]
               for p in ("25", "50", "75") if not i["stages"][p]]
    if missing:
        print(f"\n미완성 단계 {len(missing)}개:")
        for m in missing[:10]:
            print("   ", m)


def cmd_fidelity(args, cfg) -> None:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        sys.exit("[error] pip install sentence-transformers 가 필요합니다.")

    model = SentenceTransformer(args.model)
    rows = []
    for a in cfg["anchors"]:
        path = SHOT_DIR / f"{a['code']}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data["items"]:
            emb = model.encode([item["anchor_text"], item["english_text"]], normalize_embeddings=True)
            rows.append({"anchor": a["code"], "resource": a.get("resource", "?"),
                         "sample_id": item["sample_id"], "sim": float(emb[0] @ emb[1])})
    if not rows:
        sys.exit("shots 파일이 없습니다. extract 를 먼저 실행하십시오.")

    df = pd.DataFrame(rows)
    agg = df.groupby(["anchor", "resource"])["sim"].agg(["mean", "min"]).round(3).sort_values("mean")
    print("\n데모 번역 충실도 (anchor 원문 ↔ 영어 원문 의미 유사도)\n")
    print(agg.to_string())
    low = agg[agg["mean"] < args.threshold]
    if len(low):
        print(f"\n[!] 임계값 {args.threshold} 미만 {len(low)}개 — 데모 확인 필요:")
        print("   ", list(low.index.get_level_values("anchor")))
    print("\n등급별 평균 (자원수준과 상관하면 그 자체가 교락 증거):")
    print(df.groupby("resource")["sim"].mean().round(3).to_string())

    out = SHOT_DIR.parent / "demo_fidelity.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n[ok] {out} — 부록·공변량용")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="병렬 코퍼스에서 0%%/100%% 확보")
    e.add_argument("--cache-dir", help="HuggingFace 캐시 위치")

    s = sub.add_parser("switch", help="25/50/75%% 생성 + 비율 검증")
    s.add_argument("--anchors", nargs="*")
    s.add_argument("--retries", type=int, default=3)
    s.add_argument("--force", action="store_true", help="이미 채워진 단계도 다시 생성")
    s.add_argument("--dry-run", action="store_true")

    b = sub.add_parser("batch", help="채팅 UI용 요청 파일 생성 (API 키 불필요)")
    b.add_argument("--anchors", nargs="*")
    b.add_argument("--force", action="store_true")

    g = sub.add_parser("ingest", help="채팅 답변 붙여넣은 파일을 파싱해 반영")
    g.add_argument("--anchors", nargs="*")

    sub.add_parser("report", help="anchor × 단계 혼입 비율표")

    f = sub.add_parser("fidelity", help="데모 5문항의 번역 충실도 측정")
    f.add_argument("--model", default="sentence-transformers/LaBSE")
    f.add_argument("--threshold", type=float, default=0.75)

    args = ap.parse_args()
    cfg = load_cfg()
    {"extract": cmd_extract, "switch": cmd_switch, "batch": cmd_batch,
     "ingest": cmd_ingest, "report": cmd_report, "fidelity": cmd_fidelity}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
