from __future__ import annotations

import re
import unicodedata


_LATIN_RUN = re.compile(r"[A-Za-z0-9\u00C0-\u024F]+")


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text or "")
    out, i = [], 0
    for m in _LATIN_RUN.finditer(text):
        out += [c for c in text[i:m.start()] if not c.isspace() and c.isalnum()]
        out.append(m.group().lower())
        i = m.end()
    out += [c for c in text[i:] if not c.isspace() and c.isalnum()]
    return out


def english_ratio(mixed: str, anchor_text: str, english_text: str) -> float:
    anc, eng = set(tokenize(anchor_text)), set(tokenize(english_text))
    anc_only, en_only = anc - eng, eng - anc
    if not anc_only or not en_only:
        return -1.0

    toks = set(tokenize(mixed))
    cov_en = len(toks & en_only) / len(en_only)
    cov_anc = len(toks & anc_only) / len(anc_only)
    if cov_en + cov_anc == 0:
        return -1.0
    return cov_en / (cov_en + cov_anc)


def anchor_retention(mixed: str, anchor_text: str, english_text: str) -> float:
    anc, eng = set(tokenize(anchor_text)), set(tokenize(english_text))
    anc_only = anc - eng
    if not anc_only:
        return -1.0
    return len(set(tokenize(mixed)) & anc_only) / len(anc_only)


def check_progression(ratios: dict, min_step: float = 0.04, min_span: float = 0.25,
                      retention: dict | None = None,
                      min_retention: float = 0.15) -> tuple[bool, str]:
    need = [0, 25, 50, 75, 100]
    missing = [p for p in need if ratios.get(str(p)) is None]
    if missing:
        return False, f"단계 누락: {missing}"

    vals = [(p, ratios[str(p)]) for p in need]
    if any(v < 0 for _, v in vals):
        return False, "비율 측정 불가한 단계가 있음"

    for (a, x), (b, y) in zip(vals, vals[1:]):
        if y < x + min_step:
            return False, f"{a}%({x:.2f}) → {b}%({y:.2f}): 증가폭이 {min_step} 미만"

    span = ratios["75"] - ratios["25"]
    if span < min_span:
        return False, f"25%~75% 폭이 {span:.2f} 로 {min_span} 미만 — 세 단계가 뭉쳐 있음"

    if retention is not None:
        for pct, keep in retention.items():
            if keep < min_retention:
                return False, (f"{pct}% 단계의 anchor 고유토큰 잔존률 {keep:.2f} < {min_retention} "
                               "— 프레임이 무너지고 영어 번역이 된 것으로 보임")

    return True, ""


def check_stage(mixed: str, pct: int, anchor_text: str, english_text: str,
                tolerance: dict) -> tuple[bool, float, str]:
    r = english_ratio(mixed, anchor_text, english_text)
    if r < 0:
        return False, r, "비율 측정 불가"
    lo, hi = tolerance.get(pct, [0.0, 1.0])
    if not (lo <= r <= hi):
        return False, r, f"{pct}% 단계 측정 비율 {r:.2f} — 허용 범위 [{lo:.2f}, {hi:.2f}] 밖"
    return True, r, ""


def check_monotonic(ratios: list) -> str:
    usable = [(p, r) for p, r in sorted(ratios) if r >= 0]
    bad = [f"{a}%({x:.2f}) → {b}%({y:.2f})"
           for (a, x), (b, y) in zip(usable, usable[1:]) if y < x]
    return "단조 증가 위반: " + ", ".join(bad) if bad else ""