"""
영어 혼입 비율 측정.

외부 사전 없이, 같은 문항의 anchor 원문과 영어 원문을 서로의 기준점으로 삼는다.

    ANC_only = anchor 원문에만 있는 토큰
    EN_only  = 영어 원문에만 있는 토큰
    ratio(m) = |m ∩ EN_only| / (|m ∩ EN_only| + |m ∩ ANC_only|)

양쪽에 다 나오는 토큰(숫자, 고유명사, 동원어)은 자동으로 중립 처리된다.
불어·터키어처럼 anchor도 라틴 문자인 경우에 문자 기반 판별이 실패하는 문제를
이 방식이 우회한다.
"""

from __future__ import annotations

import re
import unicodedata

# 라틴 문자/숫자 연쇄는 한 토큰, 그 밖의 비공백 문자는 한 글자씩.
# 띄어쓰기가 없는 일본어와 띄어쓰기가 있는 한국어를 같은 코드로 다룰 수 있다.
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
    """0.0 = anchor 원문과 같음, 1.0 = 영어 원문과 같음. 판별 불가 시 -1.0.

    각 언어를 '그 언어 고유 토큰 전체 대비 몇 %가 남아 있는가' 로 정규화한 뒤 비교한다.
    라틴 문자는 단어 단위, 그 밖의 문자는 글자 단위로 쪼개지므로 토큰 수를 그대로
    비교하면 분모가 한쪽으로 쏠린다. 자기 집합 크기로 나누면 그 단위 차이가 상쇄된다.
    """
    anc, eng = set(tokenize(anchor_text)), set(tokenize(english_text))
    anc_only, en_only = anc - eng, eng - anc
    if not anc_only or not en_only:
        return -1.0

    toks = set(tokenize(mixed))
    cov_en = len(toks & en_only) / len(en_only)     # 영어가 얼마나 들어왔는가
    cov_anc = len(toks & anc_only) / len(anc_only)  # anchor 가 얼마나 남아 있는가
    if cov_en + cov_anc == 0:
        return -1.0
    return cov_en / (cov_en + cov_anc)


def anchor_retention(mixed: str, anchor_text: str, english_text: str) -> float:
    """anchor 고유 토큰이 얼마나 남아 있는가. MLF 붕괴(전부 영어화) 탐지용.

    주의: 이 값은 '영어 어순으로 번역해버린 경우'와 'MLF 준수'를 잘 구분하지 못한다.
    어순 위반은 이 지표로 못 잡으므로 표본을 눈으로 확인해야 한다.
    """
    anc, eng = set(tokenize(anchor_text)), set(tokenize(english_text))
    anc_only = anc - eng
    if not anc_only:
        return -1.0
    return len(set(tokenize(mixed)) & anc_only) / len(anc_only)


def check_progression(ratios: dict, min_step: float = 0.04, min_span: float = 0.25,
                      retention: dict | None = None,
                      min_retention: float = 0.15) -> tuple[bool, str]:
    """세 중간 단계가 '점진적 전환'으로 볼 수 있는 모양인지 판정한다.

    절대 구간(0.25±, 0.50± ...)으로 재지 않는다. MLF 조건상 조사·어미 같은 기능형태소는
    유지되므로, 내용어만 바꾸면 전체 대비 영어 비율은 명목 퍼센트에 도달할 수 없다.
    이름과 척도가 다르므로 절대값이 아니라 형태를 본다.

      - 0% < 25% < 50% < 75% < 100% 로 단조 증가
      - 인접 단계 간 최소 min_step 이상 벌어짐 (같은 문장 반복 방지)
      - 25%~75% 전체 폭이 min_span 이상 (세 단계가 한곳에 뭉치는 것 방지)
    """
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
    """개별 단계의 느슨한 상식 검사. 실제 판정은 check_progression 이 한다."""
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