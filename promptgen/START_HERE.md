# 시작하기

## 0. 파일 확인

```
promptgen/
  anchors.yaml          ← 언어 목록·설정. 여기만 고치면 anchor가 늘어남
  itemsel.py            ← 데모 풀 / 평가 세트 분리 (오염 방지)
  mixing.py             ← 영어 혼입 비율 측정
  make_demos.py         ← 데모 생성 (extract → switch → report)
  build_prompts.py      ← prompt.txt + input.csv 생성
  extract_answers.py    ← 사수님께 전달할 답 추출 모듈
  find_demo_rule.py     ← (선택) 기존 데모 5문항의 선별 규칙 역추적
  templates/
    csicl.txt           ← 완료
    translation.txt     ← 완료
    zero_shot.txt       ← ★ 아직 비어 있음. zeroshot.txt 지시문을 넣을 것
  shots/                ← 비어 있어도 됨. extract 가 채운다
```

## 1. 준비

```bash
pip install google-genai pandas pyyaml
echo 'export GEMINI_API_KEY="AQ...."' >> ~/.bashrc && source ~/.bashrc
```

## 2. zero_shot.txt 채우기

`templates/zero_shot.txt` 를 열어 첫 줄의 안내문을 지우고 기존 zeroshot 지시문을
붙여넣는다. `{language}` 와 `{query}` 두 자리표시자는 반드시 남겨둘 것.

## 3. 실행

```bash
# 병렬 코퍼스에서 0%/100% 확보 (LLM 불필요, 즉시)
python make_demos.py extract --benchmark <Global-MMLU 파일>

# 25/50/75% 생성 + 혼입 비율 자동 검증
python make_demos.py switch

# anchor × 단계 실측 비율표 (논문 부록용)
python make_demos.py report

# prompt.txt 29개 + input.csv 1개 생성
python build_prompts.py --benchmark <Global-MMLU 파일> --out out
```

## 4. 사수님께 전달

`out/input.csv`, `out/*/prompt.txt`, `extract_answers.py`, `HANDOFF.md`

`out/answer_key.csv` 는 전달하지 않는다. 채점은 결과를 받은 뒤 이쪽에서 한다.

## 확인해야 할 것

- `anchors.yaml` 의 `name` 값이 벤치마크 `language` 열 값과 일치하는지
- `morphology` / `word_order` 는 손으로 붙인 잠정값 — 논문에는 WALS 20A/21A/22A 에서 재도출
- 데모 5문항이 데모 풀(마지막 20개) 안에 있는지 — 벗어나면 빌드가 중단되며 id를 알려준다
