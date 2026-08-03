# 프롬프트 생성 파이프라인

anchor를 늘릴 때 손으로 쓰는 부분을 없애는 것이 목적이다.
`anchors.yaml` 에 블록 하나를 추가하면 그 anchor의 모든 프롬프트가 자동으로 나온다.

## 왜 이 구조인가

Global-MMLU는 병렬 코퍼스다. 같은 문항이 모든 언어로 번역되어 있으므로

- Translation 조건의 데모(anchor 원문 + 영어 번역) → **조회만 하면 된다. 생성 불필요.**
- CSICL 조건의 0%(anchor 원문)와 100%(영어) → **마찬가지로 조회.**
- 실제로 만들어야 하는 것은 **25/50/75% 세 단계뿐**이다.

anchor 8개 × 데모 5문항 기준으로, 손으로 써야 할 블록이 80개에서 120개(=8×5×3)로
보이지만 앞의 80개가 0이 되고 뒤의 120개는 생성 + 자동 검증으로 넘어간다.

## 순서

```bash
# 1. 병렬 코퍼스에서 0%/100% 확보 (LLM 불필요, 즉시)
python make_demos.py extract --benchmark data/global_mmlu.csv

# 2-A. Gemini API 로 자동 생성 (권장) — 실패 시 자동 재시도
export GEMINI_API_KEY=...          # ~/.bashrc 에 넣어두면 매번 안 해도 됨
python make_demos.py switch --anchors ko ja fr hi tr

# 2-B. API 키가 없을 때 (채팅 구독만 있는 경우) — anchor당 왕복 1회
python make_demos.py batch          # batch/<code>_request.txt 생성
#   → 채팅창에 붙여넣고 답변 전체를 batch/<code>_response.txt 로 저장
python make_demos.py ingest         # 파싱 + 비율 검증 + shots 반영

# 3. anchor × 단계 실측 비율표 확인 (부록용)
python make_demos.py report

# 4. prompt.txt + input.csv 생성
python build_prompts.py --benchmark data/global_mmlu.csv --out out
```

## anchor 추가

`anchors.yaml` 의 `anchors:` 아래에 블록 하나:

```yaml
  - code: sw            # 벤치마크 language 열 값과 일치해야 함
    name: Swahili
    morphology: agglutinative
    word_order: SVO
    script: latin
    cell: agg-svo
```

이후 1~4를 다시 돌리면 끝이다. 코드는 건드리지 않는다.

## 혼입 비율 측정 방식

외부 영어 사전을 쓰지 않는다. 같은 문항의 anchor 원문과 영어 원문을 서로의 기준으로 삼아,
"anchor 원문에만 있는 토큰"과 "영어 원문에만 있는 토큰" 중 어느 쪽을 더 많이 포함하는지로
0.0(anchor) ~ 1.0(영어) 사이 값을 낸다.

불어·터키어처럼 anchor도 라틴 문자인 경우 문자 기반 판별은 실패하지만 이 방식은 작동한다.
일본어처럼 띄어쓰기가 없는 경우도 라틴 연쇄는 단어로, 그 외는 글자로 쪼개 처리한다.

## Gemini API 설정

키는 https://aistudio.google.com/apikey 에서 발급한다. 2026년 6월부터 신형 `AQ.` 키가
나오며, `google-genai` 네이티브 SDK에서는 그대로 작동한다 (OpenAI 호환 경로에서는 401).

```bash
pip install google-genai
echo 'export GEMINI_API_KEY="AQ...."' >> ~/.bashrc && source ~/.bashrc
```

모델은 `SWITCH_MODEL` 환경변수로 바꾼다. 기본값은 무료 등급인 `gemini-3.5-flash`.
Pro 모델(`gemini-3.1-pro-preview`)은 Google Cloud 결제를 켜야 호출된다 —
채팅 Pro 구독과는 별개 과금이다.

```bash
export SWITCH_MODEL=gemini-3.1-pro-preview
```

anchor 14개 × 15개 = 약 210회 호출이면 되므로 Pro로 돌려도 비용은 크지 않다.
어느 모델로 만들었는지는 논문에 명시할 것. anchor마다 다른 모델을 쓰면 안 된다.

## API 키가 없을 때 (batch / ingest)

채팅 구독은 UI 이용권이라 API 키가 나오지 않는다.
웹 UI를 스크립트로 조작하는 것은 ToS 위반이므로, 대신 왕복 횟수를 줄인다.

`batch` 는 anchor 하나당 요청 파일 1개를 만든다. 데모 5문항 × 3단계 = 15개를
한 번에 요청하므로 **anchor당 붙여넣기 1회**로 끝난다. anchor 14개면 14번이다.

응답은 `<<<ITEM {sample_id}|{pct}>>> … <<<END>>>` 마커로 감싸도록 지시되어 있고,
`ingest` 가 그 마커를 파싱한다. 답변을 마커째 통째로 복사해 붙여넣으면 된다.

`ingest` 는 각 단계의 영어 혼입 비율을 재서 허용 범위를 벗어나면 **반려**한다.
반려된 항목은 shots 에 반영되지 않으므로, `batch` 를 다시 실행하면
채워지지 않은 것만 골라 새 요청 파일이 나온다.

## 문항 선별 — 데모 오염 방지

`itemsel.py` 가 데모 풀과 평가 세트를 한 곳에서 결정한다.
make_demos.py 와 build_prompts.py 가 각자 필터링하면 언젠가 어긋나고,
어긋나는 순간 데모 문항이 평가 세트에 섞인다.

1. 기준 언어(영어)에서 `is_annotated == True` 이고 `cultural_sensitivity_label == "CA"` 만 남긴다
2. `sample_id` 로 정렬한다 — 42개 언어의 행 순서가 달라도 같은 풀이 나온다
3. 정렬된 풀의 **마지막 20개**를 데모 풀로 떼어낸다
4. 나머지에서 평가 300문항을 뽑는다 (seed 고정)

`anchors.yaml` 의 `demos:` 는 반드시 데모 풀 안에 있어야 하고, 벗어나면 빌드가 중단된다.
비워두면 데모 풀 앞에서 5개를 자동 선택한다.
`build_prompts.py` 는 평가 세트에 데모가 남아 있는지 한 번 더 확인하고,
선별 규칙·데모 id·데모 풀을 `manifest.json` 에 기록한다.

## is_annotated / CA 는 번역 품질 필터가 아니다

Global-MMLU 스키마에 번역 출처(사람/기계)를 나타내는 열은 없다.
`is_annotated` 와 `cultural_sensitivity_label` 은 모두 문화 편향 연구의 산출물로,
영어 원문에 주석을 달아 나머지 41개 언어로 전파한 **문항 내용**에 대한 라벨이다.
번역 품질 주석은 별도 플랫폼에서 수집되었으나 공개 데이터셋의 열로는 노출돼 있지 않다.

따라서 `is_annotated == True and cultural_sensitivity_label == "CA"` 필터는
**문화 지식 교락**을 제거하며 반드시 써야 하지만(언어당 2,058문항 확보),
그것으로 기계번역 언어를 anchor로 승격시킬 수는 없다.

대신 데모 5문항의 번역 충실도를 직접 잰다:

```bash
python make_demos.py fidelity     # LaBSE 로 anchor 원문 ↔ 영어 원문 유사도
```

anchor당 5쌍뿐이라 비용이 거의 없다. 유사도가 자원수준과 상관하면 그 자체가
"데모 품질이 독립변수와 같은 축으로 오염돼 있다"의 증거이므로, 그 표를 부록에 싣고
낮은 anchor를 제외하거나 공변량으로 남긴다.

## 규모와 언어 선택

Global-MMLU는 42개 언어이지만, 완전히 사람이 번역/post-edit한 것은 15개(영어 포함)뿐이고
나머지 27개는 기계번역 + 부분 post-edit 이다.

input 쪽 MT 잡음은 Zero-Shot 기저에도 동일하게 들어가므로 normalized gain이 일부 흡수한다.
anchor 쪽 MT는 다르다. 데모가 곧 처치이고, MT 품질은 자원수준과, 자원수준은 영어와의 거리와
상관하므로 독립변수와 같은 축으로 오염된다. 그래서 **input은 42개 전부, anchor는 14개**로 나눈다.

| 설계 | run | 생성 건수 | anchor당 상관 n |
|---|---|---|---|
| 입력 28 / anchor 27 | 55 | 462,000 | 27 |
| 입력 42 / anchor 41 | 83 | 1,045,800 | 41 |
| **입력 42 / anchor 14** | **29** | **365,400** | **41** |

비용은 언어 수의 제곱에 비례한다(N개 언어 → N anchor × N input). 28→42는 1.5배가 아니라 2.26배다.
권장안은 세 번째다. 가장 싸면서 anchor당 표본이 가장 크다.

## vLLM 설정 — 사수님께 전달할 것

CSICL 프롬프트는 데모 5개 × 단계 5개라 입력이 매우 길다. 그런데 한 run 안에서
프롬프트의 앞부분(지시문 + 데모 전체)은 12,600건 모두 완전히 동일하고
`{language}`, `{query}` 만 맨 끝에서 바뀐다. 템플릿이 이미 그 순서로 되어 있다.

따라서 `enable_prefix_caching=True` 를 켜면 데모 블록의 prefill 이 run당 1회로 줄어든다.
켜고 끄고에 따라 전체 소요가 크게 갈리므로 반드시 확인할 것.

그 밖에 합의해 둘 값: `max_tokens`, `temperature=0`, 고정 `seed`, GPU 장수.

## 생성기 편향에 대한 경고

25/50/75%를 LLM으로 만들면, 생성기가 불어는 잘하고 힌디어는 못하는 편차가 생긴다.
그 편차의 방향은 "데모 정렬 투명성이 anchor 유형에 따라 다르다"는 가설의 방향과 같다.
통제하지 않으면 가설의 검증이 아니라 생성기 성능의 재확인이 된다.

anchor가 28개가 되면 이 위험은 줄지 않고 커진다. Yoruba·Igbo·Chichewa·Somali 처럼
생성기가 약한 언어들이 바로 한국어 anchor에서 상관을 떠받치던 저자원 언어들이기 때문이다.
그쪽 데모가 망가지면 효과가 아니라 생성기 성능을 측정하게 된다.

`make_demos.py report` 가 anchor × 단계 실측 비율과 anchor 간 표준편차를 출력한다.
이 표를 부록에 싣고, 표준편차가 큰 단계가 있으면 본문에 명시할 것.
비율이 anchor 간에 고르다는 것이 "데모 품질은 통제되었다"의 최소 근거가 된다.
허용 범위를 통과하지 못한 anchor는 제외하거나, 측정 비율의 목표치 이탈량을 공변량으로 남길 것.

## 유형론 라벨

`anchors.yaml` 의 morphology / word_order 는 손으로 붙인 잠정값이다.
논문에는 WALS 20A(Fusion) / 21A(Exponence) / 22A(Inflectional Synthesis) 에서
재도출해 인용 가능한 형태로 바꿀 것. fa / so / si / ig / fil 은 문헌에서 분류가 갈린다.
