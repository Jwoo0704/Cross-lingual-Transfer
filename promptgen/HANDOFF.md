# 전달 규격 (H100 / vLLM)

## 전달하는 것

| 파일 | 개수 | 내용 |
|---|---|---|
| `input.csv` | 1개 (전 조건 공용) | `id`, `language`, `query` |
| `prompt.txt` | 조건×anchor마다 1개 | `{language}`, `{query}` 자리표시자를 포함한 템플릿 |
| `extract_answers.py` | 1개 | 출력에서 A–D를 뽑는 스크립트 (추론 코드 아님) |

`answer_key.csv`는 전달하지 않습니다. 채점은 결과를 받은 뒤 이쪽에서 합니다.

## 치환 방식

`.format()` 대신 문자열 치환을 써 주십시오. 문항 본문에 중괄호가 들어 있어도 깨지지 않습니다.

```python
template = open("prompt.txt", encoding="utf-8").read()
prompt = template.replace("{language}", row["language"]).replace("{query}", row["query"])
```

`{language}`와 `{query}` 외의 자리표시자는 없습니다. `build_prompts.py`가 전달 전에 이를 검증합니다.

## 결과 파일에 필요한 것

`id` — `input.csv`의 `id`를 그대로. 행 순서는 뒤바뀌어도 됩니다.
`output` — 생성된 텍스트 원문. 잘라내거나 정리하지 말아 주십시오.
`finish_reason` — **가능하면 꼭 포함.** 이게 있어야 "잘린 것"과 "답 형식을 못 찾은 것"을 구분할 수 있고, 재실행 대상을 잘린 건으로만 좁힐 수 있습니다.

jsonl 또는 csv 둘 다 받습니다.

```jsonl
{"id": "Yoruba::high_school_biology/test/12", "output": "Let's translate ...\nThe answer is C", "finish_reason": "stop"}
```

## 생성 설정 합의 사항

- `max_tokens`: 값을 알려주십시오. 잘린 문항 재실행 때 상한을 올려야 합니다.
- `temperature`: 0 (재현성).
- `seed`: 고정값 하나로 통일.
- 위 세 값을 결과와 함께 남겨 주시면 논문 부록에 그대로 씁니다.

## 답 추출

```bash
python extract_answers.py --pred results.jsonl --out out
```

`out/extracted.csv` — `id`, `pred`, `status`, `matched_by`
`out/retry_ids.txt` — `status != ok` 인 id 목록 (재실행 대상)

`status`는 `ok` / `truncated` / `no_match` 세 가지입니다. `truncated`만 상한을 올려 재실행하면 되고, `no_match`는 프롬프트나 출력 형식 쪽 문제이므로 원문을 직접 봐야 합니다.

추출 규칙은 우선순위 순으로 `answer is X` → `\boxed{X}` → `**X**` → 마지막 줄 단독 A–D이며, 각 패턴에서 **마지막** 매치를 채택합니다. Gemma의 thinking 채널이 새는 경우가 있어 탐색 전에 `<think>` 블록을 제거합니다.
