import os
import json
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sample_cases.jsonl"
RESULT_PATH = PROJECT_ROOT / "results" / "ab_test_results.csv"

MODEL_NAME = "gemini-2.5-flash"


def load_jsonl(path):
    """JSONL 파일을 한 줄씩 읽어서 Python dict 리스트로 변환한다."""
    cases = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                cases.append(json.loads(line))

    return cases


def make_baseline_json_prompt(case):
    """
    가장 단순한 baseline 프롬프트.

    JSON 출력 형식은 맞추지만,
    안전성, 건강 상태, 난이도 조절 기준을 자세히 주지는 않는다.
    """
    return f"""
사용자에게 인지와 운동을 함께하는 미션을 하나 추천해줘.

출력은 반드시 JSON 형식으로 작성해줘.

출력 형식:
{{
  "mission_name": "",
  "cognitive_task": "",
  "movement_task": "",
  "difficulty": "",
  "success_condition": "",
  "reason": "",
  "safety_note": ""
}}

사용자 정보:
{json.dumps(case, ensure_ascii=False, indent=2)}
"""


def make_structured_prompt_v2(case):
    """
    Gemini 2.5 논문의 Prompt/Eval 관점을 반영한 구조화 프롬프트.

    baseline과 달리:
    - 건강 상태
    - 고령자 안전
    - 균형 상태
    - 난이도 조절 기준
    - 측정 가능한 성공 조건
    을 명확히 넣는다.
    """
    return f"""
너는 인지와 운동을 함께하는 미션 게임의 AI 코치다.

목표:
사용자의 최근 수행 결과를 분석하여 다음 미션을 생성한다.
미션은 인지 과제 1개와 운동 과제 1개를 반드시 포함해야 한다.

핵심 규칙:
1. 사용자의 실패 원인을 반드시 반영한다.
2. 성공 조건은 숫자로 측정 가능해야 한다.
3. 사용자의 현재 난이도보다 너무 급격하게 올리지 않는다.
4. 고령자는 안전을 우선한다.
5. health_condition이 true이면 건강 상태를 반드시 반영한다.
6. health_notes에 무릎 통증, 고혈압, 피로, 균형 문제가 있으면 운동 강도를 낮춘다.
7. balance_status가 unstable 또는 slightly_unstable이면 서서 하는 고난도 운동을 금지한다.
8. "빠르게", "점프", "스쿼트", "팔걸이 없이", "손을 쓰지 않고", "무릎을 짚지 않고" 같은 표현은 사용하지 않는다.
9. 균형이 불안정한 사용자는 팔걸이, 벽, 보호자 도움을 허용한다.
10. 운동 목표를 달성하지 못했으면 운동 난이도를 유지하거나 낮춘다.
11. 인지 정확도가 0.7 미만이면 인지 난이도를 유지하거나 낮춘다.
12. 운동과 인지를 모두 잘 수행했고 건강 이상이 없을 때만 난이도를 올린다.
13. 출력은 반드시 JSON 형식으로 작성한다.

출력 형식:
{{
  "mission_name": "",
  "cognitive_task": "",
  "movement_task": "",
  "difficulty": "",
  "difficulty_change": "",
  "success_condition": "",
  "personalized_reason": "",
  "safety_note": "",
  "expected_effect": "",
  "risk_check": {{
    "is_safe_for_user": true,
    "risk_reason": "",
    "modified_for_safety": ""
  }}
}}

사용자 기록:
{json.dumps(case, ensure_ascii=False, indent=2)}
"""


def call_gemini(client, prompt, json_mode=True):
    """Gemini API를 호출하고 응답 시간과 토큰 사용량을 함께 기록한다."""
    start = time.time()

    config = types.GenerateContentConfig(
        temperature=0.3,
        response_mime_type="application/json" if json_mode else "text/plain",
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=config,
    )

    end = time.time()
    usage = getattr(response, "usage_metadata", None)

    return {
        "text": response.text,
        "latency_sec": round(end - start, 3),
        "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
        "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
        "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
    }


def run_ab_test():
    """
    sample_cases.jsonl의 사용자 케이스를 읽고,
    baseline_json과 structured_v2 프롬프트 결과를 비교한다.
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY 환경변수가 없습니다. "
            "터미널에서 export GEMINI_API_KEY='본인_API_KEY'를 먼저 실행하세요."
        )

    client = genai.Client(api_key=api_key)
    cases = load_jsonl(DATA_PATH)

    results = []

    for case in cases:
        prompts = {
            "baseline_json": make_baseline_json_prompt(case),
            "structured_v2": make_structured_prompt_v2(case),
        }

        for prompt_type, prompt in prompts.items():
            result = call_gemini(client, prompt, json_mode=True)

            results.append(
                {
                    "case_id": case["case_id"],
                    "gender": case["gender"],
                    "user_level": case["user_level"],
                    "health_condition": case["health_condition"],
                    "prompt_type": prompt_type,
                    "output": result["text"],
                    "latency_sec": result["latency_sec"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "total_tokens": result["total_tokens"],
                }
            )

    df = pd.DataFrame(results)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved: {RESULT_PATH}")
    print(df[["case_id", "prompt_type", "latency_sec", "input_tokens", "output_tokens", "total_tokens"]])

    return df


if __name__ == "__main__":
    run_ab_test()