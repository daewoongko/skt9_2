import os
import json
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "project_demo_cases.json"
RESULT_PATH = PROJECT_ROOT / "results" / "coach_demo_results.csv"

MODEL_NAME = "gemini-2.5-flash"


def load_demo_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_mission_coach_prompt(case):
    return f"""
너는 고령자를 위한 인지-운동 미션 게임의 AI 코치다.

아래 데이터는 실제 서비스에서 들어온다고 가정한 값이다.
- user_profile: 사용자 기본 정보
- recent_history: 최근 미션 수행 기록
- pose_model_output: 포즈 인식 모델이 분석한 운동 결과
- cognitive_model_output: 인지 과제 수행 결과

너의 역할은 이 정보를 종합해서 다음 미션을 추천하는 것이다.

중요 규칙:
1. 고령자 대상이므로 안전을 최우선으로 한다.
2. health_condition이 true이면 운동 강도를 무리하게 올리지 않는다.
3. balance_status가 unstable 또는 slightly_unstable이면 균형 부담이 큰 운동을 피한다.
4. knee_discomfort가 있으면 무릎에 부담이 큰 스쿼트, 빠른 반복 운동을 피한다.
5. cognitive accuracy가 낮거나 reaction_time이 느리면 인지 과제를 단순하게 조정한다.
6. 사용자의 이전 결과를 근거로 난이도 변화를 설명한다.
7. 반드시 JSON만 출력한다.

출력 JSON 형식:
{{
  "case_id": "...",
  "mission_name": "...",
  "mission_goal": "...",
  "cognitive_task": "...",
  "movement_task": "...",
  "difficulty": "...",
  "difficulty_change": "...",
  "success_condition": "...",
  "coach_feedback": "...",
  "personalized_reason": "...",
  "safety_note": "...",
  "expected_effect": "..."
}}

입력 데이터:
{json.dumps(case, ensure_ascii=False, indent=2)}
"""


def call_gemini(client, prompt):
    start_time = time.time()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    latency_sec = round(time.time() - start_time, 3)
    usage = getattr(response, "usage_metadata", None)

    input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
    output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
    total_tokens = getattr(usage, "total_token_count", None) if usage else None

    return {
        "output": response.text,
        "latency_sec": latency_sec,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def summarize_pose(case):
    pose = case.get("pose_model_output", {})

    movement = (
        pose.get("movement")
        or pose.get("exercise_name")
        or pose.get("movement_type")
        or pose.get("task_type")
        or "unknown_movement"
    )

    completed_count = (
        pose.get("completed_count")
        or pose.get("count")
        or pose.get("success_count")
        or "-"
    )

    target_count = (
        pose.get("target_count")
        or pose.get("target")
        or pose.get("goal_count")
        or "-"
    )

    balance_status = (
        pose.get("balance_status")
        or pose.get("balance")
        or "unknown"
    )

    failed_reason = (
        pose.get("failed_reason")
        or pose.get("failure_reason")
        or pose.get("reason")
        or "none"
    )

    return (
        f"{movement} / "
        f"{completed_count}/{target_count}회 / "
        f"balance={balance_status} / "
        f"failed_reason={failed_reason}"
    )


def summarize_cognitive(case):
    cog = case.get("cognitive_model_output", {})

    task_type = (
        cog.get("task_type")
        or cog.get("task_name")
        or cog.get("cognitive_task")
        or "unknown_cognitive_task"
    )

    accuracy = (
        cog.get("accuracy")
        or cog.get("score")
        or "-"
    )

    reaction_time = (
        cog.get("reaction_time_sec")
        or cog.get("reaction_time")
        or cog.get("response_time_sec")
        or "-"
    )

    return (
        f"{task_type} / "
        f"accuracy={accuracy} / "
        f"reaction_time={reaction_time}초"
    )

def run_mission_coach_demo():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY 환경 변수가 없습니다.")

    client = genai.Client(api_key=api_key)
    cases = load_demo_cases(DATA_PATH)

    rows = []

    for case in cases:
        prompt = build_mission_coach_prompt(case)
        result = call_gemini(client, prompt)

        profile = case["user_profile"]

        rows.append({
            "case_id": case["case_id"],
            "age_group": profile["age_group"],
            "gender": profile["gender"],
            "user_level": profile["user_level"],
            "health_condition": profile["health_condition"],
            "pose_summary": summarize_pose(case),
            "cognitive_summary": summarize_cognitive(case),
            "output": result["output"],
            "latency_sec": result["latency_sec"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "total_tokens": result["total_tokens"],
        })

    df = pd.DataFrame(rows)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved: {RESULT_PATH}")
    print(df[[
        "case_id",
        "gender",
        "user_level",
        "health_condition",
        "latency_sec",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ]])


if __name__ == "__main__":
    run_mission_coach_demo()