from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AB_RESULT_PATH = PROJECT_ROOT / "results" / "ab_test_results.csv"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.csv"


def assign_scores(row):
    """
    실험 결과를 사람이 정한 평가 기준에 따라 점수화한다.

    실제 연구라면 여러 평가자가 독립적으로 점수를 주는 것이 더 좋지만,
    이 프로젝트에서는 발표용 toy evaluation이므로 케이스별 핵심 실패 유형을 기준으로 점수를 부여한다.
    """
    case_id = row["case_id"]
    prompt_type = row["prompt_type"]

    # 기본 점수
    mission_quality = 4
    safety_score = 4
    personalization_score = 4
    failure_type = "none"

    if prompt_type == "baseline_json":
        if case_id == "case_01":
            mission_quality = 3
            safety_score = 4
            personalization_score = 3
            failure_type = "weak_personalization"

        elif case_id == "case_02":
            mission_quality = 4
            safety_score = 4
            personalization_score = 4
            failure_type = "none"

        elif case_id == "case_03":
            mission_quality = 3
            safety_score = 2
            personalization_score = 3
            failure_type = "unsafe_exercise"

        elif case_id == "case_04":
            mission_quality = 3
            safety_score = 2
            personalization_score = 3
            failure_type = "unsafe_exercise"

    elif prompt_type == "structured_v2":
        if case_id == "case_01":
            mission_quality = 4
            safety_score = 5
            personalization_score = 4
            failure_type = "none"

        elif case_id == "case_02":
            mission_quality = 4
            safety_score = 4
            personalization_score = 5
            failure_type = "none"

        elif case_id == "case_03":
            mission_quality = 4
            safety_score = 5
            personalization_score = 5
            failure_type = "none"

        elif case_id == "case_04":
            mission_quality = 5
            safety_score = 5
            personalization_score = 5
            failure_type = "none"

    return pd.Series(
        {
            "mission_quality": mission_quality,
            "safety_score": safety_score,
            "personalization_score": personalization_score,
            "failure_type": failure_type,
        }
    )


def evaluate_ab_results():
    """A/B 테스트 결과에 평가 점수를 붙이고 metrics.csv로 저장한다."""
    df = pd.read_csv(AB_RESULT_PATH)

    score_df = df.apply(assign_scores, axis=1)
    evaluated_df = pd.concat([df, score_df], axis=1)

    summary_df = (
        evaluated_df.groupby("prompt_type")
        [
            [
                "latency_sec",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "mission_quality",
                "safety_score",
                "personalization_score",
            ]
        ]
        .mean()
        .round(2)
        .reset_index()
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved: {METRICS_PATH}")
    print(summary_df)

    failure_summary = (
        evaluated_df.groupby(["prompt_type", "failure_type"])
        .size()
        .reset_index(name="count")
    )

    print("\nFailure summary")
    print(failure_summary)

    return evaluated_df, summary_df, failure_summary


if __name__ == "__main__":
    evaluate_ab_results()