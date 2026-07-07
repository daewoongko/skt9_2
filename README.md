# SKT FLY AI Paper Project - Gemini 기반 인지-운동 미션 코치

## 1. 프로젝트 개요

본 프로젝트는 Gemini 2.5 논문에서 다루는 LLM의 추론, 프롬프트 설계, 평가, 토큰 사용량, 지연시간 측정 아이디어를 바탕으로 진행한 소규모 재현 및 응용 프로젝트입니다.

팀 주제는 **인지와 운동을 함께하는 미션 게임**입니다.  
사용자의 포즈 인식 결과와 인지 과제 수행 결과를 입력으로 받아, Gemini API가 다음 개인화 미션을 추천하는 구조를 실험했습니다.

## 2. 프로젝트 목표

이 프로젝트의 목표는 Gemini 모델 자체를 학습시키는 것이 아니라, 실제 서비스에서 LLM을 사용할 때 중요한 요소를 작게 재현하는 것입니다.

- 사용자 상태에 따른 개인화 미션 생성
- baseline prompt와 structured prompt 비교
- 응답 품질, 안전성, 개인화 정도 평가
- latency, input token, output token, total token 측정
- 포즈 인식 결과와 인지 결과를 활용한 AI 코치 데모 구현

## 3. 프로젝트 구조

```text
skt9_2/
├── README.md
├── requirements.txt
├── notebooks/
│   └── demo.ipynb
├── src/
│   ├── baseline.py
│   ├── mission_coach.py
│   └── evaluate.py
├── data/
│   ├── sample_cases.jsonl
│   ├── project_demo_cases.json
│   └── scoring_criteria.md
├── results/
│   ├── metrics.csv
│   ├── ab_test_results.csv
│   └── coach_demo_results.csv
└── docs/
    ├── paper_card.md
    ├── project_canvas.md
    ├── experiment_plan.md
    ├── prompt_design.md
    └── failure_analysis.md