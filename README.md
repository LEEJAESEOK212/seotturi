# 서투리 (Seotturi)

AI 기반 사투리 → 표준어 코칭 서비스

## 프로젝트 소개

서투리는 사용자의 사투리를 자연스러운 표준어로 변환하고,
변경된 표현과 의미를 설명하는 AI 코칭 서비스입니다.

장기적으로는 음성 입력을 받아
STT → 사투리 분석 → 표준어 변환 → TTS까지 제공하는
음성 기반 AI 서비스로 확장하는 것을 목표로 합니다.

## 현재 기능

- Streamlit 기반 웹 UI
- Qwen3-0.6B 로컬 실행
- 사투리 텍스트 입력
- 표준어 변환
- 변경 표현 및 설명 생성

## 기술 스택

- Python
- PyTorch
- Hugging Face Transformers / HuBert https://huggingface.co/team-lucid/hubert-base-korean
- Qwen
- Streamlit
- Git / GitHub

## 현재 구조

사용자 입력
→ Streamlit
→ Tokenizer
→ Qwen
→ Generate
→ 결과 출력

## 향후 계획

- Qwen 4B~8B 모델 테스트
- Whisper 기반 음성 입력
- 사투리 데이터셋 구축
- LoRA Fine-tuning
- OpenAI Teacher 기반 Distillation
- DPO / GRPO 기반 정렬
- TTS 및 발음·억양 피드백
- 서비스 배포
