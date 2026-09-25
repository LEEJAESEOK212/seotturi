# 서투리 (Seotturi)

사투리를 원래 의미와 말투를 유지하면서 자연스러운 표준어로 바꾸는 AI 코칭 프로젝트입니다.
사투리를 틀린 말로 평가하지 않고, 상황에 맞는 표준어 표현을 익히도록 돕습니다.

현재 개발은 **[new_seotturi/](new_seotturi/)** 에서 진행합니다. 작업 브랜치는 **`codex/new-seotturi`** 입니다.

## 현재 상태

- Apple Silicon에서 MLX 기반 Gemma 3 4B 텍스트용 4비트 모델 실행 확인.
- 의미·시제·부정·말투 보존을 위한 서투리 전용 프롬프트 적용.
- 한글 입력 편집을 지원하는 터미널 변환 실행기 준비.
- 강원·경상·전라·제주·충청 데이터 전체 점검 및 지역별 통합 완료.
- **QLoRA-SFT 학습은 아직 시작하지 않았습니다.** 학습 속도와 품질도 아직 측정 전입니다.

## 데이터 준비 결과

| 항목 | 발화 수 |
|---|---:|
| 처리한 원본 | 11,720,550 |
| 정확 문맥 중복 제외 | 298,548 |
| 보존한 통합 데이터 | 11,422,002 |
| 보존 데이터 중 입력과 정답이 같은 사례 | 8,364,126 |

개수 제한이나 변환/유지 비율 조정 없이 통합했습니다. 같은 지역·같은 분할 안에서 원문·정답·화자 ID·앞뒤 발화 등 중복 기준 필드가 모두 같은 경우만 제외했습니다.
Training과 Validation은 별도 파일로 유지합니다. 이 수치는 정답 품질 검수를 통과한 학습 문장 수가 아니라 통합 발화 수입니다.
전라도 TXT에서 기계적으로 추출한 대응쌍과 확인 필요 항목도 보존했습니다.

모델 가중치, 원본/가공 데이터, 가상환경, 캐시는 GitHub에 포함하지 않습니다.
[지역별 집계](new_seotturi/docs/regional-corpus-results.md) · [중복 정제 기준](new_seotturi/docs/regional-corpus.md)

## 로컬 실행

Apple Silicon Mac에서 검증했습니다. 전체 설치 버전과 모델 리비전은 [환경 안내](new_seotturi/docs/setup.md)에 기록되어 있습니다.

```bash
cd new_seotturi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
HF_HUB_DISABLE_XET=1 HF_HOME="$PWD/.cache/huggingface" hf download mlx-community/gemma-3-text-4b-it-4bit --revision 4f665a4c50ecfe4ecdc34056ab52fe3e3c4abf9e --local-dir models/gemma-3-text-4b-it-4bit
python scripts/chat.py
```

이미 설치한 환경에서는 `cd new_seotturi`, `source .venv/bin/activate`, `python scripts/chat.py`만 실행하면 됩니다.
이 실행기는 일반 질의응답이 아니라 **입력 문장 자체를 표준어로 변환**하는 모드입니다. `q`로 종료하고 `r`로 대화를 초기화합니다.
아직 사투리 추가 학습 전 모델이므로 부정확한 변환이 나올 수 있습니다.

## 개발 방향

1. 통합 원본을 실제 모델 입력 형식으로 변환하고, 평가 데이터와의 겹침 및 정답 품질을 정리합니다.
2. 학습 전 변환 결과를 기록한 뒤 짧은 QLoRA 시험 실행으로 속도·메모리·소요 시간을 측정합니다.
3. 측정 결과를 바탕으로 본 학습 규모와 실행 방식을 결정합니다.
4. 텍스트 변환 품질을 확인한 뒤 음성 인식과 설명·음성 출력으로 확장합니다.

주 학습 방식은 **MLX 기반 4비트 QLoRA-SFT**입니다. GRPO는 현재 구현 범위가 아닙니다.

## 주요 파일

- [프로젝트 상세](new_seotturi/README.md)
- [변환 프롬프트](new_seotturi/prompts/converter.txt)
- [데이터 점검 보고서](new_seotturi/docs/data-audit.md)
- [데이터 통합 도구](new_seotturi/scripts/build_regional_corpus.py)
- [모델 정보](new_seotturi/docs/model-manifest.json)

루트의 기존 Qwen·Gemma·Streamlit 파일은 이전 실험 코드입니다. 현재 개발 코드와 설정은 `new_seotturi/`를 기준으로 확인하세요.
