# 로컬 환경

설치일: 2026-09-24

- Python 3.14.7, 프로젝트 전용 `.venv`.
- MLX 0.32.2 / MLX-LM 0.31.3 (train 추가 의존성 포함).
- Hugging Face Hub CLI 1.32.0.
- 전체 설치 버전: `requirements.lock.txt`.
- 모델: `mlx-community/gemma-3-text-4b-it-4bit`.
- 고정 리비전: `4f665a4c50ecfe4ecdc34056ab52fe3e3c4abf9e`.
- 모델 위치: `models/gemma-3-text-4b-it-4bit`.
- 모델 출처: https://huggingface.co/mlx-community/gemma-3-text-4b-it-4bit

## VS Code

`new_seotturi` 폴더 자체를 열면 `.vscode/settings.json`의 가상환경 설정이 적용된다.
상위 `서투리 프로젝트` 폴더를 열어둔 경우 Python 인터프리터로
`new_seotturi/.venv/bin/python`을 선택한다.

## 터미널에서 실행

```bash
cd '/Users/iamhiphop/Documents/ChatGPT/서투리 프로젝트/new_seotturi'
source .venv/bin/activate
mlx_lm.generate --model ./models/gemma-3-text-4b-it-4bit --prompt '다음 사투리를 표준어로 바꾸고 변환된 문장만 출력해: 니 밥 묵었나?' --max-tokens 64 --temp 0
```

이 명령은 설치 확인용이다. 최종 서비스 프롬프트나 성능 평가 규약은 아니다.

## 터미널 대화

프로젝트 폴더에서 `.venv/bin/python scripts/chat.py`를 실행한다.
이 실행기는 Python readline을 활성화해 방향키와 줄 편집을 지원한 뒤 기존 MLX 채팅을 시작한다.
`q`는 종료, `r`은 대화 초기화다.
`prompts/converter.txt`의 서투리 전용 프롬프트가 기본 적용된다. 질문에 답하는 일반 대화가 아니라 입력 문장을 표준어로 바꾸는 모드다.
프롬프트 파일을 수정하면 채팅을 종료한 뒤 다시 실행한다. 명령줄 `--system-prompt`로 기본 지시를 교체할 수 있다.
한글은 터미널에서 보통 두 칸 너비로 표시되므로 자간만으로 실제 공백 여부를 판단하지 않는다.

전용 프롬프트 적용 후 수동 확인: `니 밥 묵었나?` → `너 밥 먹었니?`,
`오늘 좀 늦을 것 같아요.` 및 `너 이름이 뭐야?`는 그대로 유지했다.
이 세 문장은 프롬프트 예시에 포함되어 있으므로 일반화 성능 평가가 아니다.
예시에 없는 `그거 내 안 했다.`에서는 `그거 내 안 했어.`가 나와 대명사 변환이 미흡했다.
프롬프트 적용만으로 품질 개선이 완료된 것은 아니다.

## 재설치

같은 Python 버전과 Apple Silicon macOS 환경에서:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
HF_HUB_DISABLE_XET=1 HF_HOME="$PWD/.cache/huggingface" .venv/bin/hf download mlx-community/gemma-3-text-4b-it-4bit --revision 4f665a4c50ecfe4ecdc34056ab52fe3e3c4abf9e --local-dir models/gemma-3-text-4b-it-4bit
```

이 공개 모델 다운로드에는 이번 설치에서 Hugging Face 로그인이 필요하지 않았다.
향후 접근 제한 저장소를 사용할 때는 터미널에서 `hf auth login`으로 로그인한다.
모델은 Gemma 라이선스를 따르며 원본 모델과 변환 모델의 문서를 참고한다.

실제 데이터로 학습하거나 어댑터를 생성하는 작업은 아직 수행하지 않았다.

## 설치 검증 결과

- 패키지 의존성 검사: `pip check` 통과.
- Metal GPU 연산 확인 완료.
- MLX 생성·LoRA 학습 CLI의 도움말 실행 확인 완료.
- 고정 리비전의 모델 다운로드와 오프라인 추론 확인 완료.
- 설치 확인 입력: `니 밥 묵었나?`
- 실제 출력: `네, 밥 먹었나요?`
- 모델 로딩 1.505초, 프롬프트 처리 포함 생성 1.557초.
- MLX가 기록한 최대 메모리 2.649GB. 전체 프로세스·시스템 메모리나 학습 메모리를 뜻하지 않는다.

위 출력은 원문 반말을 존댓말로 바꾸고 불필요한 `네,`를 추가했다.
설치는 정상이나 이 사례는 프로젝트의 변환 규칙을 충족하지 못한다.
한 문장의 결과이므로 전체 품질·속도 벤치마크로 해석하지 않는다.

재확인 명령:

```bash
.venv/bin/python scripts/check_environment.py
```

결과는 `outputs/environment-check.json`에 저장된다.
