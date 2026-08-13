import tempfile

import streamlit as st
import torch
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer


# =========================================================
# 모델 설정
# =========================================================

QWEN_MODEL_NAME = "Qwen/Qwen3-0.6B"
WHISPER_MODEL_NAME = "base"


# =========================================================
# Qwen 모델 로딩
# =========================================================

@st.cache_resource
def load_qwen():
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

    return tokenizer, model


# =========================================================
# Whisper 모델 로딩
# =========================================================

@st.cache_resource
def load_whisper():
    return whisper.load_model(WHISPER_MODEL_NAME)


# =========================================================
# Streamlit 기본 설정
# =========================================================

st.set_page_config(
    page_title="서투리",
    page_icon="🗣️",
    layout="centered"
)

st.title("🗣️ 서투리")
st.subheader("AI 사투리 코칭 서비스")

st.write(
    "사투리를 입력하거나 직접 말하면 "
    "사투리를 분석하고 자연스러운 표준어 표현을 제공합니다."
)


# =========================================================
# 모델 불러오기
# =========================================================

with st.spinner("AI 모델을 불러오는 중입니다..."):
    tokenizer, qwen_model = load_qwen()
    whisper_model = load_whisper()


# =========================================================
# 음성 입력
# =========================================================

st.markdown("---")
st.markdown("## 🎤 음성 입력")

st.write(
    "사투리로 직접 말해보세요. "
    "Whisper가 음성을 텍스트로 변환합니다."
)

audio = st.audio_input(
    "🎙️ 사투리로 말해보세요"
)


if audio is not None:

    # 사용자가 녹음한 음성을 다시 재생
    st.audio(audio)

    # Whisper가 읽을 수 있도록 임시 wav 파일 생성
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_audio:

        temp_audio.write(
            audio.getbuffer()
        )

        audio_path = temp_audio.name

    # Whisper 음성 인식
    with st.spinner(
        "Whisper가 음성을 인식하고 있습니다..."
    ):

        whisper_result = whisper_model.transcribe(
            audio_path,
            language="ko"
        )

    recognized_text = whisper_result["text"].strip()

    st.markdown("### 📝 Whisper 인식 결과")

    st.success(recognized_text)


# =========================================================
# 텍스트 입력
# =========================================================

st.markdown("---")
st.markdown("## ⌨️ 텍스트 입력")

dialect = st.text_area(
    "사투리 문장을 입력하세요.",
    placeholder="예) 니 밥 무읏나?"
)


# =========================================================
# Qwen 사투리 변환
# =========================================================

if st.button("교정하기"):

    if not dialect.strip():

        st.warning(
            "사투리 문장을 입력해주세요."
        )

    else:

        messages = [
            {
                "role": "system",
                "content": """
너는 대한민국 지역 사투리를
표준어로 변환하는 전문가다.

반드시 다음 규칙을 지켜라.

1. 입력 문장의 의미를 임의로 바꾸지 않는다.
2. 시제와 의문문 여부를 유지한다.
3. 사투리 어휘를 직접 대응되는 표준어로 변환한다.
4. 존댓말과 반말 수준을 유지한다.
5. 사투리를 틀린 표현이라고 평가하지 않는다.
6. 확실하지 않으면 추측하지 않는다.

예시:

입력: 니 뭐하노?
표준어: 너 뭐 하니?

입력: 밥 무읏나?
표준어: 밥 먹었니?

입력: 와 이카노?
표준어: 왜 이러니?

반드시 아래 형식으로만 답한다.

표준어:
변경된 표현:
설명:
"""
            },

            {
                "role": "user",
                "content": dialect
            }
        ]


        # Qwen Chat Template 적용
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )


        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt"
        ).to(qwen_model.device)


        # Qwen 추론
        with st.spinner(
            "Qwen이 사투리를 분석하고 있습니다..."
        ):

            with torch.no_grad():

                outputs = qwen_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False
                )


        # 새로 생성된 부분만 가져오기
        generated = outputs[0][
            inputs.input_ids.shape[1]:
        ]


        # 토큰 → 문자열
        answer = tokenizer.decode(
            generated,
            skip_special_tokens=True
        )


        # =================================================
        # 결과 출력
        # =================================================

        st.success(
            "교정 완료!"
        )

        st.markdown(
            "### 입력 문장"
        )

        st.write(
            dialect
        )

        st.markdown(
            "### Qwen 교정 결과"
        )

        st.write(
            answer
        )