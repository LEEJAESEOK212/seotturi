import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-0.6B"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

    return tokenizer, model


st.set_page_config(
    page_title="서투리",
    page_icon="🗣️",
    layout="centered"
)

st.title("🗣️ 서투리")
st.subheader("Qwen 기반 AI 사투리 코칭 서비스")

st.write(
    "사투리 문장을 입력하면 자연스러운 표준어와 함께 설명을 제공합니다."
)


with st.spinner("Qwen 모델을 불러오는 중입니다..."):
    tokenizer, model = load_model()


dialect = st.text_area(
    "사투리 입력",
    placeholder="예) 니 밥 무읏나?"
)


if st.button("교정하기"):

    if not dialect.strip():
        st.warning("문장을 입력해주세요.")

    else:
        messages = [
            {
                "role": "system",
                "content": """
너는 대한민국 지역 사투리를 표준어로 변환하는 전문가다.

반드시 다음 규칙을 지켜라.

1. 입력 문장의 의미를 임의로 바꾸지 않는다.
2. 시제와 의문문 여부를 반드시 유지한다.
3. 사투리 어휘를 가능한 한 직접 대응되는 표준어로 변환한다.
4. 존댓말/반말 수준을 유지한다.
5. 사투리를 틀린 표현이라고 평가하지 않는다.
6. 모르면 추측하지 말고 "확실하지 않음"이라고 표시한다.

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

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = tokenizer(
            text,
            return_tensors="pt"
        ).to(model.device)

        with st.spinner("사투리를 분석하고 있습니다..."):

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False
                    )
                

        generated = outputs[0][inputs.input_ids.shape[1]:]

        answer = tokenizer.decode(
            generated,
            skip_special_tokens=True
        )

        st.success("교정 완료!")

        st.markdown("### 입력 문장")
        st.write(dialect)

        st.markdown("### Qwen 교정 결과")
        st.write(answer)