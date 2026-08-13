import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


MODEL_ID = "google/gemma-3-4b-it"
ADAPTER_DIR = "gemma3-dialect-lora"


st.set_page_config(
    page_title="서투리",
    page_icon="🗣️",
    layout="centered",
)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
    )

    model.eval()

    return tokenizer, model


def convert_dialect(text, tokenizer, model):

    system_prompt = """
너는 부산·경남 지역 사투리를 자연스러운 표준어로 변환하는 모델이다.

규칙:
1. 원래 문장의 의미를 바꾸지 않는다.
2. 사투리 표현만 자연스러운 표준어로 변환한다.
3. 반말은 반말로, 존댓말은 존댓말로 유지한다.
4. 설명이나 부가적인 문장을 출력하지 않는다.
5. 변환된 문장 하나만 출력한다.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    return answer


st.title("서투리")
st.caption("부산·경남 사투리를 자연스러운 표준어로 변환합니다.")

tokenizer, model = load_model()

dialect = st.text_area(
    "사투리를 입력하세요",
    placeholder="예: 와 이리 늦었노?",
    height=120,
)

if st.button("표준어로 변환", use_container_width=True):

    if not dialect.strip():
        st.warning("문장을 입력해주세요.")

    else:
        with st.spinner("변환 중..."):
            result = convert_dialect(
                dialect.strip(),
                tokenizer,
                model,
            )

        st.subheader("변환 결과")
        st.success(result)