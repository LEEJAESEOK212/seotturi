import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "google/gemma-3-4b-it"
ADAPTER_DIR = "gemma3-dialect-lora"

dialects = [
    "니 지금 뭐하노?",
    "밥 묵었나?",
    "단디 해라.",
    "퍼뜩 온나.",
    "와 이리 늦었노?",
    "그거 니가 했제?",
    "마 됐다 아이가.",
    "이거 우예 하는데?",
]

system_prompt = """
너는 부산·경남 지역 사투리를 자연스러운 표준어로 변환하는 모델이다.

규칙:
1. 원래 문장의 의미를 바꾸지 않는다.
2. 사투리 표현만 자연스러운 표준어로 변환한다.
3. 반말은 반말로, 존댓말은 존댓말로 유지한다.
4. 설명이나 부가적인 문장을 출력하지 않는다.
5. 변환된 문장 하나만 출력한다.
"""


def generate(model, tokenizer, dialect):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": dialect},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


print("=== Tokenizer 로딩 ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


# -------------------------------------------------
# 1. 원본 Gemma
# -------------------------------------------------

print("\n=== 원본 Gemma 로딩 ===")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)

base_model.eval()

base_results = []

for dialect in dialects:
    answer = generate(base_model, tokenizer, dialect)
    base_results.append(answer)

print("원본 Gemma 평가 완료")


# 메모리 정리
del base_model
torch.cuda.empty_cache()


# -------------------------------------------------
# 2. LoRA 학습 Gemma
# -------------------------------------------------

print("\n=== LoRA Gemma 로딩 ===")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)

lora_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
)

lora_model.eval()

lora_results = []

for dialect in dialects:
    answer = generate(lora_model, tokenizer, dialect)
    lora_results.append(answer)


# -------------------------------------------------
# 결과 비교
# -------------------------------------------------

print("\n")
print("=" * 100)
print("원본 Gemma vs 사투리 LoRA Gemma")
print("=" * 100)

for dialect, base, lora in zip(
    dialects,
    base_results,
    lora_results,
):
    print()
    print("사투리       :", dialect)
    print("원본 Gemma   :", base)
    print("LoRA Gemma   :", lora)
    print("-" * 100)