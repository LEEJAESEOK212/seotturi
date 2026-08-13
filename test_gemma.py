import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-3-4b-it"

print("Gemma 로딩 중...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

print("Device:", next(model.parameters()).device)
print("Gemma 로딩 완료!\n")

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

examples = [
    ("단디 해라.", "제대로 해라."),
    ("퍼뜩 온나.", "빨리 와라."),
    ("와 이리 늦었노?", "왜 이렇게 늦었어?"),
    ("마 됐다 아이가.", "됐잖아."),
    ("밥 묵었나?", "밥 먹었어?"),
    ("그거 니가 했제?", "그거 네가 했지?"),
]

for dialect in dialects:
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    for src, tgt in examples:
        messages.append({"role": "user", "content": src})
        messages.append({"role": "assistant", "content": tgt})

    messages.append(
        {"role": "user", "content": dialect}
    )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
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

    print("=" * 60)
    print("사투리 :", dialect)
    print("표준어 :", answer)

print("=" * 60)