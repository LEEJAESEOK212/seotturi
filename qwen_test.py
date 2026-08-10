import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-0.6B"


print("Qwen 로딩 중...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

messages = [
    {
        "role": "system",
        "content": """
너는 한국어 사투리 코칭 AI다.

사용자가 입력한 사투리의 의미와 말투를 최대한 유지하면서
자연스러운 표준어 표현을 제안한다.

사투리를 틀렸거나 열등한 표현으로 평가하지 않는다.

다음 형식으로 답한다.

표준어:
변경된 표현:
설명:
"""
    },
    {
        "role": "user",
        "content": "니 밥 무읏나?"
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

with torch.no_grad():git --version
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

generated = outputs[0][inputs.input_ids.shape[1]:]

answer = tokenizer.decode(
    generated,
    skip_special_tokens=True
)

print("\n===== 서투리 결과 =====")
print(answer)