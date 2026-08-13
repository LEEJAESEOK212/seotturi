import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_ID = "google/gemma-3-4b-it"

TRAIN_FILE = "dialect_train_10k.jsonl"
VAL_FILE = "dialect_val_1k.jsonl"

OUTPUT_DIR = "gemma3-dialect-lora"


print("=== 데이터 로딩 ===")

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
    },
)

print(dataset)
print("Train:", len(dataset["train"]))
print("Validation:", len(dataset["validation"]))


print("\n=== Tokenizer 로딩 ===")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("\n=== Gemma 3 4B 로딩 ===")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)

model.config.use_cache = False


print("\n=== LoRA 설정 ===")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,

    # Gemma의 linear layer들에 LoRA 적용
    target_modules="all-linear",

    bias="none",
    task_type="CAUSAL_LM",
)


print("\n=== Training 설정 ===")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # 1차 실험이므로 1 epoch
    num_train_epochs=1,

    # GB10에서 우선 안전하게 시작
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    bf16=True,
    fp16=False,

    logging_steps=10,

    eval_strategy="steps",
    eval_steps=100,

    save_strategy="steps",
    save_steps=100,

    save_total_limit=2,

    max_length=512,

    report_to="none",

    gradient_checkpointing=True,

    # conversational dataset
    assistant_only_loss=True,

    dataset_num_proc=4,
)


print("\n=== Trainer 생성 ===")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    peft_config=peft_config,
)


print("\n=== 학습 시작 ===")

trainer.train()


print("\n=== LoRA 저장 ===")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print()
print("=" * 60)
print("학습 완료")
print("LoRA adapter:", OUTPUT_DIR)
print("=" * 60)