import json
from pathlib import Path

TRAIN_ROOT = Path("data/train")
VAL_ROOT = Path("data/val")

TRAIN_OUT = Path("dialect_train.jsonl")
VAL_OUT = Path("dialect_val.jsonl")


def extract_pairs(root):
    pairs = []
    seen = set()

    for path in root.rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print("SKIP:", path, e)
            continue

        utterances = data.get("utterance", [])

        for utt in utterances:
            dialect = utt.get("dialect_form", "").strip()
            standard = utt.get("standard_form", "").strip()

            if not dialect or not standard:
                continue

            # 실제 변환할 사투리 요소가 있는 문장만 사용
            if dialect == standard:
                continue

            key = (dialect, standard)

            if key in seen:
                continue

            seen.add(key)

            pairs.append({
                "messages": [
                    {
                        "role": "user",
                        "content": dialect
                    },
                    {
                        "role": "assistant",
                        "content": standard
                    }
                ]
            })

    return pairs


def save_jsonl(data, path):
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )


print("Training 데이터 추출 중...")
train = extract_pairs(TRAIN_ROOT)

print("Validation 데이터 추출 중...")
val = extract_pairs(VAL_ROOT)

save_jsonl(train, TRAIN_OUT)
save_jsonl(val, VAL_OUT)

print()
print("=" * 50)
print("완료")
print("Train pairs:", len(train))
print("Validation pairs:", len(val))
print("Train file:", TRAIN_OUT)
print("Val file:", VAL_OUT)
print("=" * 50)