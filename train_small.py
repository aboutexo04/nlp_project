"""
KoBART Fine-tuning - 소량 테스트용 (MPS 지원)
소량 데이터로 빠르게 파이프라인 테스트
"""

import torch
from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ===== 설정 =====
SAMPLE_SIZE = 1000          # 사용할 데이터 개수
NUM_EPOCHS = 2              # 에포크 수
BATCH_SIZE = 4              # 배치 크기
LEARNING_RATE = 5e-5        # 학습률
MAX_INPUT_LENGTH = 512      # 최대 입력 길이
MAX_TARGET_LENGTH = 128     # 최대 요약 길이
OUTPUT_DIR = "./kobart_test"

print("="*70)
print("KoBART Fine-tuning - 소량 테스트 (MPS)")
print("="*70)
print(f"샘플 크기: {SAMPLE_SIZE}")
print(f"에포크: {NUM_EPOCHS}")
print(f"배치 크기: {BATCH_SIZE}")
print("="*70)

# ===== 1. 디바이스 설정 =====
def setup_device():
    """디바이스 설정 (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS (Apple Silicon GPU)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "CUDA GPU"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    print(f"\n✓ 디바이스: {device_name}")

    # 디바이스 테스트
    try:
        x = torch.randn(3, 3).to(device)
        y = x + x
        print(f"✓ 디바이스 연산 테스트 성공")
    except Exception as e:
        print(f"⚠️ 디바이스 오류: {e}")
        device = torch.device("cpu")
        print("CPU로 전환합니다.")

    return device

device = setup_device()

# ===== 2. 데이터 로드 =====
def load_data(data_path='./data/train/', sample_size=SAMPLE_SIZE):
    """JSON 데이터 로드"""
    print("\n" + "="*70)
    print("데이터 로드")
    print("="*70)

    dialogues = []
    summaries = []

    json_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))

    print(f"JSON 파일 {len(json_files)}개 발견")

    # 각 파일에서 데이터 수집
    for json_file in tqdm(json_files, desc="파일 로딩"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'data' in data and isinstance(data['data'], list):
                for item in data['data']:
                    if 'body' in item and 'dialogue' in item['body']:
                        dialogue_list = item['body']['dialogue']
                        dialogue_text = ' '.join([utt['utterance'] for utt in dialogue_list if 'utterance' in utt])

                        if 'summary' in item['body']:
                            summary = item['body']['summary']

                            if dialogue_text and summary:
                                dialogues.append(dialogue_text)
                                summaries.append(summary)

                        # 충분한 데이터를 수집하면 중단
                        if len(dialogues) >= sample_size * 2:
                            break

        except Exception as e:
            print(f"에러: {os.path.basename(json_file)}: {e}")
            continue

        if len(dialogues) >= sample_size * 2:
            break

    df = pd.DataFrame({'dialogue': dialogues, 'summary': summaries})

    # 샘플링
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print(f"✓ 데이터 로드 완료: {len(df):,}개")
    return df

train_df = load_data()

# ===== 3. Train/Val 분할 =====
print("\n데이터 분할 (80:20)")
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

print(f"Train: {len(train_data):,}개")
print(f"Validation: {len(val_data):,}개")

print(f"\n샘플 확인:")
print(f"대화: {train_data.iloc[0]['dialogue'][:100]}...")
print(f"요약: {train_data.iloc[0]['summary']}")

# ===== 4. 모델과 토크나이저 로드 =====
print("\n" + "="*70)
print("모델 로드")
print("="*70)

model_name = "gogamza/kobart-base-v2"
print(f"모델: {model_name}")

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 모델을 디바이스로 이동
model = model.to(device)

print(f"✓ 모델 로드 완료")
print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# ===== 5. 토크나이징 함수 =====
def preprocess_function(examples):
    """대화와 요약을 토크나이징"""
    # 입력 (대화)
    inputs = tokenizer(
        examples['dialogue'],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding='max_length'
    )

    # 타겟 (요약)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['summary'],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding='max_length'
        )

    inputs['labels'] = labels['input_ids']
    return inputs

# ===== 6. Dataset 변환 및 토크나이징 =====
print("\n" + "="*70)
print("데이터셋 토크나이징")
print("="*70)

train_dataset = Dataset.from_pandas(train_data.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_data.reset_index(drop=True))

print("토크나이징 중...")
train_tokenized = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Train"
)

val_tokenized = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Validation"
)

print(f"✓ 토크나이징 완료")

# ===== 7. Training Arguments =====
print("\n" + "="*70)
print("Training 설정")
print("="*70)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_accumulation_steps=2,
    report_to="none",
    disable_tqdm=False,
)

total_steps = len(train_tokenized) // (BATCH_SIZE * 2) * NUM_EPOCHS

print(f"에포크: {NUM_EPOCHS}")
print(f"배치 크기: {BATCH_SIZE}")
print(f"학습률: {LEARNING_RATE}")
print(f"총 스텝 수: {total_steps}")
print(f"예상 시간: 약 {total_steps * 2 / 60:.0f}분 (MPS 기준)")

# ===== 8. Data Collator =====
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# ===== 9. Trainer 생성 =====
print("\n" + "="*70)
print("Trainer 생성")
print("="*70)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("✓ Trainer 준비 완료")

# ===== 10. Fine-tuning 시작 =====
print("\n" + "="*70)
print("Fine-tuning 시작!")
print("="*70)
print("(MPS 사용 시 처음엔 느릴 수 있습니다)")
print()

try:
    trainer.train()
    print("\n" + "="*70)
    print("✓ Fine-tuning 완료!")
    print("="*70)
except Exception as e:
    print(f"\n⚠️ Training 중 오류 발생: {e}")
    print("CPU로 재시도를 권장합니다.")
    raise

# ===== 11. 평가 =====
print("\n" + "="*70)
print("최종 평가")
print("="*70)

eval_results = trainer.evaluate()
print(f"Validation Loss: {eval_results['eval_loss']:.4f}")

# ===== 12. 모델 저장 =====
print("\n모델 저장 중...")
final_output_dir = f"{OUTPUT_DIR}_final"
trainer.save_model(final_output_dir)
tokenizer.save_pretrained(final_output_dir)
print(f"✓ 모델 저장 완료: {final_output_dir}")

# ===== 13. 테스트 생성 =====
print("\n" + "="*70)
print("테스트 생성")
print("="*70)

test_idx = 0
test_dialogue = val_data.iloc[test_idx]['dialogue']
test_summary = val_data.iloc[test_idx]['summary']

# 요약 생성
inputs = tokenizer(test_dialogue, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).to(device)
summary_ids = model.generate(
    inputs['input_ids'],
    max_length=MAX_TARGET_LENGTH,
    min_length=10,
    num_beams=4,
    length_penalty=2.0,
    early_stopping=True
)
pred_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(f"원본 대화 ({len(test_dialogue)}자):")
print(test_dialogue[:200] + "...")
print(f"\n실제 요약 ({len(test_summary)}자):")
print(test_summary)
print(f"\n생성 요약 ({len(pred_summary)}자):")
print(pred_summary)

# ===== 14. 여러 샘플 테스트 =====
print("\n" + "="*70)
print("추가 샘플 테스트 (5개)")
print("="*70)

for i in range(min(5, len(val_data))):
    test_dialogue = val_data.iloc[i]['dialogue']
    test_summary = val_data.iloc[i]['summary']

    inputs = tokenizer(test_dialogue, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).to(device)
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=MAX_TARGET_LENGTH,
        num_beams=4,
        length_penalty=2.0
    )
    pred_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print(f"\n[샘플 {i+1}]")
    print(f"대화: {test_dialogue[:100]}...")
    print(f"실제: {test_summary}")
    print(f"생성: {pred_summary}")
    print(f"길이: 대화 {len(test_dialogue)}자 -> 실제 {len(test_summary)}자 / 생성 {len(pred_summary)}자")

print("\n" + "="*70)
print("완료!")
print("="*70)
print(f"모델 위치: {final_output_dir}")
print(f"체크포인트: {OUTPUT_DIR}")
