"""
train_bart_r3f.py - 대화 요약 특화 모델
"""
import torch
from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback
)
from datasets import Dataset
import os
import warnings
import numpy as np
from rouge_score import rouge_scorer
from datetime import datetime
import sys
import logging

warnings.filterwarnings('ignore')

# ===== 설정 =====
EXPERIMENT_NAME = "bart_r3f"  # ✅ 실험 이름
NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
EARLY_STOPPING_PATIENCE = 3

# 실험별 폴더 구조
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_DIR = f"./outputs/{EXPERIMENT_NAME}_{timestamp}"
OUTPUT_DIR = f"{EXPERIMENT_DIR}/checkpoints"
FINAL_MODEL_DIR = f"{EXPERIMENT_DIR}/models"
LOG_DIR = f"{EXPERIMENT_DIR}/logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ===== 로깅 설정 =====
log_file = f"{LOG_DIR}/training.log"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("="*70)
logger.info("BART-R3F Fine-tuning - 대화 요약 특화")
logger.info("="*70)
logger.info(f"실험 이름: {EXPERIMENT_NAME}")
logger.info(f"실험 디렉토리: {EXPERIMENT_DIR}")
logger.info("="*70)

# ===== 디바이스 설정 =====
def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "CUDA GPU"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS (Apple Silicon GPU)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    logger.info(f"\n✓ 디바이스: {device_name}")
    return device

device = setup_device()

# ===== 데이터 로드 =====
print("\n" + "="*70)
print("데이터 로드")
print("="*70)

from src.data_loader import load_json_data
from src.postprocessing import fix_summary_punctuation_and_format

train_data = load_json_data('./data/sample/train_sample/')
val_data = load_json_data('./data/sample/val_sample/')

print(f"Train: {len(train_data):,}개")
print(f"Validation: {len(val_data):,}개")

# ===== 모델과 토크나이저 로드 =====
print("\n" + "="*70)
print("모델 로드")
print("="*70)

model_name = "alaggung/bart-r3f"  # ✅ 대화 요약 특화!
print(f"모델: {model_name}")

try:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    print(f"✓ 모델 로드 완료")
except Exception as e:
    print(f"⚠️ 오류: {e}")
    print("대체 토크나이저 시도...")
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    print(f"✓ 모델 로드 완료 (BartTokenizer)")

model = model.to(device)
print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# ===== 토크나이징 함수 =====
def preprocess_function(examples):
    inputs = tokenizer(
        examples['dialogue'],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding='max_length'
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['summary'],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding='max_length'
        )

    inputs['labels'] = labels['input_ids']
    return inputs

# ===== ROUGE 평가 함수 =====
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [fix_summary_punctuation_and_format(pred) for pred in decoded_preds]
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(label, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    rouge1_avg = np.mean(rouge1_scores)
    rouge2_avg = np.mean(rouge2_scores)
    rougeL_avg = np.mean(rougeL_scores)
    final_score = rouge1_avg + rouge2_avg + rougeL_avg
    
    return {
        'rouge1': rouge1_avg,
        'rouge2': rouge2_avg,
        'rougeL': rougeL_avg,
        'rouge_score': final_score
    }

# ===== Dataset 변환 =====
print("\n" + "="*70)
print("데이터셋 토크나이징")
print("="*70)

train_dataset = Dataset.from_pandas(train_data.reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_data.reset_index(drop=True))

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

# ===== Training Arguments =====
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rouge_score",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=120,
    generation_num_beams=5,
    gradient_accumulation_steps=2,
    report_to="none",
    disable_tqdm=False,
    fp16=torch.cuda.is_available()
)

# ===== Data Collator =====
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# ===== Callback =====
class RoundedLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, float):
                    logs[key] = round(value, 4)

# ===== Trainer =====
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
        RoundedLoggingCallback()
    ]
)

# ===== Fine-tuning =====
print("\n" + "="*70)
print("Fine-tuning 시작!")
print("="*70)

try:
    trainer.train()
    print("\n✓ Fine-tuning 완료!")
except Exception as e:
    print(f"\n⚠️ Training 중 오류: {e}")
    raise

# ===== 평가 =====
eval_results = trainer.evaluate()
print(f"\n최종 ROUGE Score: {eval_results['eval_rouge_score']:.4f}")

# ===== 모델 저장 =====
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"✓ 모델 저장: {FINAL_MODEL_DIR}")

# ===== 비교 =====
print("\n" + "="*70)
print("비교")
print("="*70)
print(f"kobart-summarization: 0.2423")
print(f"bart-r3f (현재):      {eval_results['eval_rouge_score']:.4f}")
print(f"차이:                 {eval_results['eval_rouge_score'] - 0.2423:+.4f}")

if eval_results['eval_rouge_score'] > 0.2423:
    print("\n🎉 bart-r3f가 더 좋습니다!")
else:
    print("\n⚠️ kobart-summarization이 더 좋습니다.")