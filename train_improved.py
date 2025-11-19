"""
KoBART Fine-tuning - 개선된 버전 (Improved Performance)
- Cosine Learning Rate Scheduler with Warmup
- Label Smoothing
- Optimized Generation Parameters
- Better Hyperparameters
"""

import torch
from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
    get_cosine_schedule_with_warmup
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

# ===== 개선된 설정 =====
EXPERIMENT_NAME = "kobart_improved"  # ✅ 실험 이름
NUM_EPOCHS = 8                # ✅ 5 → 8 에포크 (더 많은 학습)
BATCH_SIZE = 16               # 배치 크기
LEARNING_RATE = 3e-5          # ✅ 5e-5 → 3e-5 (더 안정적인 학습)
MAX_INPUT_LENGTH = 512        # 최대 입력 길이
MAX_TARGET_LENGTH = 128       # 최대 요약 길이
EARLY_STOPPING_PATIENCE = 4   # ✅ 3 → 4 (더 긴 patience)
WARMUP_RATIO = 0.1            # ✅ Warmup 비율 (전체 스텝의 10%)
LABEL_SMOOTHING = 0.1         # ✅ Label Smoothing (과적합 방지)

# 실험별 폴더 구조 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_DIR = f"./outputs/{EXPERIMENT_NAME}_{timestamp}"
OUTPUT_DIR = f"{EXPERIMENT_DIR}/checkpoints"
FINAL_MODEL_DIR = f"{EXPERIMENT_DIR}/models"
LOG_DIR = f"{EXPERIMENT_DIR}/logs"
SUBMISSION_DIR = f"{EXPERIMENT_DIR}/submissions"

# 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

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
logger.info("KoBART Fine-tuning - 개선된 버전 (Improved)")
logger.info("="*70)
logger.info(f"실험 이름: {EXPERIMENT_NAME}")
logger.info(f"실험 디렉토리: {EXPERIMENT_DIR}")
logger.info(f"에포크: {NUM_EPOCHS}")
logger.info(f"배치 크기: {BATCH_SIZE}")
logger.info(f"학습률: {LEARNING_RATE}")
logger.info(f"Warmup 비율: {WARMUP_RATIO}")
logger.info(f"Label Smoothing: {LABEL_SMOOTHING}")
logger.info(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
logger.info("="*70)

# ===== 1. 디바이스 설정 =====
def setup_device():
    """디바이스 설정 (CUDA > MPS > CPU)"""
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

    try:
        x = torch.randn(3, 3).to(device)
        y = x + x
        logger.info(f"✓ 디바이스 연산 테스트 성공")
    except Exception as e:
        logger.warning(f"⚠️ 디바이스 오류: {e}")
        device = torch.device("cpu")
        logger.info("CPU로 전환합니다.")

    return device

device = setup_device()

# ===== 2. 데이터 로드 =====
print("\n" + "="*70)
print("데이터 로드")
print("="*70)

from src.data_loader import load_json_data
from src.postprocessing import fix_summary_punctuation_and_format

# Train 데이터 로드 (15,000개 subset)
train_data = load_json_data('/Users/seoyeonmun/Downloads/korean_data/train_15000/')

# Validation 데이터 로드 (3,000개 subset)
val_data = load_json_data('/Users/seoyeonmun/Downloads/korean_data/val_3000/')

print(f"\nTrain: {len(train_data):,}개")
print(f"Validation: {len(val_data):,}개")

# ===== 3. 모델과 토크나이저 로드 =====
print("\n" + "="*70)
print("모델 로드")
print("="*70)

model_name = "gogamza/kobart-summarization"
print(f"모델: {model_name}")

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 모델을 디바이스로 이동
model = model.to(device)

print(f"✓ 모델 로드 완료")
print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# ✅ 개선된 generation config 설정
model.config.length_penalty = 1.2        # ✅ 1.5 → 1.2 (더 균형잡힌 길이)
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True
model.config.forced_eos_token_id = tokenizer.eos_token_id

# ===== 4. 토크나이징 함수 =====
def preprocess_function(examples):
    """대화와 요약을 토크나이징"""
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

# ===== 5. ROUGE 평가 함수 =====
def compute_metrics(eval_pred):
    """ROUGE 점수 계산 (대회 평가 기준)"""
    predictions, labels = eval_pred

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 후처리
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

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,           # ✅ Warmup 비율 사용
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,                   # ✅ 2 → 3 (더 많은 체크포인트 보관)
    load_best_model_at_end=True,
    metric_for_best_model="rouge_score",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=128,            # ✅ 120 → 128 (더 긴 요약 허용)
    generation_num_beams=6,               # ✅ 5 → 6 (더 많은 빔 탐색)
    gradient_accumulation_steps=2,
    label_smoothing_factor=LABEL_SMOOTHING,  # ✅ Label Smoothing 추가
    lr_scheduler_type="cosine",           # ✅ Cosine Learning Rate Scheduler
    report_to="none",
    disable_tqdm=False,
    fp16=torch.cuda.is_available()
)

total_steps = len(train_tokenized) // (BATCH_SIZE * 2) * NUM_EPOCHS

print(f"에포크: {NUM_EPOCHS}")
print(f"배치 크기: {BATCH_SIZE}")
print(f"학습률: {LEARNING_RATE}")
print(f"총 스텝 수: {total_steps}")
print(f"Warmup 스텝: {int(total_steps * WARMUP_RATIO)}")
print(f"예상 시간: 약 {total_steps * 3 / 60:.0f}분 (GPU 기준)")

# ===== 8. Data Collator =====
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# ===== 9. Custom Logging Callback =====
class RoundedLoggingCallback(TrainerCallback):
    """로그 값을 소수점 4자리로 반올림"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, float):
                    logs[key] = round(value, 4)

# ===== 10. Trainer 생성 =====
print("\n" + "="*70)
print("Trainer 생성")
print("="*70)

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

print("✓ Trainer 준비 완료")

# ===== 11. Fine-tuning 시작 =====
print("\n" + "="*70)
print("Fine-tuning 시작!")
print("="*70)
print("개선 사항:")
print("- Cosine Learning Rate Scheduler with Warmup")
print("- Label Smoothing (0.1)")
print("- 더 많은 에포크 (8)")
print("- 더 안정적인 학습률 (3e-5)")
print("- 더 많은 빔 탐색 (6)")
print()

try:
    trainer.train()
    print("\n" + "="*70)
    print("✓ Fine-tuning 완료!")
    print("="*70)
except Exception as e:
    print(f"\n⚠️ Training 중 오류 발생: {e}")
    raise

# ===== 12. 평가 =====
print("\n" + "="*70)
print("최종 평가")
print("="*70)

eval_results = trainer.evaluate()
print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
print(f"ROUGE-1 F1: {eval_results['eval_rouge1']:.4f}")
print(f"ROUGE-2 F1: {eval_results['eval_rouge2']:.4f}")
print(f"ROUGE-L F1: {eval_results['eval_rougeL']:.4f}")
print(f"최종 ROUGE Score: {eval_results['eval_rouge_score']:.4f}")

# ===== 13. 모델 저장 =====
print("\n모델 저장 중...")
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"✓ 모델 저장 완료: {FINAL_MODEL_DIR}")

# ===== 14. 테스트 생성 (개선된 파라미터) =====
print("\n" + "="*70)
print("테스트 생성")
print("="*70)

test_idx = 0
test_dialogue = val_data.iloc[test_idx]['dialogue']
test_summary = val_data.iloc[test_idx]['summary']

# ✅ 개선된 생성 파라미터
inputs = tokenizer(test_dialogue, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).to(device)
summary_ids = model.generate(
    inputs['input_ids'],
    max_length=128,                    # ✅ 120 → 128
    min_length=20,                     # ✅ 15 → 20 (더 긴 최소 길이)
    num_beams=8,                       # ✅ 6 → 8 (더 많은 빔)
    length_penalty=1.2,                # ✅ 균형잡힌 길이 페널티
    no_repeat_ngram_size=3,
    early_stopping=True,
    repetition_penalty=1.2,            # ✅ 1.3 → 1.2 (덜 공격적)
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    forced_eos_token_id=tokenizer.eos_token_id
)
pred_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
pred_summary = fix_summary_punctuation_and_format(pred_summary)

print(f"원본 대화 ({len(test_dialogue)}자):")
print(test_dialogue[:200] + "...")
print(f"\n실제 요약 ({len(test_summary)}자):")
print(test_summary)
print(f"\n생성 요약 ({len(pred_summary)}자):")
print(pred_summary)

# ===== 15. 여러 샘플 테스트 =====
print("\n" + "="*70)
print("추가 샘플 테스트 (5개)")
print("="*70)

for i in range(min(5, len(val_data))):
    test_dialogue = val_data.iloc[i]['dialogue']
    test_summary = val_data.iloc[i]['summary']

    inputs = tokenizer(test_dialogue, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).to(device)
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=128,
        min_length=20,
        num_beams=6,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
        repetition_penalty=1.2
    )
    pred_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    pred_summary = fix_summary_punctuation_and_format(pred_summary)

    print(f"\n[샘플 {i+1}]")
    print(f"대화 ({len(test_dialogue)}자): {test_dialogue[:100]}...")
    print(f"실제: {test_summary}")
    print(f"생성: {pred_summary}")

print("\n" + "="*70)
print("완료!")
print("="*70)
print(f"실험 디렉토리: {EXPERIMENT_DIR}")
print(f"최종 모델: {FINAL_MODEL_DIR}")
print(f"체크포인트: {OUTPUT_DIR}")
print(f"로그: {LOG_DIR}")
print("\n개선 사항 요약:")
print("✅ Cosine Learning Rate Scheduler with Warmup")
print("✅ Label Smoothing (0.1)")
print("✅ 에포크 증가 (5 → 8)")
print("✅ 학습률 조정 (5e-5 → 3e-5)")
print("✅ 빔 탐색 증가 (5 → 6)")
print("✅ 더 긴 요약 허용 (120 → 128)")
print("="*70)
