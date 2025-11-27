"""
KoBART Fine-tuning - 최종 최적화 버전
MPS/CUDA/CPU 지원
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
import pandas as pd
from rouge_score import rouge_scorer
from datetime import datetime
import sys
import logging
from src.preprocessing import preprocess_text
from src.postprocessing import fix_summary_punctuation_and_format

warnings.filterwarnings('ignore')

# ===== 설정 =====
MODEL_NAME = "gogamza/kobart-summarization"  # 사용 모델

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = f"{MODEL_NAME.replace('/', '_')}_{timestamp}"
NUM_EPOCHS = 10              # 에포크 수
BATCH_SIZE = 16             # 배치 크기 (RTX 3090 최적화)
LEARNING_RATE = 5e-5        # 학습률
MAX_INPUT_LENGTH = 512      # 최대 입력 길이
MAX_TARGET_LENGTH = 128     # 최대 요약 길이
EARLY_STOPPING_PATIENCE = 3 # Early stopping patience (에포크 단위)
USE_CACHE = True            # 캐싱 사용 여부
USE_WANDB = True            # Weights & Biases 로깅 사용 여부
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "kobart-summarization")
WANDB_ENTITY = os.getenv("WANDB_ENTITY") or None

# 실험별 폴더 구조 생성
EXPERIMENT_DIR = f"./outputs/{EXPERIMENT_NAME}"
OUTPUT_DIR = f"{EXPERIMENT_DIR}/checkpoints"
FINAL_MODEL_DIR = f"{EXPERIMENT_DIR}/models"
LOG_DIR = f"{EXPERIMENT_DIR}/logs"
SUBMISSION_DIR = f"{EXPERIMENT_DIR}/submissions"

# 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

WANDB_ACTIVE = False

# 동적 길이 설정 (EDA 분석 기반)
USE_DYNAMIC_LENGTH = True   # 동적 길이 사용 여부
COMPRESSION_RATIO = 0.24    # 토큰 기준 압축 비율 (EDA 분석 기반)
MIN_LENGTH_ABSOLUTE = 10    # 절대 최소 길이 (토큰)
MAX_LENGTH_ABSOLUTE = 128   # 절대 최대 길이 (토큰)

# ===== 로깅 설정 =====
log_file = f"{LOG_DIR}/training.log"

# 로거 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 파일 핸들러 (실시간 저장)
file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 콘솔 핸들러
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)

# 핸들러 추가
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("="*70)
logger.info("KoBART Fine-tuning - 최종 최적화 버전")
logger.info("="*70)
logger.info(f"실험 이름: {EXPERIMENT_NAME}")
logger.info(f"모델: {MODEL_NAME}")
logger.info(f"실험 디렉토리: {EXPERIMENT_DIR}")
logger.info(f"에포크: {NUM_EPOCHS}")
logger.info(f"배치 크기: {BATCH_SIZE}")
logger.info(f"캐싱 사용: {USE_CACHE}")
logger.info(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
logger.info(f"로그 파일: {log_file}")
logger.info("="*70)

if USE_WANDB:
    try:
        import wandb

        wandb_config = {
            "model_name": MODEL_NAME,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_input_length": MAX_INPUT_LENGTH,
            "max_target_length": MAX_TARGET_LENGTH,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "use_dynamic_length": USE_DYNAMIC_LENGTH,
            "compression_ratio": COMPRESSION_RATIO,
        }

        wandb_run_id = EXPERIMENT_NAME

        wandb_kwargs = {
            "project": WANDB_PROJECT,
            "name": EXPERIMENT_NAME,
            "config": wandb_config,
            "dir": EXPERIMENT_DIR,
            "id": wandb_run_id,
        }
        if WANDB_ENTITY:
            wandb_kwargs["entity"] = WANDB_ENTITY

        wandb.init(**wandb_kwargs)
        WANDB_ACTIVE = True
        logger.info("W&B 로깅 활성화")
    except Exception as e:
        WANDB_ACTIVE = False
        logger.warning(f"⚠️ W&B 초기화 실패: {e}")
else:
    logger.info("W&B 로깅 비활성화")

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

    # 디바이스 테스트
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

# ===== 2. 동적 길이 계산 함수 (토큰 기반) =====
def calculate_dynamic_length(input_text, tokenizer, compression_ratio=COMPRESSION_RATIO):
    """
    입력 텍스트의 토큰 길이에 기반하여 동적으로 요약 길이 계산

    Args:
        input_text: 입력 대화 텍스트
        tokenizer: 토크나이저
        compression_ratio: 압축 비율 (기본값: 0.24)

    Returns:
        (min_length, max_length) 튜플 (토큰 단위)
    """
    # 입력 텍스트를 토큰화하여 실제 토큰 길이 계산
    input_tokens = tokenizer.encode(input_text, add_special_tokens=False)
    input_token_length = len(input_tokens)

    # 목표 길이 = 입력 토큰 길이 * 압축 비율
    target_length = int(input_token_length * compression_ratio)

    # min_length: 목표 길이의 70%
    min_length = int(target_length * 0.7)

    # max_length: 목표 길이의 200% (✅ 1.8 → 2.0)
    max_length = int(target_length * 2.0)

    # 절대 최소/최대 길이로 제한
    min_length = max(10, min(min_length, MAX_LENGTH_ABSOLUTE))
    max_length = max(min_length + 20, min(max_length, MAX_LENGTH_ABSOLUTE))  # ✅ +15 → +20

    return min_length, max_length

# ===== 3. 데이터 로드 =====
print("\n" + "="*70)
print("데이터 로드")
print("="*70)


def load_csv_dataset(csv_path):
    """
    CSV 형식의 대화 데이터 로드 및 전처리
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {'dialogue', 'summary'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing_cols}")

    # 결측치 제거 후 전처리 적용
    df = df.dropna(subset=['dialogue', 'summary']).copy()
    df['dialogue'] = df['dialogue'].astype(str).apply(preprocess_text)
    df['summary'] = df['summary'].astype(str).apply(preprocess_text)

    return df[['dialogue', 'summary']]


def load_test_csv_dataset(csv_path):
    """
    테스트 CSV 데이터를 로드하고 전처리
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"테스트 데이터 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {'fname', 'dialogue'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"테스트 CSV에 필요한 컬럼이 없습니다: {missing_cols}")

    df = df.dropna(subset=['dialogue']).copy()
    df['fname'] = df['fname'].astype(str)
    df['dialogue'] = df['dialogue'].astype(str).apply(preprocess_text)

    return df[['fname', 'dialogue']]


def validate_submission_format(submission_df, sample_submission_path):
    """
    생성된 제출 파일이 샘플 제출 형식과 일치하는지 검증
    """
    print("\n제출 형식 검증 중...")

    required_cols = ['fname', 'summary']

    missing_submission_cols = [col for col in required_cols if col not in submission_df.columns]
    if missing_submission_cols:
        raise ValueError(f"제출 데이터프레임에 필요한 컬럼이 없습니다: {missing_submission_cols}")

    submission_df = submission_df[required_cols].copy()

    if not os.path.exists(sample_submission_path):
        print(f"⚠️ 샘플 제출 파일을 찾을 수 없습니다: {sample_submission_path}")
        return submission_df

    sample_df = pd.read_csv(sample_submission_path)
    sample_cols = [col for col in sample_df.columns if col in required_cols]
    missing_sample_cols = [col for col in required_cols if col not in sample_cols]

    if missing_sample_cols:
        print(f"⚠️ 샘플 제출 파일에 필요한 컬럼이 없습니다: {missing_sample_cols}")
    else:
        sample_df = sample_df[sample_cols].copy()

    sample_len = len(sample_df)
    submission_len = len(submission_df)
    if sample_len == submission_len:
        print(f"✓ 행 수 일치: {submission_len:,}개")
    else:
        print(f"⚠️ 행 수 불일치: sample={sample_len:,}, submission={submission_len:,}")

    if 'fname' in sample_df.columns:
        sample_fnames = set(sample_df['fname'].astype(str))
        submission_fnames = set(submission_df['fname'].astype(str))

        missing_in_submission = sample_fnames - submission_fnames
        extra_in_submission = submission_fnames - sample_fnames

        if not missing_in_submission and not extra_in_submission:
            print("✓ fname 목록 일치")
        else:
            if missing_in_submission:
                preview = sorted(list(missing_in_submission))[:5]
                suffix = " ..." if len(missing_in_submission) > 5 else ""
                print(f"⚠️ 누락된 fname: {preview}{suffix}")
            if extra_in_submission:
                preview = sorted(list(extra_in_submission))[:5]
                suffix = " ..." if len(extra_in_submission) > 5 else ""
                print(f"⚠️ 추가된 fname: {preview}{suffix}")

    return submission_df

# Train/Validation 데이터 로드 (CSV)
train_csv_path = './data/train.csv'
val_csv_path = './data/dev.csv'

print(f"Train CSV 경로: {train_csv_path}")
print(f"Validation CSV 경로: {val_csv_path}")

train_data = load_csv_dataset(train_csv_path)
val_data = load_csv_dataset(val_csv_path)

# ===== 4. 데이터 확인 =====
print("\n데이터 확인")

print(f"Train: {len(train_data):,}개")
print(f"Validation: {len(val_data):,}개")

print(f"\n샘플 확인:")
print(f"대화: {train_data.iloc[0]['dialogue'][:100]}...")
print(f"요약: {train_data.iloc[0]['summary']}")

# ===== 5. 모델과 토크나이저 로드 =====
print("\n" + "="*70)
print("모델 로드")
print("="*70)

print(f"모델: {MODEL_NAME}")

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# 모델을 디바이스로 이동
model = model.to(device)

print(f"✓ 모델 로드 완료")
print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# 모델의 generation config 설정
model.config.length_penalty = 1.5
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True
model.config.forced_eos_token_id = tokenizer.eos_token_id

# ===== 6. 토크나이징 함수 =====
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


def preprocess_test_function(examples):
    """테스트 데이터를 위한 토크나이징"""
    return tokenizer(
        examples['dialogue'],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding='max_length'
    )

# ===== 7. ROUGE 평가 함수 =====
def compute_metrics(eval_pred):
    """ROUGE 점수 계산 (대회 평가 기준)"""
    predictions, labels = eval_pred

    # -100을 pad token으로 변경
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # 디코딩
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 후처리: 불완전한 문장 제거
    decoded_preds = [fix_summary_punctuation_and_format(pred) for pred in decoded_preds]

    # ROUGE 계산
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(label, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    # 평균 계산
    rouge1_avg = np.mean(rouge1_scores)
    rouge2_avg = np.mean(rouge2_scores)
    rougeL_avg = np.mean(rougeL_scores)

    final_score = rouge1_avg + rouge2_avg + rougeL_avg

    return {
        'rouge1': rouge1_avg,
        'rouge2': rouge2_avg,
        'rougeL': rougeL_avg,
        'rouge_score': final_score  # 대회 최종 점수
    }

# ===== 8. Dataset 변환 및 토크나이징 =====
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

# ===== 9. Training Arguments =====
print("\n" + "="*70)
print("Training 설정")
print("="*70)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    run_name=EXPERIMENT_NAME,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="rouge_score",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=120,  # ✅ 80 → 120
    generation_num_beams=5,
    gradient_accumulation_steps=2,
    report_to="wandb" if WANDB_ACTIVE else "none",
    disable_tqdm=False,
    fp16=torch.cuda.is_available()  # CUDA 사용시 FP16
)

total_steps = len(train_tokenized) // (BATCH_SIZE * 2) * NUM_EPOCHS

print(f"에포크: {NUM_EPOCHS}")
print(f"배치 크기: {BATCH_SIZE}")
print(f"학습률: {LEARNING_RATE}")
print(f"총 스텝 수: {total_steps}")
print(f"예상 시간: 약 {total_steps * 3 / 60:.0f}분 (GPU 기준)")

# ===== 10. Data Collator =====
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# ===== 10-1. Custom Logging Callback (소수점 4자리) =====
class RoundedLoggingCallback(TrainerCallback):
    """로그 값을 소수점 4자리로 반올림"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, float):
                    logs[key] = round(value, 4)

# ===== 11. Trainer 생성 =====
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
        RoundedLoggingCallback()  # 소수점 4자리 반올림
    ]
)

print("✓ Trainer 준비 완료")

# ===== 12. Fine-tuning 시작 =====
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

# ===== 13. 평가 =====
print("\n" + "="*70)
print("최종 평가")
print("="*70)

eval_results = trainer.evaluate()
print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
print(f"ROUGE-1 F1: {eval_results['eval_rouge1']:.4f}")
print(f"ROUGE-2 F1: {eval_results['eval_rouge2']:.4f}")
print(f"ROUGE-L F1: {eval_results['eval_rougeL']:.4f}")
print(f"최종 ROUGE Score: {eval_results['eval_rouge_score']:.4f}")

# ===== 14. 모델 저장 =====
print("\n모델 저장 중...")
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"✓ 모델 저장 완료: {FINAL_MODEL_DIR}")

# ===== 15. 제출 파일 생성 =====
print("\n" + "="*70)
print("제출 파일 생성")
print("="*70)

test_csv_path = './data/test.csv'
print(f"Test CSV 경로: {test_csv_path}")

test_data = load_test_csv_dataset(test_csv_path)
print(f"Test: {len(test_data):,}개")

test_dataset = Dataset.from_pandas(test_data.reset_index(drop=True))

print("테스트 데이터 토크나이징 중...")
test_tokenized = test_dataset.map(
    preprocess_test_function,
    batched=True,
    remove_columns=['dialogue'],
    desc="Test"
)

print("생성 중...")
test_predictions = trainer.predict(
    test_tokenized,
    max_length=120,
    num_beams=5
)

generated_ids = test_predictions.predictions
if isinstance(generated_ids, tuple):
    generated_ids = generated_ids[0]

decoded_test_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
decoded_test_preds = [
    fix_summary_punctuation_and_format(pred) for pred in decoded_test_preds
]

submission_df = pd.DataFrame({
    'fname': test_data['fname'],
    'summary': decoded_test_preds
})

sample_submission_path = './data/sample_submission.csv'
submission_df = validate_submission_format(submission_df, sample_submission_path)

submission_path = f"{SUBMISSION_DIR}/submission_{timestamp}.csv"
submission_df.to_csv(submission_path, index=False)

print(f"✓ 제출 파일 저장 완료: {submission_path}")

# ===== 16. 검증 샘플 생성 =====
print("\n" + "="*70)
print("검증 샘플 생성")
print("="*70)

test_idx = 0
test_dialogue = val_data.iloc[test_idx]['dialogue']
test_summary = val_data.iloc[test_idx]['summary']

# 동적 길이 계산
if USE_DYNAMIC_LENGTH:
    min_len, max_len = calculate_dynamic_length(test_dialogue, tokenizer, COMPRESSION_RATIO)
else:
    min_len = MIN_LENGTH_ABSOLUTE
    max_len = MAX_LENGTH_ABSOLUTE

# 요약 생성
inputs = tokenizer(test_dialogue, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).to(device)
summary_ids = model.generate(
    inputs['input_ids'],
    max_length=120,  # ✅ 80 → 120
    min_length=15,
    num_beams=6,
    length_penalty=1.5,
    no_repeat_ngram_size=4,
    early_stopping=True,
    repetition_penalty=1.3,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    forced_eos_token_id=tokenizer.eos_token_id
)
pred_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 후처리: 불완전한 문장 제거
pred_summary = fix_summary_punctuation_and_format(pred_summary)

print(f"원본 대화 ({len(test_dialogue)}자):")
print(test_dialogue[:200] + "...")
if USE_DYNAMIC_LENGTH:
    input_tokens = len(tokenizer.encode(test_dialogue, add_special_tokens=False))
    print(f"동적 길이 설정: min={min_len} tokens, max={max_len} tokens")
    print(f"입력 토큰 수: {input_tokens} tokens (압축률 목표: {COMPRESSION_RATIO*100:.4f}%)")
print(f"\n실제 요약 ({len(test_summary)}자):")
print(test_summary)
print(f"\n생성 요약 ({len(pred_summary)}자):")
print(pred_summary)
pred_tokens = len(tokenizer.encode(pred_summary, add_special_tokens=False))
if USE_DYNAMIC_LENGTH:
    input_tokens = len(tokenizer.encode(test_dialogue, add_special_tokens=False))
    print(f"생성 토큰 수: {pred_tokens} tokens (실제 압축률: {pred_tokens/input_tokens*100:.4f}%)")

# ===== 17. 여러 검증 샘플 테스트 =====
print("\n" + "="*70)
print("추가 검증 샘플 테스트 (5개)")
print("="*70)

for i in range(min(5, len(val_data))):
    test_dialogue = val_data.iloc[i]['dialogue']
    test_summary = val_data.iloc[i]['summary']

    # 동적 길이 계산
    if USE_DYNAMIC_LENGTH:
        min_len, max_len = calculate_dynamic_length(test_dialogue, tokenizer, COMPRESSION_RATIO)
    else:
        min_len = MIN_LENGTH_ABSOLUTE
        max_len = MAX_LENGTH_ABSOLUTE

    inputs = tokenizer(test_dialogue, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).to(device)
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=120,  # ✅ max_len → 120 (고정)
        min_length=15,
        num_beams=5,
        length_penalty=1.5,
        no_repeat_ngram_size=4,
        early_stopping=True,
        repetition_penalty=1.3
    )
    pred_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # 후처리: 불완전한 문장 제거
    pred_summary = fix_summary_punctuation_and_format(pred_summary)

    print(f"\n[샘플 {i+1}]")
    print(f"대화 ({len(test_dialogue)}자): {test_dialogue[:100]}...")
    if USE_DYNAMIC_LENGTH:
        input_tokens = len(tokenizer.encode(test_dialogue, add_special_tokens=False))
        pred_tokens = len(tokenizer.encode(pred_summary, add_special_tokens=False))
        print(f"동적 길이: min={min_len}, max={max_len} tokens")
        print(f"입력 토큰: {input_tokens}, 생성 토큰: {pred_tokens} (압축률: {pred_tokens/input_tokens*100:.4f}%)")
    print(f"실제: {test_summary}")
    print(f"생성: {pred_summary}")

print("\n" + "="*70)
print("완료!")
print("="*70)
print(f"실험 디렉토리: {EXPERIMENT_DIR}")
print(f"최종 모델: {FINAL_MODEL_DIR}")
print(f"체크포인트: {OUTPUT_DIR}")
print(f"로그: {LOG_DIR}")
print(f"제출 파일: {SUBMISSION_DIR}")

if WANDB_ACTIVE:
    wandb.finish()