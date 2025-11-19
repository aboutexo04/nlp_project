"""
KoBART Fine-tuning - Google Colab 버전
- GPU 최적화 (T4/V100/A100)
- Cosine Learning Rate Scheduler with Warmup
- Label Smoothing
- Optimized Generation Parameters
- Google Drive 연동 지원

사용 방법:
1. Colab에서 런타임 > 런타임 유형 변경 > GPU 선택
2. 데이터를 Google Drive에 업로드
3. 이 스크립트 실행
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

# ===== Google Drive 마운트 (선택사항) =====
USE_DRIVE = True  # Google Drive 사용 여부

if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive 마운트 완료")
        # 데이터 경로를 Google Drive로 설정
        DATA_BASE_PATH = '/content/drive/MyDrive/korean_data'
    except:
        print("⚠️ Google Drive 마운트 실패 또는 Colab이 아닙니다.")
        USE_DRIVE = False
        DATA_BASE_PATH = './korean_data'
else:
    DATA_BASE_PATH = './korean_data'

# ===== 개선된 설정 =====
EXPERIMENT_NAME = "kobart_colab"   # 실험 이름
NUM_EPOCHS = 10                    # ✅ Colab GPU로 더 많은 에포크
BATCH_SIZE = 32                    # ✅ GPU 메모리에 따라 조정 (T4: 16, V100: 32, A100: 64)
LEARNING_RATE = 3e-5               # 학습률
MAX_INPUT_LENGTH = 512             # 최대 입력 길이
MAX_TARGET_LENGTH = 128            # 최대 요약 길이
EARLY_STOPPING_PATIENCE = 4        # Early stopping patience
WARMUP_RATIO = 0.1                 # Warmup 비율
LABEL_SMOOTHING = 0.1              # Label Smoothing
GRADIENT_ACCUMULATION_STEPS = 2    # Gradient Accumulation

# 실험별 폴더 구조 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_DIR = f"./outputs/{EXPERIMENT_NAME}_{timestamp}"
OUTPUT_DIR = f"{EXPERIMENT_DIR}/checkpoints"
FINAL_MODEL_DIR = f"{EXPERIMENT_DIR}/models"
LOG_DIR = f"{EXPERIMENT_DIR}/logs"

# 폴더 생성
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
logger.info("KoBART Fine-tuning - Google Colab 버전")
logger.info("="*70)
logger.info(f"실험 이름: {EXPERIMENT_NAME}")
logger.info(f"실험 디렉토리: {EXPERIMENT_DIR}")
logger.info(f"에포크: {NUM_EPOCHS}")
logger.info(f"배치 크기: {BATCH_SIZE}")
logger.info(f"학습률: {LEARNING_RATE}")
logger.info(f"Warmup 비율: {WARMUP_RATIO}")
logger.info(f"Label Smoothing: {LABEL_SMOOTHING}")
logger.info(f"데이터 경로: {DATA_BASE_PATH}")
logger.info("="*70)

# ===== 1. 디바이스 설정 =====
def setup_device():
    """디바이스 설정 (CUDA > CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
        logger.info(f"\n✓ 디바이스: {device_name}")
        logger.info(f"✓ GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        logger.info(f"\n⚠️ GPU를 사용할 수 없습니다. CPU로 실행합니다.")
        logger.info("Colab에서 런타임 > 런타임 유형 변경 > GPU를 선택하세요.")

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

# ===== 2. 필요한 패키지 설치 (Colab용) =====
print("\n" + "="*70)
print("패키지 확인 및 설치")
print("="*70)

try:
    import konlpy
    print("✓ konlpy 설치됨")
except:
    print("konlpy 설치 중...")
    os.system("pip install konlpy -q")

# ===== 3. 데이터 로드 함수 =====
def load_json_data(data_dir):
    """JSON 데이터 로드"""
    import json
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm
    
    data_path = Path(data_dir)
    json_files = list(data_path.glob('*.json'))
    
    print(f"JSON 파일 {len(json_files)}개 발견")
    
    all_data = []
    for json_file in tqdm(json_files, desc="파일 로딩"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data.get('data', []):
            dialogue_info = item['header']['dialogueInfo']
            
            # 대화 텍스트 구성
            utterances = []
            for utterance in item['body']:
                speaker = utterance['participantID']
                text = utterance['utterance']
                utterances.append(f"{speaker}: {text}")
            
            dialogue_text = " ".join(utterances)
            summary_text = dialogue_info.get('summary', '')
            
            if dialogue_text and summary_text:
                all_data.append({
                    'dialogue': dialogue_text,
                    'summary': summary_text
                })
    
    df = pd.DataFrame(all_data)
    print(f"✓ 로드 완료: {len(df):,}개")
    return df

# ===== 4. 후처리 함수 =====
def fix_summary_punctuation_and_format(text):
    """요약문 후처리"""
    import re
    
    # 기본 정리
    text = text.strip()
    
    # 불완전한 문장 제거 (마지막 문장이 온점/물음표/느낌표로 끝나지 않으면 제거)
    sentences = re.split(r'([.!?])', text)
    complete_text = ""
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            complete_text += sentences[i] + sentences[i+1]
    
    # 마지막 문장이 완전하면 추가
    if len(sentences) % 2 == 1 and len(sentences) > 1:
        last_sentence = sentences[-1].strip()
        if last_sentence and last_sentence[-1] in '.!?':
            complete_text += last_sentence
    
    return complete_text.strip()

# ===== 5. 데이터 로드 =====
print("\n" + "="*70)
print("데이터 로드")
print("="*70)

# Train 데이터 로드
train_data = load_json_data(f'{DATA_BASE_PATH}/train_15000/')

# Validation 데이터 로드
val_data = load_json_data(f'{DATA_BASE_PATH}/val_3000/')

print(f"\nTrain: {len(train_data):,}개")
print(f"Validation: {len(val_data):,}개")

# ===== 6. 모델과 토크나이저 로드 =====
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
model.config.length_penalty = 1.2
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True
model.config.forced_eos_token_id = tokenizer.eos_token_id

# ===== 7. 토크나이징 함수 =====
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

# ===== 8. ROUGE 평가 함수 =====
def compute_metrics(eval_pred):
    """ROUGE 점수 계산"""
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

# ===== 9. Dataset 변환 및 토크나이징 =====
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

# ===== 10. Training Arguments =====
print("\n" + "="*70)
print("Training 설정")
print("="*70)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="rouge_score",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=6,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    label_smoothing_factor=LABEL_SMOOTHING,
    lr_scheduler_type="cosine",
    report_to="none",
    disable_tqdm=False,
    fp16=torch.cuda.is_available(),  # ✅ GPU 사용시 FP16 자동 활성화
    dataloader_num_workers=2,        # ✅ Colab에서 데이터 로딩 속도 향상
)

total_steps = len(train_tokenized) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS

print(f"에포크: {NUM_EPOCHS}")
print(f"배치 크기: {BATCH_SIZE}")
print(f"학습률: {LEARNING_RATE}")
print(f"총 스텝 수: {total_steps}")
print(f"Warmup 스텝: {int(total_steps * WARMUP_RATIO)}")
print(f"FP16: {torch.cuda.is_available()}")

# ===== 11. Data Collator =====
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

# ===== 12. Custom Logging Callback =====
class RoundedLoggingCallback(TrainerCallback):
    """로그 값을 소수점 4자리로 반올림"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, float):
                    logs[key] = round(value, 4)

# ===== 13. Trainer 생성 =====
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

# ===== 14. Fine-tuning 시작 =====
print("\n" + "="*70)
print("Fine-tuning 시작!")
print("="*70)
print("개선 사항:")
print("- Cosine Learning Rate Scheduler with Warmup")
print("- Label Smoothing (0.1)")
print("- FP16 Mixed Precision Training (GPU)")
print("- 더 많은 에포크 (10)")
print("- GPU 최적화된 배치 크기")
print()

try:
    trainer.train()
    print("\n" + "="*70)
    print("✓ Fine-tuning 완료!")
    print("="*70)
except Exception as e:
    print(f"\n⚠️ Training 중 오류 발생: {e}")
    raise

# ===== 15. 평가 =====
print("\n" + "="*70)
print("최종 평가")
print("="*70)

eval_results = trainer.evaluate()
print(f"Validation Loss: {eval_results['eval_loss']:.4f}")
print(f"ROUGE-1 F1: {eval_results['eval_rouge1']:.4f}")
print(f"ROUGE-2 F1: {eval_results['eval_rouge2']:.4f}")
print(f"ROUGE-L F1: {eval_results['eval_rougeL']:.4f}")
print(f"최종 ROUGE Score: {eval_results['eval_rouge_score']:.4f}")

# ===== 16. 모델 저장 =====
print("\n모델 저장 중...")
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"✓ 모델 저장 완료: {FINAL_MODEL_DIR}")

# Google Drive에도 저장 (선택사항)
if USE_DRIVE:
    drive_model_dir = f'/content/drive/MyDrive/kobart_models/{EXPERIMENT_NAME}_{timestamp}'
    os.makedirs(drive_model_dir, exist_ok=True)
    trainer.save_model(drive_model_dir)
    tokenizer.save_pretrained(drive_model_dir)
    print(f"✓ Google Drive에도 저장: {drive_model_dir}")

# ===== 17. 테스트 생성 =====
print("\n" + "="*70)
print("테스트 생성 (5개 샘플)")
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
if USE_DRIVE:
    print(f"Google Drive 모델: {drive_model_dir}")
print("\n개선 사항 요약:")
print("✅ Cosine Learning Rate Scheduler with Warmup")
print("✅ Label Smoothing (0.1)")
print("✅ FP16 Mixed Precision Training")
print("✅ GPU 최적화")
print("✅ Google Drive 연동")
print("="*70)
