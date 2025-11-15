"""
학습된 KoBART 모델 테스트 스크립트
저장된 모델을 불러와서 요약 생성 테스트
"""

import torch
from transformers import (
    BartForConditionalGeneration,
    PreTrainedTokenizerFast
)
import pandas as pd
import json
import os
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer

# ===== 설정 =====
MODEL_PATH = "./kobart_test_final"  # 학습된 모델 경로
DATA_PATH = "./data_sample/val_sample/"  # 테스트 데이터 경로
NUM_SAMPLES = 10  # 테스트할 샘플 수

# 생성 파라미터 (EDA 분석 기반)
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
USE_DYNAMIC_LENGTH = True  # 동적 길이 사용 여부
COMPRESSION_RATIO = 0.24  # EDA 분석: 평균 압축 비율 24.39%
MIN_LENGTH_ABSOLUTE = 15  # 절대 최소 길이 (너무 짧은 요약 방지)
MAX_LENGTH_ABSOLUTE = 128  # 절대 최대 길이
NUM_BEAMS = 4
LENGTH_PENALTY = 1.0  # 중립적
NO_REPEAT_NGRAM_SIZE = 3

print("="*70)
print("KoBART 모델 테스트")
print("="*70)
print(f"모델 경로: {MODEL_PATH}")
print(f"테스트 샘플 수: {NUM_SAMPLES}")
print("="*70)

# ===== 1. 디바이스 설정 =====
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

# ===== 2. 모델과 토크나이저 로드 =====
print("\n" + "="*70)
print("모델 로드")
print("="*70)

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
model = model.to(device)
model.eval()  # 평가 모드

print(f"✓ 모델 로드 완료")

# ===== 3. 동적 길이 계산 함수 =====
def calculate_dynamic_length(input_text, compression_ratio=COMPRESSION_RATIO):
    """
    입력 텍스트 길이에 기반하여 동적으로 요약 길이 계산

    Args:
        input_text: 입력 대화 텍스트
        compression_ratio: 압축 비율 (기본값: 0.24)

    Returns:
        (min_length, max_length) 튜플
    """
    input_length = len(input_text)

    # 목표 길이 = 입력 길이 * 압축 비율
    target_length = int(input_length * compression_ratio)

    # min_length: 목표 길이의 70%
    min_length = int(target_length * 0.7)

    # max_length: 목표 길이의 150%
    max_length = int(target_length * 1.5)

    # 절대 최소/최대 길이로 제한
    min_length = max(MIN_LENGTH_ABSOLUTE, min(min_length, MAX_LENGTH_ABSOLUTE))
    max_length = max(min_length + 10, min(max_length, MAX_LENGTH_ABSOLUTE))

    return min_length, max_length

# ===== 4. 테스트 데이터 로드 =====
print("\n" + "="*70)
print("데이터 로드")
print("="*70)

dialogues = []
summaries = []

json_files = []
for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.json'):
            json_files.append(os.path.join(root, file))

print(f"JSON 파일 {len(json_files)}개 발견")

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

                    if len(dialogues) >= NUM_SAMPLES:
                        break

    except Exception as e:
        print(f"에러: {os.path.basename(json_file)}: {e}")
        continue

    if len(dialogues) >= NUM_SAMPLES:
        break

test_df = pd.DataFrame({'dialogue': dialogues, 'summary': summaries})
print(f"✓ 테스트 데이터 로드 완료: {len(test_df)}개")

# ===== 4. 요약 생성 및 평가 =====
print("\n" + "="*70)
print("요약 생성 및 평가")
print("="*70)

# ROUGE 점수 계산을 위한 준비
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

predictions = []
references = []

print("\n생성 파라미터:")
print(f"  동적 길이 사용: {USE_DYNAMIC_LENGTH}")
if USE_DYNAMIC_LENGTH:
    print(f"  압축 비율: {COMPRESSION_RATIO * 100:.1f}%")
    print(f"  절대 최소 길이: {MIN_LENGTH_ABSOLUTE}")
    print(f"  절대 최대 길이: {MAX_LENGTH_ABSOLUTE}")
else:
    print(f"  고정 min_length: {MIN_LENGTH_ABSOLUTE}")
    print(f"  고정 max_length: {MAX_LENGTH_ABSOLUTE}")
print(f"  num_beams: {NUM_BEAMS}")
print(f"  length_penalty: {LENGTH_PENALTY}")
print(f"  no_repeat_ngram_size: {NO_REPEAT_NGRAM_SIZE}")
print()

for idx in range(len(test_df)):
    dialogue = test_df.iloc[idx]['dialogue']
    reference = test_df.iloc[idx]['summary']

    # 동적 길이 계산
    if USE_DYNAMIC_LENGTH:
        min_len, max_len = calculate_dynamic_length(dialogue, COMPRESSION_RATIO)
    else:
        min_len = MIN_LENGTH_ABSOLUTE
        max_len = MAX_LENGTH_ABSOLUTE

    # 토크나이징 및 생성
    inputs = tokenizer(dialogue, return_tensors="pt", max_length=MAX_INPUT_LENGTH, truncation=True).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=max_len,
            min_length=min_len,
            num_beams=NUM_BEAMS,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            early_stopping=True
        )

    prediction = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # ROUGE 점수 계산
    scores = scorer.score(reference, prediction)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

    predictions.append(prediction)
    references.append(reference)

    # 샘플 출력
    print(f"\n[샘플 {idx+1}]")
    print(f"대화 ({len(dialogue)}자): {dialogue[:100]}...")
    if USE_DYNAMIC_LENGTH:
        print(f"동적 길이 설정: min={min_len}, max={max_len} (압축률 {COMPRESSION_RATIO*100:.1f}%)")
    print(f"실제 요약 ({len(reference)}자): {reference}")
    print(f"생성 요약 ({len(prediction)}자): {prediction}")
    print(f"압축률: {len(prediction)/len(dialogue)*100:.1f}%")
    print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}, ROUGE-2: {scores['rouge2'].fmeasure:.4f}, ROUGE-L: {scores['rougeL'].fmeasure:.4f}")

# ===== 5. 전체 평가 결과 =====
print("\n" + "="*70)
print("전체 평가 결과")
print("="*70)

rouge1_avg = np.mean(rouge1_scores)
rouge2_avg = np.mean(rouge2_scores)
rougeL_avg = np.mean(rougeL_scores)
final_score = rouge1_avg + rouge2_avg + rougeL_avg

print(f"\nROUGE-1 F1: {rouge1_avg:.4f}")
print(f"ROUGE-2 F1: {rouge2_avg:.4f}")
print(f"ROUGE-L F1: {rougeL_avg:.4f}")
print(f"최종 ROUGE Score (대회 평가 기준): {final_score:.4f}")

# ===== 6. 길이 통계 =====
print("\n" + "="*70)
print("길이 통계")
print("="*70)

pred_lengths = [len(p) for p in predictions]
ref_lengths = [len(r) for r in references]

print(f"\n생성 요약 길이:")
print(f"  평균: {np.mean(pred_lengths):.1f}자")
print(f"  중앙값: {np.median(pred_lengths):.1f}자")
print(f"  최소: {np.min(pred_lengths)}자")
print(f"  최대: {np.max(pred_lengths)}자")

print(f"\n실제 요약 길이:")
print(f"  평균: {np.mean(ref_lengths):.1f}자")
print(f"  중앙값: {np.median(ref_lengths):.1f}자")

print("\n" + "="*70)
print("테스트 완료!")
print("="*70)
