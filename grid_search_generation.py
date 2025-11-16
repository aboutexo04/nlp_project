"""
grid_search_generation.py - Generation 파라미터 최적화
"""
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from src.data_loader import load_json_data
from src.postprocessing import fix_summary_punctuation_and_format
from rouge_score import rouge_scorer
import numpy as np
from itertools import product
import random

# 최고 체크포인트
checkpoint = "./outputs/kobart_final_20251116_112722_rouge0_2423/checkpoints/checkpoint-1876"

print("="*70)
print("Generation 파라미터 Grid Search")
print("="*70)

# 모델 로드
print("\n모델 로드 중...")
tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2")
model = BartForConditionalGeneration.from_pretrained(checkpoint)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"✓ 디바이스: {device}")

# Validation 데이터
val_data = load_json_data('./data/sample/val_sample/')
print(f"✓ Validation: {len(val_data):,}개")

# 파라미터 공간
params = {
    'num_beams': [5, 6, 7],
    'length_penalty': [1.2, 1.5, 1.8],
    'repetition_penalty': [1.2, 1.3, 1.5],
    'max_length': [100, 120],
}

# 전체 조합
all_combos = list(product(
    params['num_beams'],
    params['length_penalty'],
    params['repetition_penalty'],
    params['max_length']
))

print(f"\n전체 조합: {len(all_combos)}개")

# 랜덤 샘플링 (20개)
selected = random.sample(all_combos, min(20, len(all_combos)))

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

best_rouge = 0
best_params = None

print("\n" + "="*70)
print("Grid Search 시작")
print("="*70)

for i, (beams, length_pen, rep_pen, max_len) in enumerate(selected, 1):
    print(f"\n[{i}/{len(selected)}]")
    print(f"  beams={beams}, length_penalty={length_pen}")
    print(f"  rep_penalty={rep_pen}, max_length={max_len}")
    
    rouge_scores = []
    
    # 처음 100개로 빠른 평가
    for idx in range(min(100, len(val_data))):
        dialogue = val_data.iloc[idx]['dialogue']
        summary = val_data.iloc[idx]['summary']
        
        inputs = tokenizer(dialogue, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            output = model.generate(
                inputs['input_ids'],
                max_length=max_len,
                min_length=15,
                num_beams=beams,
                length_penalty=length_pen,
                no_repeat_ngram_size=4,
                early_stopping=True,
                repetition_penalty=rep_pen,
            )
        
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = fix_summary_punctuation_and_format(pred)
        
        scores = scorer.score(summary, pred)
        rouge_score = scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure
        rouge_scores.append(rouge_score)
    
    avg_rouge = np.mean(rouge_scores)
    print(f"  → ROUGE: {avg_rouge:.4f}")
    
    if avg_rouge > best_rouge:
        best_rouge = avg_rouge
        best_params = {
            'num_beams': beams,
            'length_penalty': length_pen,
            'repetition_penalty': rep_pen,
            'max_length': max_len
        }
        print(f"  ⭐ 새로운 최고 기록!")

print("\n" + "="*70)
print("Grid Search 완료!")
print("="*70)
print(f"\n최적 파라미터:")
for key, value in best_params.items():
    print(f"  {key}: {value}")
print(f"\n예상 ROUGE: {best_rouge:.4f}")

# 전체 validation으로 최종 검증
print("\n전체 Validation으로 최종 검증 중...")
final_scores = []

for idx in range(len(val_data)):
    dialogue = val_data.iloc[idx]['dialogue']
    summary = val_data.iloc[idx]['summary']
    
    inputs = tokenizer(dialogue, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'],
            **best_params,
            min_length=15,
            no_repeat_ngram_size=4,
            early_stopping=True,
        )
    
    pred = tokenizer.decode(output[0], skip_special_tokens=True)
    pred = fix_summary_punctuation_and_format(pred)
    
    scores = scorer.score(summary, pred)
    rouge_score = scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure
    final_scores.append(rouge_score)

final_rouge = np.mean(final_scores)

print(f"\n최종 ROUGE (전체 Val): {final_rouge:.4f}")
print(f"기존 대비 변화: {final_rouge - 0.2423:+.4f}")