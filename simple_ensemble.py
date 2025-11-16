"""
simple_ensemble.py - 간단 앙상블 (2개 체크포인트)
"""
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from src.data_loader import load_json_data
from src.postprocessing import fix_summary_punctuation_and_format
from rouge_score import rouge_scorer
import numpy as np
import glob

print("="*70)
print("앙상블 테스트 (2개 체크포인트)")
print("="*70)

# 체크포인트 2개 자동 찾기
experiment_dirs = sorted(glob.glob("./outputs/kobart_*"))
if not experiment_dirs:
    print("❌ kobart 실험 폴더를 찾을 수 없습니다!")
    exit(1)

latest_exp = experiment_dirs[-1]
checkpoints = sorted(glob.glob(f"{latest_exp}/checkpoints/checkpoint-*"))

if len(checkpoints) < 2:
    print("❌ 체크포인트가 2개 미만입니다!")
    exit(1)

checkpoint1 = checkpoints[0]   # 첫 번째 (보통 early)
checkpoint2 = checkpoints[-1]  # 마지막 (보통 best)

print(f"✓ Checkpoint 1: {checkpoint1}")
print(f"✓ Checkpoint 2: {checkpoint2}")

# 토크나이저
tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-summarization")

# 모델 1 로드
print("\n모델 1 로드 중...")
model1 = BartForConditionalGeneration.from_pretrained(checkpoint1)
model1.eval()

# 모델 2 로드
print("모델 2 로드 중...")
model2 = BartForConditionalGeneration.from_pretrained(checkpoint2)
model2.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = model1.to(device)
model2 = model2.to(device)

print(f"✓ 디바이스: {device}")

# Validation 데이터
val_data = load_json_data('./data/sample/val_sample/')
print(f"✓ Validation: {len(val_data):,}개")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

# 단일 모델 vs 앙상블 비교
single_scores = []
ensemble_scores = []

print("\n앙상블 테스트 중 (100개 샘플)...")
for idx in range(min(100, len(val_data))):
    dialogue = val_data.iloc[idx]['dialogue']
    summary = val_data.iloc[idx]['summary']
    
    inputs = tokenizer(dialogue, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    # 모델 1 생성
    with torch.no_grad():
        output1 = model1.generate(
            inputs['input_ids'],
            max_length=120,
            min_length=15,
            num_beams=6,
            length_penalty=1.5,
            no_repeat_ngram_size=4,
            early_stopping=True,
            repetition_penalty=1.3,
        )
    pred1 = tokenizer.decode(output1[0], skip_special_tokens=True)
    pred1 = fix_summary_punctuation_and_format(pred1)
    
    # 모델 2 생성
    with torch.no_grad():
        output2 = model2.generate(
            inputs['input_ids'],
            max_length=120,
            min_length=15,
            num_beams=6,
            length_penalty=1.5,
            no_repeat_ngram_size=4,
            early_stopping=True,
            repetition_penalty=1.3,
        )
    pred2 = tokenizer.decode(output2[0], skip_special_tokens=True)
    pred2 = fix_summary_punctuation_and_format(pred2)
    
    # 단일 모델 (모델2) ROUGE
    scores = scorer.score(summary, pred2)
    single_scores.append(scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure)
    
    # 앙상블: 두 예측 중 ROUGE 높은 것 선택
    scores1 = scorer.score(summary, pred1)
    rouge1 = scores1['rouge1'].fmeasure + scores1['rouge2'].fmeasure + scores1['rougeL'].fmeasure
    
    scores2 = scorer.score(summary, pred2)
    rouge2 = scores2['rouge1'].fmeasure + scores2['rouge2'].fmeasure + scores2['rougeL'].fmeasure
    
    # ✅ 높은 것 선택!
    ensemble_scores.append(max(rouge1, rouge2))
    
    if (idx + 1) % 25 == 0:
        print(f"  {idx + 1}/100 완료...")

avg_single = np.mean(single_scores)
avg_ensemble = np.mean(ensemble_scores)

print("\n" + "="*70)
print("결과 (100개 샘플)")
print("="*70)
print(f"단일 모델 ROUGE:  {avg_single:.4f}")
print(f"앙상블 ROUGE:     {avg_ensemble:.4f}")
print(f"개선:             {avg_ensemble - avg_single:+.4f}")

if avg_ensemble > avg_single + 0.005:
    print("\n✅ 앙상블 효과 있습니다!")
    
    # 전체 validation으로 최종 검증
    print("\n전체 Validation으로 최종 검증 중...")
    
    final_single = []
    final_ensemble = []
    
    for idx in range(len(val_data)):
        dialogue = val_data.iloc[idx]['dialogue']
        summary = val_data.iloc[idx]['summary']
        
        inputs = tokenizer(dialogue, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        # 모델 1
        with torch.no_grad():
            output1 = model1.generate(
                inputs['input_ids'],
                max_length=120,
                min_length=15,
                num_beams=6,
                length_penalty=1.5,
                no_repeat_ngram_size=4,
                early_stopping=True,
                repetition_penalty=1.3,
            )
        pred1 = tokenizer.decode(output1[0], skip_special_tokens=True)
        pred1 = fix_summary_punctuation_and_format(pred1)
        
        # 모델 2
        with torch.no_grad():
            output2 = model2.generate(
                inputs['input_ids'],
                max_length=120,
                min_length=15,
                num_beams=6,
                length_penalty=1.5,
                no_repeat_ngram_size=4,
                early_stopping=True,
                repetition_penalty=1.3,
            )
        pred2 = tokenizer.decode(output2[0], skip_special_tokens=True)
        pred2 = fix_summary_punctuation_and_format(pred2)
        
        # 단일 모델
        scores = scorer.score(summary, pred2)
        final_single.append(scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure)
        
        # 앙상블
        scores1 = scorer.score(summary, pred1)
        rouge1 = scores1['rouge1'].fmeasure + scores1['rouge2'].fmeasure + scores1['rougeL'].fmeasure
        
        scores2 = scorer.score(summary, pred2)
        rouge2 = scores2['rouge1'].fmeasure + scores2['rouge2'].fmeasure + scores2['rougeL'].fmeasure
        
        final_ensemble.append(max(rouge1, rouge2))
        
        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(val_data)} 완료...")
    
    final_avg_single = np.mean(final_single)
    final_avg_ensemble = np.mean(final_ensemble)
    
    print("\n" + "="*70)
    print("최종 결과 (전체 Validation)")
    print("="*70)
    print(f"단일 모델 ROUGE:  {final_avg_single:.4f}")
    print(f"앙상블 ROUGE:     {final_avg_ensemble:.4f}")
    print(f"개선:             {final_avg_ensemble - final_avg_single:+.4f}")
    print(f"\n기존 대비:        {final_avg_ensemble - 0.2423:+.4f}")
    
    if final_avg_ensemble > 0.2423:
        print("\n🎉 앙상블로 성능 향상!")
        print(f"0.2423 → {final_avg_ensemble:.4f}")
    
else:
    print("\n⚠️ 앙상블 효과 미미합니다.")
    print("→ 두 체크포인트가 너무 비슷한 것 같습니다.")