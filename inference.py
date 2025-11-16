"""
inference.py - 최종 제출용 추론 스크립트

Usage:
    python inference.py --checkpoint ./outputs/kobart_summ_xxx/checkpoints/checkpoint-469
                        --test_data ./data/test/
                        --output ./submission.csv
"""

import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from src.data_loader import load_json_data
from src.preprocessing import postprocess_summary
import pandas as pd
import argparse
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='KoBART Summarization Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='체크포인트 경로')
    parser.add_argument('--test_data', type=str, default='./data/test/',
                        help='테스트 데이터 경로')
    parser.add_argument('--output', type=str, default='./submission.csv',
                        help='제출 파일 경로')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='배치 크기')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*70)
    print("KoBART 요약 추론")
    print("="*70)
    print(f"체크포인트: {args.checkpoint}")
    print(f"테스트 데이터: {args.test_data}")
    print(f"출력 파일: {args.output}")
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    
    # 모델 로드
    print("\n모델 로드 중...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-summarization")
    model = BartForConditionalGeneration.from_pretrained(args.checkpoint)
    model.eval()
    model = model.to(device)
    print("✓ 모델 로드 완료")
    
    # 테스트 데이터 로드
    print("\n테스트 데이터 로드 중...")
    test_data = load_json_data(args.test_data)
    print(f"✓ 테스트 데이터: {len(test_data):,}개")
    
    # 추론
    print("\n추론 시작...")
    predictions = []
    
    for i, row in tqdm(test_data.iterrows(), total=len(test_data)):
        dialogue = row['dialogue']
        
        # 토크나이징
        inputs = tokenizer(dialogue, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        # 생성
        with torch.no_grad():
            output = model.generate(
                inputs['input_ids'],
                max_length=80,
                min_length=15,
                num_beams=5,
                length_penalty=1.5,
                no_repeat_ngram_size=4,
                early_stopping=True,
                repetition_penalty=1.3,  # ✅ 최적값
            )
        
        # 디코딩 & 후처리
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        pred = postprocess_summary(pred)
        predictions.append(pred)
    
    print("✓ 추론 완료")
    
    # 제출 파일 생성
    print("\n제출 파일 생성 중...")
    submission = pd.DataFrame({
        'id': test_data.index,  # 실제 대회 포맷에 맞게 수정
        'summary': predictions
    })
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    submission.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"✓ 제출 파일 저장: {args.output}")
    
    # 샘플 확인
    print("\n" + "="*70)
    print("생성 샘플 (5개)")
    print("="*70)
    for i in range(min(5, len(test_data))):
        print(f"\n[샘플 {i+1}]")
        print(f"대화: {test_data.iloc[i]['dialogue'][:100]}...")
        print(f"생성: {predictions[i]}")
    
    print("\n" + "="*70)
    print("완료!")
    print("="*70)

if __name__ == '__main__':
    main()