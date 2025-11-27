"""
Solar 모델을 활용한 한국어 대화 요약 - Zero-shot / Few-shot 테스트
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc

# ============================================
# 1. 설정
# ============================================
MODEL_NAME = "upstage/SOLAR-10.7B-Instruct-v1.0"  # 또는 "upstage/SOLAR-10.7B-v1.0"

# 데이터 경로
TRAIN_PATH = "./data/train.csv"
DEV_PATH = "./data/dev.csv"
TEST_PATH = "./data/test.csv"
OUTPUT_PATH = "./submission.csv"

# ============================================
# 2. 모델 로드
# ============================================
def load_model():
    """Solar 모델 로드 (4bit 양자화로 메모리 절약)"""
    print("모델 로딩 중...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # padding token 설정 (없는 경우)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4bit 양자화 설정 (VRAM 절약)
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model.eval()  # 평가 모드로 설정
    print("모델 로드 완료!")
    return model, tokenizer

# ============================================
# 3. 프롬프트 템플릿
# ============================================
def create_prompt(dialogue, few_shot_examples=None):
    """요약을 위한 프롬프트 생성"""
    
    # Zero-shot 프롬프트
    if few_shot_examples is None:
        prompt = f"""### 지시사항
다음 대화를 읽고 핵심 내용을 간결하게 요약하세요.

### 대화
{dialogue}

### 요약
"""
    # Few-shot 프롬프트
    else:
        examples_text = ""
        for ex in few_shot_examples:
            examples_text += f"""### 대화
{ex['dialogue']}

### 요약
{ex['summary']}

"""
        prompt = f"""### 지시사항
다음 대화를 읽고 핵심 내용을 간결하게 요약하세요.

{examples_text}### 대화
{dialogue}

### 요약
"""
    
    return prompt

# ============================================
# 4. 추론 함수
# ============================================
def generate_summary(model, tokenizer, dialogue, few_shot_examples=None, max_new_tokens=150):
    """단일 대화에 대한 요약 생성"""

    prompt = create_prompt(dialogue, few_shot_examples)

    # 입력 길이 제한 (메모리 관리)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=2048,
        truncation=True,
        padding=False
    )

    # GPU로 이동
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,  # 낮은 temperature로 일관성 있는 출력
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 생성된 부분만 추출
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    summary = tokenizer.decode(generated, skip_special_tokens=True)

    # 후처리: 불필요한 부분 제거
    summary = summary.strip()
    if "###" in summary:
        summary = summary.split("###")[0].strip()
    if "\n\n" in summary:
        summary = summary.split("\n\n")[0].strip()

    return summary

# ============================================
# 5. 배치 추론
# ============================================
def run_inference(model, tokenizer, df, few_shot_examples=None, desc="Generating"):
    """전체 데이터셋에 대한 추론"""

    summaries = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        dialogue = row['dialogue']

        try:
            summary = generate_summary(model, tokenizer, dialogue, few_shot_examples)
        except Exception as e:
            print(f"\nError at {row['id']}: {e}")
            summary = ""

        summaries.append(summary)

        # 메모리 정리 (10개마다 - GPU 메모리 관리)
        if (idx + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return summaries

# ============================================
# 6. 평가 함수 (ROUGE)
# ============================================
def evaluate_rouge(predictions, references):
    """ROUGE 점수 계산"""
    from rouge_score import rouge_scorer
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    results = {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores),
        'rouge2': sum(rouge2_scores) / len(rouge2_scores),
        'rougeL': sum(rougeL_scores) / len(rougeL_scores),
    }
    results['total'] = results['rouge1'] + results['rouge2'] + results['rougeL']
    
    return results

# ============================================
# 7. 메인 실행
# ============================================
def main():
    # GPU 초기화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 데이터 로드
    print("데이터 로딩 중...")
    train_df = pd.read_csv(TRAIN_PATH)
    dev_df = pd.read_csv(DEV_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    # 모델 로드
    model, tokenizer = load_model()

    # Few-shot 예시 준비 (train에서 좋은 예시 3개 선택)
    few_shot_examples = [
        {"dialogue": train_df.iloc[0]['dialogue'], "summary": train_df.iloc[0]['summary']},
        {"dialogue": train_df.iloc[1]['dialogue'], "summary": train_df.iloc[1]['summary']},
        {"dialogue": train_df.iloc[2]['dialogue'], "summary": train_df.iloc[2]['summary']},
    ]

    # ===== Dev 세트로 테스트 =====
    print("\n===== Dev 세트 평가 (Zero-shot) =====")
    dev_summaries_zero = run_inference(model, tokenizer, dev_df, few_shot_examples=None, desc="Zero-shot")

    rouge_zero = evaluate_rouge(dev_summaries_zero, dev_df['summary'].tolist())
    print(f"Zero-shot ROUGE-1: {rouge_zero['rouge1']:.4f}")
    print(f"Zero-shot ROUGE-2: {rouge_zero['rouge2']:.4f}")
    print(f"Zero-shot ROUGE-L: {rouge_zero['rougeL']:.4f}")
    print(f"Zero-shot Total: {rouge_zero['total']:.4f}")

    # GPU 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n===== Dev 세트 평가 (Few-shot) =====")
    dev_summaries_few = run_inference(model, tokenizer, dev_df, few_shot_examples=few_shot_examples, desc="Few-shot")

    rouge_few = evaluate_rouge(dev_summaries_few, dev_df['summary'].tolist())
    print(f"Few-shot ROUGE-1: {rouge_few['rouge1']:.4f}")
    print(f"Few-shot ROUGE-2: {rouge_few['rouge2']:.4f}")
    print(f"Few-shot ROUGE-L: {rouge_few['rougeL']:.4f}")
    print(f"Few-shot Total: {rouge_few['total']:.4f}")

    # GPU 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===== 더 좋은 방식으로 Test 추론 =====
    best_method = "few_shot" if rouge_few['total'] > rouge_zero['total'] else "zero_shot"
    print(f"\n더 좋은 방식: {best_method}")

    if best_method == "few_shot":
        test_summaries = run_inference(model, tokenizer, test_df, few_shot_examples=few_shot_examples, desc="Test")
    else:
        test_summaries = run_inference(model, tokenizer, test_df, few_shot_examples=None, desc="Test")

    # 제출 파일 생성
    submission = pd.DataFrame({
        'id': test_df['id'],
        'summary': test_summaries
    })
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\n제출 파일 저장: {OUTPUT_PATH}")

    # 최종 메모리 정리
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nFinal GPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    main()