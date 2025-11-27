from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from rouge_score import rouge_scorer
import pandas as pd
from tqdm import tqdm

# 체크포인트 경로
CHECKPOINT_PATH = "/root/nlp_project/outputs/gogamza_kobart-summarization_20251127_020530/checkpoints/checkpoint-389"

# 모델 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained(CHECKPOINT_PATH)
model = BartForConditionalGeneration.from_pretrained(CHECKPOINT_PATH).to("cuda")

# Dev 데이터 로드 (정답 있음!)
dev_df = pd.read_csv("./data/dev.csv")

# 생성 파라미터 실험
configs = [
    {"num_beams": 6, "length_penalty": 1.3, "max_length": 110},  # length_penalty 올리기
    {"num_beams": 7, "length_penalty": 1.2, "max_length": 110},  # num_beams 올리기
    {"num_beams": 6, "length_penalty": 1.2, "max_length": 115},  # max_length 올리기
]

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

for i, config in enumerate(configs):
    print(f"\n=== Config {i+1}: {config} ===")
    
    summaries = []
    for _, row in tqdm(dev_df.iterrows(), total=len(dev_df)):
        inputs = tokenizer(row['dialogue'], return_tensors="pt", 
                          max_length=512, truncation=True).to("cuda")
        
        output = model.generate(
            inputs['input_ids'],
            num_beams=config['num_beams'],
            length_penalty=config['length_penalty'],
            max_length=config['max_length'],
            min_length=15,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)
    
    # ROUGE 계산
    r1, r2, rL = [], [], []
    for pred, ref in zip(summaries, dev_df['summary']):
        scores = scorer.score(ref, pred)
        r1.append(scores['rouge1'].fmeasure)
        r2.append(scores['rouge2'].fmeasure)
        rL.append(scores['rougeL'].fmeasure)
    
    print(f"ROUGE-1: {sum(r1)/len(r1):.4f}")
    print(f"ROUGE-2: {sum(r2)/len(r2):.4f}")
    print(f"ROUGE-L: {sum(rL)/len(rL):.4f}")
    print(f"Total: {sum(r1)/len(r1) + sum(r2)/len(r2) + sum(rL)/len(rL):.4f}")