import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
from collections import Counter

# 시드 고정 (재현성)
random.seed(42)

def load_all_data(data_dir):
    """모든 JSON 파일에서 데이터 로드"""
    all_items = []

    for json_file in Path(data_dir).glob('*.json'):
        with open(json_file, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
            category = json_file.stem

            # data 키 안에 실제 대화 데이터가 있음
            if 'data' in file_data and isinstance(file_data['data'], list):
                for item in file_data['data']:
                    # 카테고리 정보를 별도로 추가
                    all_items.append({
                        'category': category,
                        'item': item
                    })

    return all_items

def prepare_samples_for_stratification(data):
    """카테고리(토픽) 분포를 유지하기 위한 데이터 준비"""
    samples = []
    labels = []

    for wrapped_item in data:
        item = wrapped_item['item']
        category = wrapped_item['category']

        # 카테고리를 레이블로 사용 (토픽 분포 유지)
        samples.append(wrapped_item)
        labels.append(category)

    return samples, labels

def save_sampled_data(sampled_items, output_dir):
    """샘플링된 데이터를 카테고리별로 저장"""
    # 카테고리별로 데이터 그룹화
    category_data = {}

    for sample in sampled_items:
        category = sample['category']
        item = sample['item']

        if category not in category_data:
            category_data[category] = []

        # dialogueID 기준으로 중복 제거
        dialogue_id = item['header']['dialogueInfo']['dialogueID']
        existing_ids = [c['header']['dialogueInfo']['dialogueID'] for c in category_data[category]]

        if dialogue_id not in existing_ids:
            category_data[category].append(item)

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 카테고리별로 파일 저장
    total_items = 0
    for category, items in category_data.items():
        output_file = os.path.join(output_dir, f'{category}.json')

        # 원본 파일 형식 유지 (numberOfItems, data)
        output_data = {
            'numberOfItems': len(items),
            'data': items
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f'{category}: {len(items)} dialogues')
        total_items += len(items)

    print(f'\nTotal dialogues: {total_items}')

def main():
    # 경로 설정
    train_dir = 'data/train'
    val_dir = 'data/val'
    output_train_dir = 'data/train_sample'
    output_val_dir = 'data/val_sample'

    # 샘플 크기 설정 (대화 수 기준)
    target_train_dialogues = 15000

    print("=" * 60)
    print("데이터 로딩 중...")
    print("=" * 60)

    # 훈련 데이터 로드
    train_data = load_all_data(train_dir)
    print(f'Original train dialogues: {len(train_data)}')

    # 검증 데이터 로드
    val_data = load_all_data(val_dir)
    print(f'Original val dialogues: {len(val_data)}')

    # Stratified sampling을 위한 준비 (카테고리 분포 유지)
    print("\n" + "=" * 60)
    print("Stratified Sampling 준비 중 (카테고리 분포 유지)...")
    print("=" * 60)

    train_samples, train_labels = prepare_samples_for_stratification(train_data)
    print(f'Total train dialogues: {len(train_samples)}')

    # 카테고리 분포 확인
    label_dist = Counter(train_labels)
    print("\n원본 카테고리(토픽) 분포:")
    for category, count in sorted(label_dist.items()):
        print(f'  {category}: {count} ({count/len(train_labels)*100:.1f}%)')

    # Stratified sampling 수행 (카테고리 비율 유지)
    print("\n" + "=" * 60)
    print(f"Stratified Sampling 수행 중 (목표: {target_train_dialogues} dialogues)...")
    print("=" * 60)

    sampled_items, _ = train_test_split(
        train_samples,
        train_size=target_train_dialogues,
        stratify=train_labels,
        random_state=42
    )

    # 샘플링된 데이터의 카테고리 분포 확인
    sampled_labels = [item['category'] for item in sampled_items]
    sampled_label_dist = Counter(sampled_labels)
    print("\n샘플링된 카테고리(토픽) 분포:")
    for category, count in sorted(sampled_label_dist.items()):
        print(f'  {category}: {count} ({count/len(sampled_labels)*100:.1f}%)')

    # 훈련 데이터 저장
    print("\n" + "=" * 60)
    print("샘플링된 훈련 데이터 저장 중...")
    print("=" * 60)
    save_sampled_data(sampled_items, output_train_dir)

    # 검증 데이터도 비율에 맞춰 샘플링
    val_target_dialogues = int(target_train_dialogues * len(val_data) / len(train_data))

    print("\n" + "=" * 60)
    print(f"검증 데이터 Stratified Sampling 수행 중 (목표: {val_target_dialogues} dialogues)...")
    print("=" * 60)

    val_samples, val_labels = prepare_samples_for_stratification(val_data)

    sampled_val_items, _ = train_test_split(
        val_samples,
        train_size=val_target_dialogues,
        stratify=val_labels,
        random_state=42
    )

    # 검증 데이터 저장
    print("\n샘플링된 검증 데이터 저장 중...")
    save_sampled_data(sampled_val_items, output_val_dir)

    print("\n" + "=" * 60)
    print("완료!")
    print("=" * 60)
    print(f"샘플링된 훈련 데이터: {output_train_dir}/")
    print(f"샘플링된 검증 데이터: {output_val_dir}/")

if __name__ == '__main__':
    main()
