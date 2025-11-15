"""
샘플 데이터셋 EDA 분석 스크립트
"""
import json
import os
from pathlib import Path
from collections import Counter
import pandas as pd

def analyze_dataset(data_dir, dataset_name="Train"):
    """데이터셋 분석"""
    print(f"\n{'='*70}")
    print(f"{dataset_name} 샘플 데이터 분석")
    print(f"{'='*70}")

    all_dialogues = []
    category_stats = {}

    # 카테고리별 데이터 로드
    for json_file in sorted(Path(data_dir).glob('*.json')):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        category = json_file.stem
        dialogues = data['data']

        # 카테고리별 통계
        category_stats[category] = {
            'num_dialogues': len(dialogues),
            'total_utterances': 0,
            'total_turns': 0,
            'avg_utterances': 0,
            'avg_turns': 0,
            'participant_counts': Counter(),
            'gender_dist': Counter(),
            'age_dist': Counter()
        }

        for dialogue in dialogues:
            all_dialogues.append({
                'category': category,
                'dialogue': dialogue
            })

            # 대화 정보
            dialogue_info = dialogue['header']['dialogueInfo']
            category_stats[category]['total_utterances'] += dialogue_info['numberOfUtterances']
            category_stats[category]['total_turns'] += dialogue_info['numberOfTurns']
            category_stats[category]['participant_counts'][dialogue_info['numberOfParticipants']] += 1

            # 참여자 정보
            for participant in dialogue['header']['participantsInfo']:
                category_stats[category]['gender_dist'][participant['gender']] += 1
                category_stats[category]['age_dist'][participant['age']] += 1

        # 평균 계산
        num_dialogues = category_stats[category]['num_dialogues']
        category_stats[category]['avg_utterances'] = category_stats[category]['total_utterances'] / num_dialogues
        category_stats[category]['avg_turns'] = category_stats[category]['total_turns'] / num_dialogues

    return all_dialogues, category_stats

def print_category_summary(category_stats):
    """카테고리별 요약 출력"""
    print(f"\n## 카테고리별 대화 수")
    print(f"{'카테고리':<20} {'대화 수':>10} {'발화 수':>10} {'턴 수':>10} {'평균 발화':>10} {'평균 턴':>10}")
    print("-" * 80)

    total_dialogues = 0
    total_utterances = 0
    total_turns = 0

    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        total_dialogues += stats['num_dialogues']
        total_utterances += stats['total_utterances']
        total_turns += stats['total_turns']

        print(f"{category:<20} {stats['num_dialogues']:>10,} {stats['total_utterances']:>10,} "
              f"{stats['total_turns']:>10,} {stats['avg_utterances']:>10.1f} {stats['avg_turns']:>10.1f}")

    print("-" * 80)
    print(f"{'전체':<20} {total_dialogues:>10,} {total_utterances:>10,} "
          f"{total_turns:>10,} {total_utterances/total_dialogues:>10.1f} {total_turns/total_dialogues:>10.1f}")

    return total_dialogues, total_utterances, total_turns

def print_participant_stats(category_stats):
    """참여자 통계 출력"""
    print(f"\n## 참여자 수 분포")
    participant_total = Counter()
    for stats in category_stats.values():
        participant_total.update(stats['participant_counts'])

    for num_participants in sorted(participant_total.keys()):
        count = participant_total[num_participants]
        print(f"  {num_participants}명 참여: {count:,}개 대화")

    print(f"\n## 성별 분포")
    gender_total = Counter()
    for stats in category_stats.values():
        gender_total.update(stats['gender_dist'])

    total = sum(gender_total.values())
    for gender, count in gender_total.most_common():
        print(f"  {gender}: {count:,}명 ({count/total*100:.1f}%)")

    print(f"\n## 연령대 분포")
    age_total = Counter()
    for stats in category_stats.values():
        age_total.update(stats['age_dist'])

    total = sum(age_total.values())
    for age, count in age_total.most_common():
        print(f"  {age}: {count:,}명 ({count/total*100:.1f}%)")

def analyze_utterances(all_dialogues, sample_size=100):
    """발화 길이 분석"""
    print(f"\n## 발화 길이 분석 (샘플 {sample_size}개 대화)")

    utterance_lengths = []

    for i, item in enumerate(all_dialogues[:sample_size]):
        dialogue = item['dialogue']
        for utt in dialogue['body']['dialogue']:
            utterance_lengths.append(len(utt['utterance']))

    if utterance_lengths:
        print(f"  평균 길이: {sum(utterance_lengths)/len(utterance_lengths):.1f}자")
        print(f"  최소 길이: {min(utterance_lengths)}자")
        print(f"  최대 길이: {max(utterance_lengths)}자")
        print(f"  중앙값: {sorted(utterance_lengths)[len(utterance_lengths)//2]}자")

def main():
    print("="*70)
    print("샘플 데이터셋 EDA 분석")
    print("="*70)

    # Train 데이터 분석
    train_dialogues, train_stats = analyze_dataset('data_sample/train_sample', 'Train')
    train_total_dialogues, train_total_utterances, train_total_turns = print_category_summary(train_stats)
    print_participant_stats(train_stats)
    analyze_utterances(train_dialogues)

    # Val 데이터 분석
    val_dialogues, val_stats = analyze_dataset('data_sample/val_sample', 'Validation')
    val_total_dialogues, val_total_utterances, val_total_turns = print_category_summary(val_stats)
    print_participant_stats(val_stats)
    analyze_utterances(val_dialogues)

    # 전체 요약
    print(f"\n{'='*70}")
    print("전체 샘플 데이터셋 요약")
    print(f"{'='*70}")
    print(f"Train 대화 수: {train_total_dialogues:,}개")
    print(f"Validation 대화 수: {val_total_dialogues:,}개")
    print(f"전체 대화 수: {train_total_dialogues + val_total_dialogues:,}개")
    print(f"Train/Val 비율: {train_total_dialogues/val_total_dialogues:.1f}:1")

    # 리포트 작성용 데이터 반환
    return {
        'train_stats': train_stats,
        'val_stats': val_stats,
        'train_total': (train_total_dialogues, train_total_utterances, train_total_turns),
        'val_total': (val_total_dialogues, val_total_utterances, val_total_turns)
    }

if __name__ == '__main__':
    results = main()
