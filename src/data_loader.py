"""
데이터 로더 with 캐싱
한국어 일상대화 요약 데이터셋 전처리
"""

import hashlib
import pickle
import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from .preprocessing import preprocess_text


def load_json_data(data_dir):
    """
    JSON 형식의 대화 데이터 로드 및 전처리

    Args:
        data_dir: 데이터 디렉토리 경로

    Returns:
        pd.DataFrame: 대화와 요약이 포함된 데이터프레임
    """
    dialogues = []
    summaries = []

    # JSON 파일 수집
    json_files = list(Path(data_dir).glob('*.json'))
    print(f"JSON 파일 {len(json_files)}개 발견")

    for json_file in tqdm(json_files, desc="파일 로딩"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'data' in data and isinstance(data['data'], list):
                for item in data['data']:
                    if 'body' in item and 'dialogue' in item['body']:
                        dialogue_list = item['body']['dialogue']

                        # 대화 텍스트 추출
                        dialogue_text = ' '.join([
                            utt['utterance'] for utt in dialogue_list
                            if 'utterance' in utt
                        ])

                        if 'summary' in item['body']:
                            summary = item['body']['summary']

                            if dialogue_text and summary:
                                # 전처리 적용
                                dialogue_text = preprocess_text(dialogue_text)
                                summary = preprocess_text(summary)

                                dialogues.append(dialogue_text)
                                summaries.append(summary)

        except Exception as e:
            print(f"에러: {json_file.name}: {e}")
            continue

    # DataFrame 생성
    df = pd.DataFrame({'dialogue': dialogues, 'summary': summaries})
    print(f"✓ 로드 완료: {len(df):,}개")

    return df


def load_data_with_cache(data_path='./data_sample/',
                          sample_size=15000,
                          use_cache=True):
    """
    캐싱을 사용한 데이터 로드

    Args:
        data_path: 데이터 경로
        sample_size: 샘플 크기
        use_cache: 캐시 사용 여부

    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    # 캐시 키 생성
    cache_key = hashlib.md5(
        f"{data_path}_{sample_size}".encode()
    ).hexdigest()
    cache_dir = '.cache'
    cache_file = f'{cache_dir}/{cache_key}.pkl'

    # 캐시가 있으면 로드
    if use_cache and os.path.exists(cache_file):
        print(f"✓ 캐시에서 로드 중: {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ 캐시 로드 완료: {len(data):,}개")
        return data

    # 캐시가 없으면 전처리
    print(f"\n전처리 시작")

    # 여러 서브디렉토리에서 데이터 로드
    all_data = []
    for subdir in ['train_sample', 'val_sample']:
        subdir_path = os.path.join(data_path, subdir)
        if os.path.exists(subdir_path):
            df = load_json_data(subdir_path)
            all_data.append(df)

    # 모든 데이터 합치기
    if all_data:
        data = pd.concat(all_data, ignore_index=True)
    else:
        # 서브디렉토리가 없으면 직접 로드
        data = load_json_data(data_path)

    # 샘플링
    if len(data) > sample_size:
        data = data.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"✓ 샘플링: 전체 → {len(data):,}개")

    # 캐시 저장
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ 캐시 저장 완료: {cache_file}")

    return data


if __name__ == '__main__':
    # 테스트
    print("="*70)
    print("데이터 로더 테스트")
    print("="*70)

    # 데이터 로드 테스트
    print("\n데이터 로드 테스트:")
    data = load_data_with_cache(
        data_path='./data_sample/',
        sample_size=100
    )
    print(f"로드된 데이터: {len(data)}개")
    print(f"\n샘플:\n{data.iloc[0]['dialogue'][:100]}...")
