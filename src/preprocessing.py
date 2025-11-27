"""
텍스트 전처리 모듈
한국어 대화 요약을 위한 전처리 함수들
"""

import re

def remove_noise(text):
    """
    불필요한 노이즈 제거
    """
    # 1. 의미 없는 자음/모음 반복 (ㅋㅋㅋㅋ → ㅋㅋ)
    text = re.sub(r'([ㅋㅎㅠㅜㅡ])\1{2,}', r'\1\1', text)
    
    # 2. 과도한 느낌표/물음표 (!!!! → !)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    # 3. 불필요한 기호 반복 (~~~~ → ~)
    text = re.sub(r'~{2,}', '~', text)
    
    # 4. 반복 점 (...... → .)
    text = re.sub(r'\.{2,}', '.', text)
    
    return text


def preprocess_text(text):
    # 1. 양쪽 공백 제거
    text = text.strip()

    # 2. 노이즈 제거
    text = remove_noise(text)

    # 3. 연속된 공백 → 단일 공백
    text = re.sub(r'\s+', ' ', text)

    # 4. 반복 문자 축약
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 5. 중복 어절 제거
    words = text.split()
    cleaned_words = []
    prev_word = None
    repeat_count = 0

    for word in words:
        if word == prev_word:
            repeat_count += 1
            if repeat_count < 2:
                cleaned_words.append(word)
        else:
            cleaned_words.append(word)
            repeat_count = 0
        prev_word = word

    text = ' '.join(cleaned_words)

    return text

if __name__ == '__main__':
    # 테스트
    print("="*70)
    print("텍스트 전처리 테스트")
    print("="*70)

    # 전처리 테스트
    test_text = "안녕하세요!!!    저는   학생입니다.    ㅋㅋㅋㅋㅋ  정말 정말 정말 좋아요"

    print(f"\n[전처리]")
    print(f"원본:")
    print(f"  '{test_text}'")
    print(f"\n전처리 후:")
    print(f"  '{preprocess_text(test_text)}'")

    print(f"\n적용된 단계:")
    print(f"  1. 양쪽 공백 제거")
    print(f"  2. 연속된 공백 → 단일 공백")
    print(f"  3. 연속된 문장부호 정리 (!!!!! → !)")
    print(f"  4. 반복 문자 축약 (ㅋㅋㅋㅋㅋ → ㅋㅋ)")
    print(f"  5. 중복 어절 제거 (동일 단어 3회 이상 반복)")
