"""
텍스트 전처리 모듈
한국어 대화 요약을 위한 전처리 함수들
"""

import re


def preprocess_text(text):
    """
    텍스트 전처리 - 모든 전처리 단계를 순차적으로 적용

    Args:
        text: 입력 텍스트

    Returns:
        str: 전처리된 텍스트
    """
    # 1. 양쪽 공백 제거
    text = text.strip()

    # 2. 연속된 공백 → 단일 공백
    text = re.sub(r'\s+', ' ', text)

    # 3. 연속된 문장부호 정리 (예: !!!!! → !)
    text = re.sub(r'([.!?])\1+', r'\1', text)

    # 4. 반복 문자 축약 (예: ㅋㅋㅋㅋㅋ → ㅋㅋ)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 5. 중복 어절 제거 (동일 단어 3회 이상 반복)
    words = text.split()
    cleaned_words = []
    prev_word = None
    repeat_count = 0

    for word in words:
        if word == prev_word:
            repeat_count += 1
            if repeat_count < 2:  # 최대 2번까지만 허용
                cleaned_words.append(word)
        else:
            cleaned_words.append(word)
            repeat_count = 0
        prev_word = word

    text = ' '.join(cleaned_words)

    return text

def postprocess_summary(summary):
    """
    생성된 요약 후처리 - 한국어 요약문 특화 버전
    """
    if not summary:
        return summary
    
    # 1. 기본 정리
    summary = re.sub(r'[^\w\s가-힣.,!?~\-()\"\']+', '', summary)
    summary = re.sub(r'\s+', ' ', summary)
    summary = re.sub(r'\s+([.,!?])', r'\1', summary)
    summary = re.sub(r'([.!?]){2,}', r'\1', summary)
    summary = summary.strip()
    
    if not summary:
        return summary
    
    # 2. 요약문은 거의 항상 "~다"로 끝남
    # 문장부호로 끝나는지 확인
    if not re.search(r'[.!?]$', summary):
        # 마지막 문장부호 찾기
        last_punct_pos = max(
            summary.rfind('.'),
            summary.rfind('!'),
            summary.rfind('?')
        )
        
        if last_punct_pos > len(summary) * 0.3:
            # 마지막 완성된 문장까지만 자르기
            summary = summary[:last_punct_pos + 1].strip()
        else:
            # 요약문 전용 종결어미 패턴 (간결하게)
            # "~다" 계열만 체크
            summary_endings = r'(?:한다|있다|없다|된다|받다|같다|말한다|이야기한다|논의한다|계획한다|약속한다|결정한다|했다|갔다|왔다|봤다|먹었다|만났다|주었다|받았다|보냈다)$'
            
            if re.search(summary_endings, summary):
                summary += '.'
            else:
                # "~다"로 안 끝나면 불완전한 요약
                print(f"⚠️ Warning: Incomplete summary: '{summary}'")
                # 경고만 하고 그대로 반환 (ROUGE 평가를 위해)
    
    return summary

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

    # 후처리 테스트
    print(f"\n{'='*70}")
    print("[후처리 테스트]")
    print(f"{'='*70}")

    test_summaries = [
        "오늘 날씨가 정말 좋은데요   !!!  ",
        "친구와 영화를 봤어요. 정말 재미있었고",  # 불완전한 문장
        "맛있는 음식을 먹었다. 기분이",  # 끊긴 문장
        "운동을 열심히 합니다",  # 완전한 문장
        "학교에 갔어요. 수업을 들었습니다.",  # 완전한 복수 문장
        "책을 읽고 있는",  # 불완전
    ]

    for i, test_summary in enumerate(test_summaries, 1):
        print(f"\n테스트 {i}:")
        print(f"  원본: '{test_summary}'")
        print(f"  처리: '{postprocess_summary(test_summary)}'")
