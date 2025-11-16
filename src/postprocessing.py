from konlpy.tag import Okt
import re


def fix_summary_punctuation_and_format(summary):
    """
    생성된 요약의 문장부호와 형식 수정 - 한국어 요약문 특화 버전
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


def _get_key_nouns(dialogue, summary):
    """
    대화와 요약에서 핵심 명사를 추출하고 누락된 명사를 반환

    Args:
        dialogue: 원본 대화 텍스트
        summary: 요약 텍스트

    Returns:
        tuple: (key_nouns, missing_nouns)
            - key_nouns: 대화의 빈도 높은 상위 3개 명사 리스트
            - missing_nouns: 요약에 누락된 핵심 명사 리스트
    """
    from collections import Counter

    okt = Okt()

    # 대화에서 명사 추출 (1글자 제외)
    dialogue_nouns = [n for n in okt.nouns(dialogue) if len(n) > 1]

    # 빈도 높은 상위 3개 명사
    key_nouns = [n for n, _ in Counter(dialogue_nouns).most_common(3)]

    # 요약에 포함된 명사
    summary_nouns = set(okt.nouns(summary))

    # 누락된 핵심 명사
    missing = [n for n in key_nouns if n not in summary_nouns]

    return key_nouns, missing


def validate_key_nouns_coverage(summary, dialogue):
    """
    대화의 핵심 명사가 요약에 포함되었는지 검증하고 경고 출력

    Args:
        summary: 요약 텍스트
        dialogue: 원본 대화 텍스트

    Returns:
        str: 입력받은 summary를 그대로 반환
    """
    _, missing = _get_key_nouns(dialogue, summary)

    if missing:
        print(f"⚠️ 핵심 명사 누락: {missing}")

    return summary


def adjust_by_nouns(summary, dialogue):
    """
    명사 개수로 요약 적절성 판단
    """
    okt = Okt()
    d_nouns = okt.nouns(dialogue)
    s_nouns = okt.nouns(summary)

    # 대화 명사의 20% 정도가 요약에 있어야 함
    target_noun_count = len(set(d_nouns)) * 0.2
    actual_noun_count = len(set(s_nouns))

    if actual_noun_count < target_noun_count * 0.5:
        # 너무 적음 → 재생성 (min_length 늘림)
        return "too_short"

    return "ok"