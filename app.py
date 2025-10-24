import streamlit as st
import json
from openai import OpenAI
from pathlib import Path
import re
from config.env import OPENAI_API_KEY

# 페이지 설정
st.set_page_config(
    page_title="차트 QNA 생성기",
    page_icon="📊",
    layout="wide"
)

# 기본 프롬프트 템플릿
DEFAULT_PROMPT = """역할:
당신은 데이터 분석 전문가로서, 입력으로 주어지는 혼합형 차트(막대+선형 등 복수 지표)의 수치를 근거로 정량적 인사이트를 도출하는 질문-추론-답변 세트 3개를 작성합니다.

[출력 조건]
1. 출력은 반드시 질문 / reasoning 3단계 / 답변 구조로 구성
2. 질문은 정확히 3개만 생성하며, 각 질문은 서로 다른 reasoning_subtype을 사용
3. ⭐ 논리추론 2개 + 연산추론 1개 구성 (논리추론 우선, 연산추론은 보충 용도)
4. 하나의 질문에는 하나의 답변 포인트만 포함 (복합질문 금지)
5. 모든 수치는 반드시 차트 내 실제 값으로 검증 가능해야 함
6. 외부 지식, 추측, 감정적 단어 사용 금지 (일반 상식은 허용)
7. 문장은 완전한 서술형으로 작성

[혼합형 차트 데이터 구조]
입력 데이터는 배열 형식으로 제공되며, 각 요소는 하나의 지표를 나타냅니다.
- 동일한 category(X축)를 공유
- 각 요소는 서로 다른 legend(지표명)와 unit(단위)를 가짐
- 첫 번째 요소는 주로 막대 차트 (절댓값), 두 번째는 선형 차트 (비율/추세)

예시:
[
  {{
    "chart_type": "혼합형",
    "chart_subtype": "막대형+선형",
    "title": "발주시기별 예산 현황",
    "legend": ["발주금액"],
    "unit": "억원",
    "category": ["1분기", "2분기", "3분기", "4분기"],
    "data_label": [["2,071.1", "1,692.8", "696.7", "575.9"]]
  }},
  {{
    "chart_type": "혼합형",
    "chart_subtype": "막대형+선형",
    "title": "발주시기별 예산 현황",
    "legend": ["예산비율"],
    "unit": "%",
    "category": ["1분기", "2분기", "3분기", "4분기"],
    "data_label": [["41.1", "33.6", "13.8", "11.4"]]
  }}
]

[질문 유형 중 3개 선택]
1. 변화 속도형 - 변화 폭, 비율, 속도 등 비교
2. 단위당 영향형 - 한 지표 변화가 다른 지표에 미치는 영향
3. 괴리·전환형 - 두 지표 간 전환 시점 분석
4. 주기·패턴형 - 변동 주기, 집중도, 반복 패턴 분석

[reasoning 작성 규칙]
모든 유형은 아래 3단계 구조를 따릅니다:

1단계(관찰): 차트의 실제 수치를 명시하고, 증감 폭·비율을 정량적으로 기술한다.
2단계(해석): 수치 간 관계나 속도를 계산하고 의미를 도출한다.
3단계(결론): 분석 결과를 정리해 한 문장으로 요약한다.


※ reasoning 배열의 각 요소는 단계별 내용만 작성 (①, ②, ③ 같은 번호나 라벨 불필요)

[reasoning_type 및 subtype 정의]
reasoning_type: "논리추론" 또는 "연산추론" (띄어쓰기 없음)

논리추론 subtypes:
  - 비교, 상관/관계, 귀납/패턴, 예외/특이점, 필터링/조건선별

연산추론 subtypes:
  - 증가량, 감소량, 합계, 평균, 증가율/증감률, 배수, 차이, 비중

[질문 유형별 reasoning 기준]

논리추론 reasoning 단계:
- 비교: 비교 대상 명시 → 값의 우열/차이 판별 → 차트 내 특징/패턴 언급
- 상관/관계: 비교 변수 명시 → 동반 변화/방향 판별 → 관계 유형 서술
- 귀납/패턴: 전체 변화 흐름 포착 → 주요 구간/항목 특징 → 주요 패턴 요약
- 예외/특이점: 특이값/예외 확인 → 예외 위치/특징 구체화 → 차트 맥락 내 의미
- 필터링/조건선별: 선별 기준 명시 → 조건 충족 항목 제시 → 분포/특징 요약

연산추론 필수 포함 수식:
- 증가량: 나중 값 − 기준 값
- 감소량: 기준 값 − 나중 값
- 합계: 값₁ + 값₂ + ... + 값ₙ
- 평균: (값₁ + ... + 값ₙ) ÷ n
- 증가율/증감률: (증가량 ÷ 기준 값) × 100
- 배수: 비교 값 ÷ 기준 값
- 차이: 큰 값 − 작은 값
- 비중(최댓값/최솟값): max(값₁, ...)/min(값₁, ...)

출력 예시 (실제 데이터 기반, 논리추론 2개 + 연산추론 1개):
{{
  "qa_reasoning": [
    {{
      "qa_id": 1,
      "question": "예산과 예산비율 중 어느 지표의 감소 속도가 더 급격한가?",
      "reasoning_type": "논리추론",
      "reasoning_subtype": "비교",
      "reasoning": [
        "예산은 1분기 2,071.1억 원에서 4분기 575.9억 원으로 72.2% 감소했고, 예산비율은 1분기 41.1%에서 4분기 11.4%로 72.3% 감소했다.",
        "두 지표의 감소율을 비교하면 예산비율이 72.3%로 예산의 72.2%보다 약간 더 크며, 특히 2→3분기 구간에서 예산비율은 19.8%p 감소로 상대적으로 더 급격한 하락을 보인다.",
        "따라서 전체적으로 예산비율의 감소 속도가 예산보다 미세하게 더 급격하다."
      ],
      "answer": "예산과 예산비율 모두 약 72% 감소하지만, 예산비율이 72.3%로 미세하게 더 급격한 감소를 보인다."
    }},
    {{
      "qa_id": 2,
      "question": "발주시기별 예산비율 변동 패턴은 어떤 시기에 집중되는가?",
      "reasoning_type": "논리추론",
      "reasoning_subtype": "귀납/패턴",
      "reasoning": [
        "예산비율은 1분기 41.1%, 2분기 33.6%로 상반기에만 74.7%를 차지하며, 이후 3분기 13.8%, 4분기 11.4%로 하반기에 25.2%에 불과하다.",
        "연간 예산의 약 3분의 2 이상이 상반기 발주에 몰려 있으며, 특히 1분기가 전체의 41.1%로 가장 높은 비중을 차지한다.",
        "이는 사업이 초기 단계에 집중되는 상반기 집행형 패턴으로, 연말 집행 효율성 저하 가능성을 시사한다."
      ],
      "answer": "전체 예산의 약 75%가 상반기에 발주되어 사업이 연초에 집중되는 상반기 편중형 패턴을 보인다."
    }},
    {{
      "qa_id": 3,
      "question": "예산이 100억 원 감소할 때 예산비율은 얼마나 줄어드는가?",
      "reasoning_type": "연산추론",
      "reasoning_subtype": "감소량",
      "reasoning": [
        "1분기에서 4분기까지 예산은 2,071.1억 원에서 575.9억 원으로 1,495.2억 원 감소했고, 같은 기간 예산비율은 41.1%에서 11.4%로 29.7포인트 감소했다.",
        "예산 100억 원 감소당 예산비율은 약 1.99포인트(29.7÷14.95) 줄어드는 셈이다.",
        "따라서 예산이 100억 원 줄면 예산비율은 약 2포인트 감소하는 것으로 볼 수 있다."
      ],
      "answer": "예산이 100억 원 줄어들 때마다 예산비율은 약 2포인트가량 감소하는 것으로 나타났다."
    }}
  ]
}}

**꼭 지킬 점:**
- 두 지표 간 관계를 반드시 분석 (단일 지표만 분석 금지)
- 각 지표의 legend와 unit을 정확히 구분하여 사용
- ⭐ 하나의 질문에는 하나의 질문 요소만 포함 (예: "어떻게 변화하며, 어느 구간에서..." 같은 복합질문 금지)
- reasoning은 정확히 3단계로 구성 (번호나 라벨 없이 내용만 작성)
- reasoning_type은 "논리추론", "연산추론" (띄어쓰기 없음)
- 모든 계산은 차트 데이터에서 직접 확인 가능해야 함
- JSON 형식을 정확히 준수

차트 데이터:
{chart_json}
"""

# 테스트 데이터 예시
TEST_DATA = """[
  {
    "chart_type": "혼합형",
    "chart_subtype": "막대형+선형",
    "title": "발주시기별 예산 현황",
    "legend": ["발주금액"],
    "unit": "억원",
    "category": ["1분기", "2분기", "3분기", "4분기"],
    "data_label": [["2,071.1", "1,692.8", "696.7", "575.9"]]
  },
  {
    "chart_type": "혼합형",
    "chart_subtype": "막대형+선형",
    "title": "발주시기별 예산 현황",
    "legend": ["예산비율"],
    "unit": "%",
    "category": ["1분기", "2분기", "3분기", "4분기"],
    "data_label": [["41.1", "33.6", "13.8", "11.4"]]
  }
]"""

def clean_and_fix_json(response_text: str) -> str:
    """OpenAI 응답에서 JSON을 정리하고 수정하는 함수"""
    try:
        # 1. 코드 블록 제거
        text = response_text.strip()
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]

        if text.endswith('```'):
            text = text[:-3]

        text = text.strip()

        # 2. 여러 줄 주석 제거
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

        # 3. 한 줄 주석 제거
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            in_string = False
            escape_next = False
            i = 0
            while i < len(line):
                if escape_next:
                    escape_next = False
                elif line[i] == '\\':
                    escape_next = True
                elif line[i] == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string and i < len(line) - 1 and line[i:i+2] == '//':
                    line = line[:i]
                    break
                i += 1
            cleaned_lines.append(line.rstrip())

        text = '\n'.join(cleaned_lines)

        # 4. 퍼센트 관련 표기 정리
        text = re.sub(r'%p\b', '%', text)

        # 5. 후행 쉼표 제거
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        # 6. JSON 유효성 테스트
        json.loads(text)
        return text

    except Exception as e:
        raise ValueError(f"유효하지 않은 JSON 형식: {str(e)}")

def generate_qna(prompt_template: str, chart_data: str, api_key: str, model: str = "gpt-4.1") -> dict:
    """QNA 생성 함수"""
    try:
        # 차트 데이터를 JSON으로 파싱
        table_data = json.loads(chart_data)

        # OpenAI 클라이언트 초기화
        client = OpenAI(api_key=api_key)

        # 차트 데이터를 JSON 문자열로 변환
        chart_json = json.dumps(table_data, ensure_ascii=False, indent=2)

        # 프롬프트에 차트 데이터 삽입
        full_prompt = prompt_template.format(chart_json=chart_json)

        # API 호출
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3,
            max_tokens=2500
        )

        # 토큰 사용량 및 비용 계산
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens

        # GPT-4.1 비용: Input $2.00/1M tokens, Output $8.00/1M tokens
        input_cost = prompt_tokens * 2.00 / 1000000
        output_cost = completion_tokens * 8.00 / 1000000
        total_cost = input_cost + output_cost

        # 응답 텍스트 추출 및 정리
        response_text = response.choices[0].message.content.strip()

        # JSON 정리 및 수정
        cleaned_response = clean_and_fix_json(response_text)

        # qa_reasoning 부분만 추출
        qna_data = json.loads(cleaned_response)
        if "qa_reasoning" in qna_data:
            qna_list = qna_data["qa_reasoning"]
        else:
            qna_list = qna_data

        return {
            'success': True,
            'qna_data': qna_list,
            'usage': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'input_cost': input_cost,
                'output_cost': output_cost,
                'total_cost': total_cost
            }
        }

    except json.JSONDecodeError as e:
        return {
            'success': False,
            'error': f"JSON 파싱 오류: {str(e)}"
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"오류 발생: {str(e)}"
        }

# 메인 앱
st.title("📊 차트 QNA 생성기")
st.markdown("혼합형 차트 데이터에서 질문-추론-답변을 자동으로 생성합니다.")

# 사이드바 - 설정
with st.sidebar:
    st.header("⚙️ 설정")

    # 모델 선택
    model = st.selectbox(
        "모델 선택",
        ["gpt-4.1", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0
    )

    st.divider()
    st.markdown("### 📖 사용 방법")
    st.markdown("""
    1. 프롬프트를 수정하세요
    2. 차트 데이터를 입력하세요
    3. 'QNA 생성' 버튼을 클릭하세요
    """)

    st.divider()
    st.markdown("### ℹ️ 정보")
    if OPENAI_API_KEY:
        st.success("✅ API 키 로드 완료")
    else:
        st.error("❌ config/env.py에서 API 키를 찾을 수 없습니다")

# 메인 컨텐츠 - 2개 컬럼
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 입력")

    # 프롬프트 편집
    st.subheader("1. 프롬프트 템플릿")
    prompt_template = st.text_area(
        "프롬프트를 수정할 수 있습니다",
        value=DEFAULT_PROMPT,
        height=300,
        help="프롬프트 내에 {chart_json}이 포함되어야 합니다"
    )

    # 차트 데이터 입력
    st.subheader("2. 차트 데이터 (JSON)")
    chart_data_input = st.text_area(
        "차트 데이터를 JSON 형식으로 입력하세요",
        value=TEST_DATA,
        height=300,
        help="JSON 배열 형식으로 입력하세요"
    )

    # 생성 버튼
    if st.button("🚀 QNA 생성", type="primary", use_container_width=True):
        if not OPENAI_API_KEY:
            st.error("❌ config/env.py에서 OPENAI_API_KEY를 찾을 수 없습니다")
        elif "{chart_json}" not in prompt_template:
            st.error("프롬프트에 {chart_json} 플레이스홀더가 필요합니다")
        else:
            with st.spinner("QNA 생성 중..."):
                result = generate_qna(prompt_template, chart_data_input, OPENAI_API_KEY, model)
                st.session_state['result'] = result

with col2:
    st.header("📤 출력")

    if 'result' in st.session_state:
        result = st.session_state['result']

        if result['success']:
            st.success("✅ QNA 생성 완료!")

            # 비용 정보
            usage = result['usage']
            st.info(f"""
            **토큰 사용량**
            - 입력: {usage['prompt_tokens']:,} tokens (${usage['input_cost']:.6f})
            - 출력: {usage['completion_tokens']:,} tokens (${usage['output_cost']:.6f})
            - 합계: {usage['total_tokens']:,} tokens (${usage['total_cost']:.6f})
            """)

            # QNA 결과
            qna_data = result['qna_data']

            for i, qa in enumerate(qna_data, 1):
                with st.expander(f"Q{i}. {qa.get('question', 'N/A')}", expanded=True):
                    # QA 정보
                    st.markdown(f"**유형:** {qa.get('reasoning_type', 'N/A')} - {qa.get('reasoning_subtype', 'N/A')}")

                    # Reasoning 단계
                    st.markdown("**추론 과정:**")
                    reasoning = qa.get('reasoning', [])
                    if isinstance(reasoning, list):
                        for j, step in enumerate(reasoning, 1):
                            st.markdown(f"{j}. {step}")
                    else:
                        st.markdown(reasoning)

                    # 답변
                    st.markdown(f"**답변:** {qa.get('answer', 'N/A')}")

            # JSON 다운로드
            st.divider()
            json_str = json.dumps(qna_data, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 JSON 다운로드",
                data=json_str,
                file_name="qna_result.json",
                mime="application/json",
                use_container_width=True
            )

            # JSON 미리보기
            with st.expander("JSON 미리보기"):
                st.json(qna_data)
        else:
            st.error(f"❌ {result['error']}")
    else:
        st.info("👈 왼쪽에서 프롬프트와 데이터를 입력하고 하단에 'QNA 생성' 버튼을 클릭하세요")
