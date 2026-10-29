import os
import json
import re
import io
import requests
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ReportLab (PDF 생성)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 환경변수 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==========================================
# 1. PDF 내 매립 원본 JSON 추출 모듈
# ==========================================
def extract_embedded_json(pdf_bytes: bytes) -> dict:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    json_data = None

    # 첨부파일(Embedded File) 탐색
    for i in range(doc.embfile_count()):
        file_info = doc.embfile_info(i)
        filename = file_info.get("filename", "")
        if filename.endswith(".json") or "json" in filename.lower():
            raw_bytes = doc.embfile_get(i)
            json_data = json.loads(raw_bytes.decode("utf-8"))
            break

    # Catalog Stream 탐색 (Fallback)
    if not json_data:
        for name in doc.embfile_names():
            if "json" in name.lower():
                raw_bytes = doc.embfile_get(name)
                json_data = json.loads(raw_bytes.decode("utf-8"))
                break

    if not json_data:
        raise ValueError("업로드된 PDF 파일 내부에서 매립된 JSON 데이터를 찾지 못했습니다.")

    return json_data


# ==========================================
# 2. 스윙 데이터 요약 및 정제
# ==========================================
def preprocess_swing_data(json_data: dict) -> dict:
    metadata = json_data.get("metadata", {})
    phases = json_data.get("swing_phases", [])
    
    total_match = sum(p.get("overall_match_rate", 0) for p in phases) / len(phases) if phases else 0
    
    issues = []
    for phase in phases:
        p_name = phase.get("phase_name", "")
        match_rate = phase.get("overall_match_rate", 0.0)
        
        cop_analysis = phase.get("cop_axis_analysis", {})
        cop_error = cop_analysis.get("cop_error", {})
        dx = cop_error.get("delta_x", 0.0)
        dy = cop_error.get("delta_y", 0.0)

        if any(k in p_name for k in ["Transition", "Downswing", "Impact"]):
            if dx >= 0.25:
                issues.append({
                    "phase": f"{p_name} (다운스윙/임팩트)",
                    "match_rate": match_rate,
                    "description": "임팩트 및 전환 순간 중심 이동 오차가 발생하여 지지벽 형성이 보완될 필요가 있음"
                })
        elif "Backswing" in p_name:
            if dx >= 0.20 or dy >= 0.20:
                issues.append({
                    "phase": f"{p_name} (백스윙)",
                    "match_rate": match_rate,
                    "description": "백스윙 구간에서 중심축 체중 이동 밸런스 점검 필요"
                })

    raw_swing_type = metadata.get("swing_type", "스윙")
    clean_swing_type = raw_swing_type.replace("/", "_").strip()

    return {
        "user_name": metadata.get("user_name", "회원"),
        "swing_type": clean_swing_type,
        "measurement_date": metadata.get("measurement_date", "2026-01-09"),
        "avg_match_rate": round(total_match, 1),
        "issues": issues
    }


# ==========================================
# 3. Gemini AI 소견 생성 모듈
# ==========================================
def generate_easy_ai_opinion(summary_data: dict, api_key: str) -> str:
    client = genai.Client(api_key=api_key)

    system_prompt = """
너는 파크골프 회원을 위한 친절하고 전문적인 AI 스윙 코치이다.
[작성 지침]
1. 일반인 사용자가 읽기 쉽고 이해하기 쉽게 간결하고 명확하게 작성하라.
2. 'COP', 'dx', 'dy', '좌표', '오차' 같은 전문 용어나 기술적 표현은 절대로 사용하지 마라. 대신 '체중 중심', '양발 밸런스', '왼발 지지'와 같은 직관적인 단어만 사용하라.
3. 마크다운 기호(###, **, ## 등)는 절대로 출력하지 마라.
4. 다음 3개 섹션 제목을 명확히 구분하여 작성하라:
   [1. 스윙 종합 평가 및 원인]
   [2. 추천 연습 방법 (2가지)]
   [3. AI 코치 메시지]
"""

    issues_str = ""
    if not summary_data["issues"]:
        issues_str = "전반적으로 양발의 체중 밸런스와 지지력이 매우 안정적입니다."
    else:
        for item in summary_data["issues"]:
            issues_str += f"- {item['phase']}: {item['description']} (일치율 {item['match_rate']}%)\n"

    user_prompt = f"""
- 회원명: {summary_data['user_name']}
- 스윙종류: {summary_data['swing_type']}
- 평균 일치율: {summary_data['avg_match_rate']}%

[감지된 체중 중심 특징]
{issues_str}

위 내용을 바탕으로 사용자가 바로 이해할 수 있는 코칭 소견을 작성해 주세요.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.6,
        )
    )

    return response.text


# ==========================================
# 4. 시각적 PDF 소견서 생성 모듈 (utf-8 인코딩 명시)
# ==========================================
def create_pdf_report(summary_data: dict, ai_opinion_text: str, output_path: str):
    main_font = "NotoSansKR"

    # 폰트 등록 여부 확인 후 메모리로 가져와 동적 등록
    if main_font not in pdfmetrics.getRegisteredFontNames():
        try:
            # 오픈소스 Noto Sans KR 폰트 다운로드
            font_url = "https://github.com/google/fonts/raw/main/ofl/notosanskr/NotoSansKR-Regular.ttf"
            res = requests.get(font_url, timeout=10)
            res.raise_for_status()
            font_bytes = io.BytesIO(res.content)
            
            # ★ 핵심 수정: utf-8 인코딩 옵션 추가 ★
            pdfmetrics.registerFont(TTFont(main_font, font_bytes, utf8=True))
        except Exception as e:
            # 온라인 다운로드 실패 시 시스템 맑은 고딕 Fallback
            font_path = "C:/Windows/Fonts/malgun.ttf"
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont("Malgun", font_path, utf8=True))
                main_font = "Malgun"
            else:
                main_font = "Helvetica"

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=30, leftMargin=30,
        topMargin=30, bottomMargin=30
    )

    styles = getSampleStyleSheet()

    primary_color = colors.HexColor('#0284C7')
    dark_navy = colors.HexColor('#0F172A')
    card_bg = colors.HexColor('#F8FAFC')
    border_color = colors.HexColor('#E2E8F0')
    text_color = colors.HexColor('#334155')

    # 스타일 정의
    title_style = ParagraphStyle('DocTitle', fontName=main_font, fontSize=20, leading=24, textColor=dark_navy)
    sub_title_style = ParagraphStyle('SubTitle', fontName=main_font, fontSize=9, leading=12, textColor=colors.HexColor('#64748B'))
    
    major_header_style = ParagraphStyle('MajorHeader', fontName=main_font, fontSize=15, leading=19, textColor=primary_color, spaceBefore=12, spaceAfter=8)
    body_style = ParagraphStyle('BodyTextCustom', fontName=main_font, fontSize=9.5, leading=15, textColor=text_color, spaceAfter=4)
    bullet_style = ParagraphStyle('BulletCustom', fontName=main_font, fontSize=9.5, leading=15, textColor=text_color, leftIndent=10, spaceAfter=4)

    elements = []

    # 헤더
    elements.append(Paragraph("<b>파크골프 AI 스윙 분석 소견서</b>", title_style))
    elements.append(Spacer(1, 2))
    elements.append(Paragraph("PediSol Biomechanics Swing AI Coaching Report", sub_title_style))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=10))

    # 요약 정보 표
    info_card_data = [
        [
            Paragraph(f"<b>회원명</b><br/><font size=10 color='#0F172A'><b>{summary_data['user_name']}</b> 님</font>", body_style),
            Paragraph(f"<b>스윙 타입</b><br/><font size=10 color='#0284C7'><b>{summary_data['swing_type']}</b></font>", body_style),
            Paragraph(f"<b>측정 일자</b><br/><font size=9 color='#334155'>{summary_data['measurement_date']}</font>", body_style),
            Paragraph(f"<b>체중 중심 일치율</b><br/><font size=11 color='#0284C7'><b>{summary_data['avg_match_rate']}%</b></font>", body_style)
        ]
    ]
    info_table = Table(info_card_data, colWidths=[130, 130, 135, 140])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), card_bg),
        ('BOX', (0, 0), (-1, -1), 1, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 10))

    # 소견 텍스트 정제 및 레이아웃 배치
    clean_text = re.sub(r'[\#\*]', '', ai_opinion_text)
    lines = clean_text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('[1.') or line.startswith('[2.') or line.startswith('[3.') or line.startswith('1.') or line.startswith('2.') or line.startswith('3.'):
            elements.append(Spacer(1, 8))
            elements.append(Paragraph(f"<b>{line}</b>", major_header_style))
            elements.append(HRFlowable(width="100%", thickness=0.5, color=border_color, spaceAfter=6))
        elif line.startswith('-') or line.startswith('•'):
            clean_bullet = line.lstrip('-• ').strip()
            elements.append(Paragraph(f"• {clean_bullet}", bullet_style))
        else:
            elements.append(Paragraph(line, body_style))

    # 푸터
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceAfter=6))
    elements.append(Paragraph("<b>SPINA SYSTEMS</b> | 문의: 070-4801-0048 | 본 소견서는 PediSol 체중 중심 분석 시스템으로 생성되었습니다.", sub_title_style))

    doc.build(elements)


# ==========================================
# 5. Streamlit 웹 인터페이스 (UI)
# ==========================================
st.set_page_config(page_title="PediSol AI 파크골프 분석", page_icon="⛳", layout="wide")

st.title("⛳ PediSol 파크골프 AI 스윙 소견서 생성 시스템")
st.markdown("PDF 보고서를 업로드하면 매립된 원본 JSON을 추출하고, AI가 분석한 **맞춤형 PDF 소견서**를 즉시 생성합니다.")

# 사이드바 API Key 확인
with st.sidebar:
    st.header("⚙️ 설정")
    api_key_input = st.text_input("Gemini API Key", value=GEMINI_API_KEY or "", type="password")
    st.info("API 키는 `.env` 파일에서 자동으로 로드되거나 위 창에서 변경할 수 있습니다.")

uploaded_file = st.file_uploader("파크골프 스윙 분석 PDF 보고서를 선택해 주세요", type=["pdf"])

if uploaded_file is not None:
    if not api_key_input:
        st.error("Gemini API Key가 설정되지 않았습니다. .env 파일을 확인하거나 사이드바에 키를 입력해 주세요.")
    else:
        try:
            pdf_bytes = uploaded_file.read()
            
            with st.spinner("1. PDF 내 매립 원본 JSON 추출 중..."):
                raw_json = extract_embedded_json(pdf_bytes)
                summary = preprocess_swing_data(raw_json)

            st.success(f"✅ 원본 JSON 추출 성공! ({summary['user_name']} 님 / {summary['swing_type']})")

            # 좌우 2컬럼 레이아웃
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("📊 분석 요약 정보")
                st.json({
                    "회원명": summary["user_name"],
                    "스윙 타입": summary["swing_type"],
                    "측정 일자": summary["measurement_date"],
                    "전체 일치율": f"{summary['avg_match_rate']}%",
                    "감지된 문제점 수": len(summary["issues"])
                })

            with col2:
                st.subheader("🤖 AI 스윙 코칭 소견 생성")
                if st.button("AI 소견서 생성하기", type="primary"):
                    with st.spinner("Gemini AI가 소견서를 작성하고 PDF 보고서를 구성 중입니다..."):
                        ai_opinion = generate_easy_ai_opinion(summary, api_key_input)
                        
                        output_pdf_filename = f"{summary['user_name']}_{summary['swing_type']}_AI소견서.pdf"
                        create_pdf_report(summary, ai_opinion, output_pdf_filename)

                        st.markdown("### 📝 생성된 소견 내역")
                        st.write(ai_opinion)

                        with open(output_pdf_filename, "rb") as pdf_file:
                            st.download_button(
                                label="📄 PDF 소견서 다운로드",
                                data=pdf_file,
                                file_name=output_pdf_filename,
                                mime="application/pdf"
                            )
                        st.success("🎉 PDF 소견서 파일 생성이 완료되었습니다!")

        except Exception as e:
            st.error(f"오류 발생: {e}")
