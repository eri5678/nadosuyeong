import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from pathlib import Path
import unicodedata
import io

# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="극지식물 최적 EC 농도 연구",
    layout="wide"
)

# 한글 폰트
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 유틸 함수 (한글 파일명 안전)
# =========================
def normalize(s):
    return unicodedata.normalize("NFC", s)

def find_file(data_dir: Path, target_name: str):
    target_norm = normalize(target_name)
    for f in data_dir.iterdir():
        if normalize(f.name) == target_norm:
            return f
    return None

# =========================
# 데이터 로딩
# =========================
@st.cache_data
def load_environment_data():
    data_dir = Path("data")
    env_data = {}
    for school in ["송도고", "하늘고", "아라고", "동산고"]:
        path = find_file(data_dir, f"{school}_환경데이터.csv")
        if path is not None:
            df = pd.read_csv(path)
            df["학교"] = school
            env_data[school] = df
    return env_data

@st.cache_data
def load_growth_data():
    data_dir = Path("data")
    path = find_file(data_dir, "4개교_생육결과데이터.xlsx")
    xls = pd.ExcelFile(path)
    return {sheet: xls.parse(sheet).assign(학교=sheet) for sheet in xls.sheet_names}

with st.spinner("📂 데이터 로딩 중..."):
    env_data = load_environment_data()
    growth_data = load_growth_data()

# =========================
# 회귀 모델용 데이터 (학교 평균 기반)
# =========================
rows = []
for school in env_data:
    if school in growth_data:
        rows.append({
            "학교": school,
            "EC": env_data[school]["ec"].mean(),
            "생중량": growth_data[school]["생중량(g)"].mean()
        })

reg_df = pd.DataFrame(rows)

x = reg_df["EC"].values
y = reg_df["생중량"].values

if len(x) >= 3:
    coef = np.polyfit(x, y, 2)
    model = np.poly1d(coef)
    x_line = np.linspace(min(x), max(x), 300)
    best_ec = x_line[np.argmax(model(x_line))]
else:
    model = lambda v: 1
    best_ec = float(np.mean(x))

# =========================
# 제목 / 사이드바
# =========================
st.title("🌱 극지식물 최적 EC 농도 연구")
schools = ["전체"] + list(env_data.keys())
selected_school = st.sidebar.selectbox("학교 선택", schools)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과", "🎮 EC 맞히기 게임", "🤖 스마트팜 시뮬레이터"]
)

# =========================
# TAB 4 : EC 맞히기 게임
# =========================
with tab4:
    st.subheader("🎯 EC 맞히기 게임")
    ec_guess = st.slider("EC 선택", float(min(x)), float(max(x)), float(np.mean(x)), 0.01)

    if st.button("결과 확인"):
        predicted = model(ec_guess)
        error = abs(ec_guess - best_ec) / best_ec * 100
        st.metric("예상 생중량", f"{predicted:.2f} g")
        st.metric("실제 최적 EC", f"{best_ec:.2f}")
        st.metric("오차율", f"{error:.1f}%")

# =========================
# TAB 5 : 스마트팜 시뮬레이터
# =========================
with tab5:
    st.subheader("🤖 미니 스마트팜 시뮬레이터")

    col1, col2, col3 = st.columns(3)
    temp = col1.slider("🌡️ 온도 (℃)", 5.0, 30.0, 18.0)
    hum = col2.slider("💧 습도 (%)", 30.0, 90.0, 60.0)
    ec = col3.slider("⚡ EC", float(min(x)), float(max(x)), float(best_ec), 0.01)

    ec_effect = model(ec) / model(best_ec) if model(best_ec) != 0 else 0
    temp_effect = max(0, 1 - abs(temp - 18) / 20)
    hum_effect = max(0, 1 - abs(hum - 60) / 60)

    growth_index = ec_effect * temp_effect * hum_effect * 100
    st.metric("🌱 예상 생육 지수", f"{growth_index:.1f} / 100")

    if "day" not in st.session_state:
        st.session_state.day = 0
        st.session_state.weight = 5.0

    def grow(d=1):
        for _ in range(d):
            st.session_state.weight += max(growth_index / 100, 0)
            st.session_state.day += 1

    colA, colB, colC = st.columns(3)
    if colA.button("+1일"): grow(1)
    if colB.button("+3일"): grow(3)
    if colC.button("+7일"): grow(7)

    st.metric("경과 일수", f"{st.session_state.day}일")
    st.metric("예상 생중량", f"{st.session_state.weight:.2f} g")

    fig = px.line(
        x=range(st.session_state.day + 1),
        y=np.linspace(5, st.session_state.weight, st.session_state.day + 1),
        labels={"x": "일(day)", "y": "생중량(g)"}
    )
    st.plotly_chart(fig, use_container_width=True)

    if st.button("🔄 초기화"):
        st.session_state.day = 0
        st.session_state.weight = 5.0
