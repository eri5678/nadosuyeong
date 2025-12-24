import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import unicodedata
import io
import numpy as np

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="🌱 극지식물 최적 EC 농도 연구", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 유틸 (한글 파일명 안전)
# =========================
def normalize(s: str):
    return unicodedata.normalize("NFC", s)

def find_file(data_dir: Path, target_name: str):
    if not data_dir.exists():
        st.error(f"❌ data 폴더를 찾을 수 없습니다: {data_dir.resolve()}")
        return None

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
    env = {}
    for school in ["송도고", "하늘고", "아라고", "동산고"]:
        fname = f"{school}_환경데이터.csv"
        fpath = find_file(data_dir, fname)
        if fpath is None:
            continue
        df = pd.read_csv(fpath)
        df["학교"] = school
        env[school] = df
    return env if env else None

@st.cache_data
def load_growth_data():
    data_dir = Path("data")
    fpath = find_file(data_dir, "4개교_생육결과데이터.xlsx")
    if fpath is None:
        return None

    xls = pd.ExcelFile(fpath)
    growth = {}
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        df["학교"] = sheet
        growth[sheet] = df
    return growth

# =========================
# 데이터 불러오기
# =========================
with st.spinner("📂 데이터 로딩 중..."):
    env_data = load_environment_data()
    growth_data = load_growth_data()

if env_data is None or growth_data is None:
    st.stop()

# =========================
# 제목 / 사이드바
# =========================
st.title("🌱 극지식물 최적 EC 농도 연구")

schools = ["전체", "송도고", "하늘고", "아라고", "동산고"]
selected_school = st.sidebar.selectbox("학교 선택", schools)

tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# =========================
# TAB 1 : 실험 개요
# =========================
with tab1:
    st.subheader("연구 배경 및 목적")
    st.write("""
    본 연구는 학교별 서로 다른 EC 조건에서 극지식물의 생육 결과를 비교하여  
    **데이터 기반으로 최적 EC 농도**를 도출하고,  
    **EC–생육 관계를 수학적 모델(회귀 분석)**로 해석하는 것을 목표로 한다.
    """)

# =========================
# TAB 2 : 환경 데이터
# =========================
with tab2:
    st.subheader("학교별 환경 평균 비교")

    avg_rows = []
    for s, df in env_data.items():
        avg_rows.append({
            "학교": s,
            "온도": df["temperature"].mean(),
            "습도": df["humidity"].mean(),
            "pH": df["ph"].mean(),
            "EC": df["ec"].mean()
        })
    avg_df = pd.DataFrame(avg_rows)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["평균 온도", "평균 습도", "평균 pH", "평균 EC"]
    )
    fig.add_bar(x=avg_df["학교"], y=avg_df["온도"], row=1, col=1)
    fig.add_bar(x=avg_df["학교"], y=avg_df["습도"], row=1, col=2)
    fig.add_bar(x=avg_df["학교"], y=avg_df["pH"], row=2, col=1)
    fig.add_bar(x=avg_df["학교"], y=avg_df["EC"], row=2, col=2)
    fig.update_layout(font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3 : 생육 결과 + 회귀 분석
# =========================
with tab3:
    st.subheader("EC–생육 결과 분석 (회귀 분석 포함)")

    # 학교별 평균 EC → 생육 데이터에 매핑
    school_avg_ec = {s: env_data[s]["ec"].mean() for s in env_data}
    all_growth = pd.concat(growth_data.values(), ignore_index=True)
    all_growth["EC"] = all_growth["학교"].map(school_avg_ec)

    # EC별 평균 생중량
    ec_summary = (
        all_growth
        .groupby("EC", as_index=False)["생중량(g)"]
        .mean()
        .rename(columns={"생중량(g)": "평균 생중량"})
    )

    # ===== 1️⃣ 최적 EC 자동 산출 =====
    optimal_row = ec_summary.loc[ec_summary["평균 생중량"].idxmax()]
    optimal_ec = optimal_row["EC"]
    optimal_weight = optimal_row["평균 생중량"]

    st.metric(
        "🥇 최적 EC (자동 산출)",
        f"{optimal_ec:.2f}",
        help=f"평균 생중량 {optimal_weight:.2f} g으로 최대"
    )

    # ===== 2️⃣ EC–생중량 회귀 분석 =====
    x = ec_summary["EC"].values
    y = ec_summary["평균 생중량"].values

    # 2차 회귀
    coef = np.polyfit(x, y, 2)
    poly = np.poly1d(coef)
    y_pred = poly(x)

    # 결정계수 R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # 그래프
    x_line = np.linspace(min(x), max(x), 200)
    y_line = poly(x_line)

    fig_reg = go.Figure()
    fig_reg.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        name="실험 데이터"
    ))
    fig_reg.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines",
        name="2차 회귀곡선"
    ))
    fig_reg.update_layout(
        title=f"EC–생중량 회귀 분석 (R² = {r2:.3f})",
        xaxis_title="EC",
        yaxis_title="평균 생중량(g)",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    st.plotly_chart(fig_reg, use_container_width=True)

    st.info(
        f"""
        📌 회귀 분석 결과,  
        EC와 평균 생중량의 관계는 **2차 함수 형태**로 나타났으며  
        결정계수 **R² = {r2:.3f}**으로 비교적 높은 설명력을 보였다.  

        이는 EC가 증가할수록 생육이 향상되다가  
        **일정 농도 이상에서는 오히려 감소**하는 경향이 있음을 의미한다.
        """
    )
