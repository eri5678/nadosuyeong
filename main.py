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
st.set_page_config(
    page_title="🌱 극지식물 최적 EC 농도 연구",
    layout="wide"
)

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
        fname = f"{school}_환경데이터.csv"
        fpath = find_file(data_dir, fname)
        if fpath is None:
            st.error(f"❌ 환경 데이터 파일 없음: {fname}")
            continue

        df = pd.read_csv(fpath)
        df["학교"] = school
        env_data[school] = df

    return env_data if env_data else None

@st.cache_data
def load_growth_data():
    data_dir = Path("data")
    fpath = find_file(data_dir, "4개교_생육결과데이터.xlsx")

    if fpath is None:
        st.error("❌ 생육 결과 파일을 찾을 수 없습니다.")
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

tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과 & 분석"])

# =========================
# TAB 1 : 실험 개요
# =========================
with tab1:
    st.subheader("연구 목적")
    st.write("""
    본 연구는 학교별로 상이한 EC 조건에서 재배된 극지식물의
    생육 데이터를 비교·분석하여 **최적 EC 농도**를
    **데이터 기반으로 도출**하는 것을 목표로 한다.
    """)

    rows, temps, hums, ecs, total = [], [], [], [], 0

    for school, df in env_data.items():
        rows.append({
            "학교": school,
            "평균 EC": round(df["ec"].mean(), 2),
            "개체 수": len(growth_data.get(school, []))
        })
        temps.append(df["temperature"].mean())
        hums.append(df["humidity"].mean())
        ecs.append(df["ec"].mean())
        total += len(growth_data.get(school, []))

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 개체 수", total)
    c2.metric("평균 온도", f"{np.mean(temps):.1f} ℃")
    c3.metric("평균 습도", f"{np.mean(hums):.1f} %")
    c4.metric("평균 EC", f"{np.mean(ecs):.2f}")

# =========================
# TAB 2 : 환경 데이터
# =========================
with tab2:
    st.subheader("학교별 환경 평균")

    avg = []
    for s, df in env_data.items():
        avg.append({
            "학교": s,
            "온도": df["temperature"].mean(),
            "습도": df["humidity"].mean(),
            "pH": df["ph"].mean(),
            "EC": df["ec"].mean()
        })
    avg_df = pd.DataFrame(avg)

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["온도", "습도", "pH", "EC"])
    fig.add_bar(x=avg_df["학교"], y=avg_df["온도"], row=1, col=1)
    fig.add_bar(x=avg_df["학교"], y=avg_df["습도"], row=1, col=2)
    fig.add_bar(x=avg_df["학교"], y=avg_df["pH"], row=2, col=1)
    fig.add_bar(x=avg_df["학교"], y=avg_df["EC"], row=2, col=2)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 3 : 생육 결과 + 고급 분석
# =========================
with tab3:
    all_growth = pd.concat(growth_data.values(), ignore_index=True)

    # 학교 평균 EC 매핑
    school_avg_ec = {s: env_data[s]["ec"].mean() for s in env_data}
    all_growth["EC"] = all_growth["학교"].map(school_avg_ec)

    # EC별 평균 생중량
    ec_summary = (
        all_growth
        .groupby("EC", as_index=False)["생중량(g)"]
        .mean()
        .rename(columns={"생중량(g)": "평균 생중량"})
    )

    # ===== 최적 EC 자동 산출 =====
    best = ec_summary.loc[ec_summary["평균 생중량"].idxmax()]
    st.metric(
        "🥇 최적 EC",
        f"{best['EC']:.2f}",
        help=f"평균 생중량 {best['평균 생중량']:.2f} g"
    )

    # ===== 회귀 분석 =====
    x = ec_summary["EC"].values
    y = ec_summary["평균 생중량"].values

    coef = np.polyfit(x, y, 2)
    poly = np.poly1d(coef)
    y_pred = poly(x)

    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

    x_line = np.linspace(min(x), max(x), 200)
    y_line = poly(x_line)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="실험값"))
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="2차 회귀"))
    fig.update_layout(
        title=f"EC–생중량 회귀 분석 (R² = {r2:.3f})",
        xaxis_title="EC",
        yaxis_title="평균 생중량(g)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"""
        EC와 생중량의 관계는 **2차 함수 형태**로 나타났으며  
        결정계수 **R² = {r2:.3f}**으로 EC가 생육에
        유의미한 영향을 미침을 확인하였다.
        """
    )
