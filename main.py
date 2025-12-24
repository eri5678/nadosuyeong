import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# 한글 폰트 (Streamlit UI)
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
        file_name = f"{school}_환경데이터.csv"
        file_path = find_file(data_dir, file_name)

        if file_path is None:
            st.error(f"❌ 환경 데이터 파일을 찾을 수 없습니다: {file_name}")
            continue

        df = pd.read_csv(file_path)
        df["학교"] = school
        env_data[school] = df

    if not env_data:
        return None

    return env_data

@st.cache_data
def load_growth_data():
    data_dir = Path("data")
    file_path = find_file(data_dir, "4개교_생육결과데이터.xlsx")

    if file_path is None:
        st.error("❌ 생육 결과 엑셀 파일을 찾을 수 없습니다.")
        return None

    xls = pd.ExcelFile(file_path)
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
# 제목
# =========================
st.title("🌱 극지식물 최적 EC 농도 연구")

# =========================
# 사이드바
# =========================
schools = ["전체", "송도고", "하늘고", "아라고", "동산고"]
selected_school = st.sidebar.selectbox("학교 선택", schools)

# =========================
# TAB 구성
# =========================
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# =========================
# TAB 1 : 실험 개요
# =========================
with tab1:
    st.subheader("연구 배경 및 목적")
    st.write("""
    극지식물은 극한 환경에서도 생존 가능한 식물로,
    EC(전기전도도)는 생육에 매우 중요한 환경 요인이다.
    본 연구는 **학교별 서로 다른 EC 조건에서 생육 결과를 비교**하여
    **극지식물의 최적 EC 농도**를 도출하는 것을 목표로 한다.
    """)

    summary_rows = []
    total_count = 0
    temps, hums, ecs = [], [], []

    for school, df in env_data.items():
        summary_rows.append({
            "학교명": school,
            "EC 목표": round(df["ec"].mean(), 2),
            "개체수": len(growth_data.get(school, [])),
            "색상": school
        })
        temps.append(df["temperature"].mean())
        hums.append(df["humidity"].mean())
        ecs.append(df["ec"].mean())
        total_count += len(growth_data.get(school, []))

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("총 개체수", total_count)
    col2.metric("평균 온도", f"{sum(temps)/len(temps):.1f} ℃")
    col3.metric("평균 습도", f"{sum(hums)/len(hums):.1f} %")
    col4.metric("최적 EC", "2.0 (하늘고)")

# =========================
# TAB 2 : 환경 데이터
# =========================
with tab2:
    st.subheader("학교별 환경 평균 비교")

    avg_env = []
    for school, df in env_data.items():
        avg_env.append({
            "학교": school,
            "온도": df["temperature"].mean(),
            "습도": df["humidity"].mean(),
            "pH": df["ph"].mean(),
            "EC": df["ec"].mean()
        })
    avg_df = pd.DataFrame(avg_env)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["평균 온도", "평균 습도", "평균 pH", "평균 EC"]
    )

    fig.add_bar(x=avg_df["학교"], y=avg_df["온도"], row=1, col=1)
    fig.add_bar(x=avg_df["학교"], y=avg_df["습도"], row=1, col=2)
    fig.add_bar(x=avg_df["학교"], y=avg_df["pH"], row=2, col=1)
    fig.add_bar(x=avg_df["학교"], y=avg_df["EC"], row=2, col=2)

    fig.update_layout(
        height=600,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )

    st.plotly_chart(fig, use_container_width=True)

    if selected_school != "전체":
        df = env_data[selected_school]
        st.subheader(f"{selected_school} 환경 변화 추이")

        fig2 = px.line(df, x="time", y=["temperature", "humidity", "ec"])
        fig2.update_layout(
            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
        )
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📂 환경 데이터 원본"):
        for school, df in env_data.items():
            st.write(f"### {school}")
            st.dataframe(df)
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            st.download_button(
                f"{school} CSV 다운로드",
                data=buffer,
                file_name=f"{school}_환경데이터.csv",
                mime="text/csv"
            )

# =========================
# TAB 3 : 생육 결과
# =========================
with tab3:
    st.subheader("EC별 생육 결과 분석")

    all_growth = pd.concat(growth_data.values(), ignore_index=True)

    ec_weight = all_growth.groupby("학교")["생중량(g)"].mean()
    best_school = ec_weight.idxmax()

    st.metric(
        "🥇 최고 평균 생중량",
        f"{ec_weight.max():.2f} g",
        help=f"최고값: {best_school}"
    )

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["평균 생중량", "평균 잎 수", "평균 지상부 길이", "개체 수"]
    )

    fig.add_bar(x=all_growth["학교"], y=all_growth["생중량(g)"], row=1, col=1)
    fig.add_bar(x=all_growth["학교"], y=all_growth["잎 수(장)"], row=1, col=2)
    fig.add_bar(x=all_growth["학교"], y=all_growth["지상부 길이(mm)"], row=2, col=1)
    fig.add_bar(x=all_growth["학교"], y=all_growth["개체번호"], row=2, col=2)

    fig.update_layout(
        height=650,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True)

    fig_box = px.box(
        all_growth,
        x="학교",
        y="생중량(g)",
        points="all"
    )
    fig_box.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    st.plotly_chart(fig_box, use_container_width=True)

    fig_corr1 = px.scatter(
        all_growth,
        x="잎 수(장)",
        y="생중량(g)",
        color="학교"
    )
    fig_corr2 = px.scatter(
        all_growth,
        x="지상부 길이(mm)",
        y="생중량(g)",
        color="학교"
    )

    st.plotly_chart(fig_corr1, use_container_width=True)
    st.plotly_chart(fig_corr2, use_container_width=True)

    with st.expander("📂 생육 데이터 원본"):
        for school, df in growth_data.items():
            st.write(f"### {school}")
            st.dataframe(df)
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, engine="openpyxl")
            buffer.seek(0)
            st.download_button(
                f"{school} XLSX 다운로드",
                data=buffer,
                file_name=f"{school}_생육결과.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
