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

# =========================
# 🔥 추가 TAB 구성 (기존 코드 아래에 이어서)
# =========================
all_growth = pd.concat(growth_data.values(), ignore_index=True)

school_avg_ec = {s: env_data[s]["ec"].mean() for s in env_data}
all_growth["EC"] = all_growth["학교"].map(school_avg_ec)

ec_summary = all_growth.groupby("EC", as_index=False)["생중량(g)"].mean()

x = ec_summary["EC"].values
y = ec_summary["생중량(g)"].values

coef = np.polyfit(x, y, 2)
model = np.poly1d(coef)

best_ec = ec_summary.loc[
    ec_summary["생중량(g)"].idxmax(), "EC"
]
tab4, tab5 = st.tabs(["🎮 EC 맞히기 게임", "🤖 스마트팜 시뮬레이터"])

# =========================
# TAB 4 : EC 맞히기 게임
# =========================
with tab4:
    st.subheader("🎯 EC 맞히기 게임")
    st.write("슬라이더로 EC를 조절하고, 해당 조건에서의 **예상 생중량**을 맞혀보세요!")

    st.image(
        "https://images.unsplash.com/photo-1582281298055-e25b84a30b0b",
        caption="극지 환경에서도 생육 가능한 식물",
        use_container_width=True
    )

    # EC-생중량 회귀 모델 생성
    all_growth = pd.concat(growth_data.values(), ignore_index=True)
    school_avg_ec = {s: env_data[s]["ec"].mean() for s in env_data}
    all_growth["EC"] = all_growth["학교"].map(school_avg_ec)

    ec_summary = all_growth.groupby("EC", as_index=False)["생중량(g)"].mean()

    x = ec_summary["EC"].values
    y = ec_summary["생중량(g)"].values
    coef = np.polyfit(x, y, 2)
    model = np.poly1d(coef)

    ec_guess = st.slider("EC 값을 선택하세요", float(min(x)), float(max(x)), float(np.mean(x)), 0.01)

    if st.button("🔍 결과 확인"):
        predicted = model(ec_guess)
        best_ec = ec_summary.loc[ec_summary["생중량(g)"].idxmax(), "EC"]
        error = abs(ec_guess - best_ec) / best_ec * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("예상 생중량", f"{predicted:.2f} g")
        col2.metric("실제 최적 EC", f"{best_ec:.2f}")
        col3.metric("오차", f"{error:.1f} %")

        if error < 5:
            st.success("🎉 거의 정답입니다! EC 감각이 뛰어나네요!")
        else:
            st.info("🙂 다시 한 번 도전해보세요!")

# =========================
# TAB 5 : 미니 스마트팜 시뮬레이터 (확장)
# =========================
with tab5:
    st.subheader("🤖 미니 스마트팜 시뮬레이터")
    st.write("환경 조건에 따라 **식물 상태 진단 + 시간 경과 생육 변화**를 확인해보세요.")

    st.image(
        "https://images.unsplash.com/photo-1581091012184-7c54ab7b2d66",
        caption="스마트팜 환경 제어 시스템",
        use_container_width=True
    )

    # -------------------------
    # 환경 입력
    # -------------------------
    col1, col2, col3 = st.columns(3)
    temp = col1.slider("🌡️ 온도 (℃)", 5.0, 30.0, 18.0)
    hum = col2.slider("💧 습도 (%)", 30.0, 90.0, 60.0)
    ec = col3.slider("⚡ EC", float(min(x)), float(max(x)), float(np.mean(x)), 0.01)

    # -------------------------
    # 환경 상태 진단
    # -------------------------
    st.markdown("### 🧠 환경 상태 분석")

    problems = []

    if temp < 15:
        problems.append("🌡️ 온도가 낮아 **대사 속도가 감소**하고 생장이 느려질 수 있습니다.")
    elif temp > 25:
        problems.append("🌡️ 온도가 높아 **호흡량 증가 → 에너지 소모**가 커질 수 있습니다.")

    if hum < 50:
        problems.append("💧 습도가 낮아 **증산 작용 증가 → 수분 부족**이 발생할 수 있습니다.")
    elif hum > 85:
        problems.append("💧 습도가 높아 **곰팡이·병해 발생 위험**이 있습니다.")

    if ec < best_ec - 0.3:
        problems.append("⚡ EC가 낮아 **양분 부족 → 잎·줄기 생장 저하**가 발생할 수 있습니다.")
    elif ec > best_ec + 0.3:
        problems.append("⚡ EC가 높아 **삼투 스트레스 → 뿌리 손상** 위험이 있습니다.")

    if not problems:
        st.success("✅ 현재 환경은 온도·습도·EC 모두 적정합니다!")
    else:
        for p in problems:
            st.warning(p)

    # -------------------------
    # 생육 지수 계산 (안전 보정)
    # -------------------------
    ec_effect = model(ec) / model(best_ec) if model(best_ec) != 0 else 0
    temp_effect = max(0, 1 - abs(temp - 18) / 20)
    hum_effect = max(0, 1 - abs(hum - 60) / 60)

    growth_index = max(ec_effect * temp_effect * hum_effect, 0)

    st.metric("🌱 현재 생육 적합도", f"{growth_index*100:.1f} / 100")

    # -------------------------
    # ⏳ 시간 경과 생육 시뮬레이션
    # -------------------------
    st.markdown("### ⏳ 시간 경과 생육 시뮬레이션")

    if "sim_day" not in st.session_state:
        st.session_state.sim_day = 0
        st.session_state.leaf = 2
        st.session_state.length = 30.0   # mm
        st.session_state.weight = 5.0    # g

    def grow_one_day():
        rate = max(growth_index, 0.1)

        st.session_state.sim_day += 1
        st.session_state.leaf += rate * 0.3
        st.session_state.length += rate * 2.0
        st.session_state.weight += rate * 0.5

    colA, colB, colC = st.columns(3)

    if colA.button("➕ 1일"):
        grow_one_day()
    if colB.button("➕ 3일"):
        for _ in range(3):
            grow_one_day()
    if colC.button("➕ 7일"):
        for _ in range(7):
            grow_one_day()

    # -------------------------
    # 결과 표시
    # -------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌿 식물 나이", f"{st.session_state.sim_day} 일")
    col2.metric("🍃 잎 개수", f"{int(st.session_state.leaf)} 장")
    col3.metric("📏 길이", f"{st.session_state.length:.1f} mm")
    col4.metric("⚖️ 생중량", f"{st.session_state.weight:.2f} g")

    # -------------------------
    # 성장 그래프
    # -------------------------
    days = np.arange(st.session_state.sim_day + 1)
    weights = np.linspace(5.0, st.session_state.weight, len(days))

    fig = px.line(
        x=days,
        y=weights,
        labels={"x": "경과 일수(day)", "y": "생중량(g)"},
        title="시간 경과에 따른 생중량 변화 (시뮬레이션)"
    )
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )

    st.plotly_chart(fig, use_container_width=True)

    if st.button("🔄 시뮬레이션 초기화"):
        del st.session_state.sim_day
        del st.session_state.leaf
        del st.session_state.length
        del st.session_state.weight
