import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

BASE_YEAR = 1908
LAST_CLASS_YEAR = 2025
MIN_OBSERVATION_DAYS = 300
RECENT_YEARS = 20


# --------------------------------------------------
# 데이터 불러오기 및 전처리
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 2025년까지만 사용
    df = df[df["연도"] <= LAST_CLASS_YEAR]

    # 연도별 평균기온과 관측일수
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )

    # 관측일 300일 미만 제외
    annual = annual[
        annual["관측일수"] >= MIN_OBSERVATION_DAYS
    ].copy()

    annual = annual.dropna(subset=["연평균기온"])

    # 전체 회귀용 독립 변수
    annual["1908년부터_지난_연수"] = annual["연도"] - BASE_YEAR

    return annual


annual = load_data()


# --------------------------------------------------
# 전체 기간 회귀
# --------------------------------------------------
x = annual["1908년부터_지난_연수"].to_numpy()
y = annual["연평균기온"].to_numpy()

slope, intercept = np.polyfit(x, y, 1)

# 1년당 기울기 → 100년당 변화량
slope_100 = slope * 100

annual["회귀예측기온"] = slope * x + intercept

correlation = annual["연도"].corr(annual["연평균기온"])

start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
year_count = len(annual)


# --------------------------------------------------
# 최근 20년 회귀
# --------------------------------------------------
recent_start_year = LAST_CLASS_YEAR - RECENT_YEARS + 1

recent = annual[
    (annual["연도"] >= recent_start_year)
    & (annual["연도"] <= LAST_CLASS_YEAR)
].copy()

recent_x = recent["연도"].to_numpy()
recent_y = recent["연평균기온"].to_numpy()

recent_slope, recent_intercept = np.polyfit(
    recent_x,
    recent_y,
    1
)

recent_slope_100 = recent_slope * 100


# --------------------------------------------------
# 화면
# --------------------------------------------------
st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연도별 평균기온을 이용해 기온 변화 추세를 살펴보고, "
    "선형 회귀로 선택한 연도의 예상 평균기온을 계산합니다."
)


# --------------------------------------------------
# 기울기 크게 표시
# --------------------------------------------------
st.subheader("🔥 서울의 기온은 얼마나 빠르게 오르고 있을까?")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label=f"전체 기간 ({start_year}~{end_year})",
        value=f"{slope_100:+.2f} ℃ / 100년",
    )

with col2:
    st.metric(
        label=f"최근 20년 ({recent_start_year}~{LAST_CLASS_YEAR})",
        value=f"{recent_slope_100:+.2f} ℃ / 100년",
    )

st.caption(
    "각 값은 연평균기온에 맞춘 회귀 직선의 기울기를 "
    "100년 단위로 바꾼 값입니다."
)


# --------------------------------------------------
# 데이터 정보
# --------------------------------------------------
st.subheader("📌 회귀 직선에 사용한 데이터")

c1, c2, c3, c4 = st.columns(4)

c1.metric("사용한 연도 수", f"{year_count}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
c4.metric("상관계수", f"{correlation:.3f}")

st.caption(
    f"2025년 이후 데이터와 연간 평균기온 관측일이 "
    f"{MIN_OBSERVATION_DAYS}일 미만인 해는 제외했습니다."
)


# --------------------------------------------------
# 산점도 + 전체 기간 회귀 직선
# --------------------------------------------------
st.subheader("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        customdata=annual["관측일수"],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "평균기온: %{y:.2f} ℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        ),
    )
)

# 전체 기간 회귀 직선
line_years = np.arange(start_year, end_year + 1)
line_x = line_years - BASE_YEAR
line_temperature = slope * line_x + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temperature,
        mode="lines",
        name="전체 기간 회귀 직선",
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "회귀선 기온: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)

# 최근 20년 회귀 직선
recent_line_years = np.arange(
    recent_start_year,
    LAST_CLASS_YEAR + 1
)

recent_line_temperature = (
    recent_slope * recent_line_years
    + recent_intercept
)

fig.add_trace(
    go.Scatter(
        x=recent_line_years,
        y=recent_line_temperature,
        mode="lines",
        name="최근 20년 회귀 직선",
        line=dict(width=4, dash="dash"),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "최근 20년 회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend_title="데이터",
)

fig.update_xaxes(
    tickformat="d",
    range=[start_year - 2, end_year + 2],
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 기울기 비교 설명
# --------------------------------------------------
st.subheader("🔎 기울기 비교")

comparison_col1, comparison_col2 = st.columns(2)

with comparison_col1:
    st.markdown("### 전체 기간")
    st.markdown(
        f"<div style='font-size:42px; font-weight:bold;'>"
        f"{slope_100:+.2f} ℃</div>"
        f"<div style='font-size:20px;'>100년당 변화</div>",
        unsafe_allow_html=True,
    )

with comparison_col2:
    st.markdown("### 최근 20년")
    st.markdown(
        f"<div style='font-size:42px; font-weight:bold;'>"
        f"{recent_slope_100:+.2f} ℃</div>"
        f"<div style='font-size:20px;'>100년당 변화</div>",
        unsafe_allow_html=True,
    )

st.info(
    "최근 20년의 값도 비교하기 쉽도록 100년당 변화량으로 "
    "환산한 것입니다. 실제로 최근 100년을 관측했다는 뜻은 아닙니다."
)


# --------------------------------------------------
# 연도 선택 및 기온 예측
# --------------------------------------------------
st.subheader("🔮 연도별 기온 예측")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
)

selected_x = selected_year - BASE_YEAR
predicted_temperature = slope * selected_x + intercept

st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 30px;
        border-radius: 15px;
        background-color: rgba(128, 128, 128, 0.12);
        margin-top: 15px;
        margin-bottom: 20px;
    ">
        <div style="font-size:24px;">
            {selected_year}년 예상 연평균기온
        </div>

        <div style="
            font-size:60px;
            font-weight:bold;
            margin-top:5px;
        ">
            {predicted_temperature:.2f} ℃
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if selected_year < start_year or selected_year > end_year:
    st.warning(
        f"{selected_year}년은 회귀 직선에 사용한 관측 기간 "
        f"({start_year}~{end_year}년) 밖이므로 "
        "전체 기간 회귀 직선을 연장한 추정값입니다."
    )

st.caption(
    "예상 기온은 과거 서울 기온의 선형 추세를 단순히 연장한 값입니다."
)
