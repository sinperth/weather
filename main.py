import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 기온 데이터를 이용해 연평균기온의 변화 추세를 살펴보고, "
    "회귀 직선으로 미래의 기온을 예상해 봅니다."
)


# =========================
# 데이터 설정
# =========================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

BASE_YEAR = 1908
LAST_YEAR = 2025
MIN_DAYS = 300


# =========================
# 데이터 불러오기
# =========================
@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 날짜가 없는 행 제거
    df = df.dropna(subset=["날짜"])

    # 연도 만들기
    df["연도"] = df["날짜"].dt.year

    # 2025년까지만 사용
    df = df[df["연도"] <= LAST_YEAR]

    # 연도별 평균기온과 관측일수 계산
    annual = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일이 300일 이상인 해만 사용
    annual = annual[
        annual["관측일수"] >= MIN_DAYS
    ].copy()

    # 연평균기온 결측치 제거
    annual = annual.dropna(
        subset=["연평균기온"]
    )

    return annual


annual = load_data()


# =========================
# 전체 기간 회귀분석
# =========================

# 1908년부터 지난 연수
annual["지난연수"] = (
    annual["연도"] - BASE_YEAR
)

x_all = annual["지난연수"].to_numpy()
y_all = annual["연평균기온"].to_numpy()

# 회귀 직선
slope_all, intercept_all = np.polyfit(
    x_all,
    y_all,
    1
)

# 1년당 기울기를 100년당 기울기로 변환
slope_all_100 = slope_all * 100

# 상관계수
correlation = annual["연도"].corr(
    annual["연평균기온"]
)

# 전체 기간 정보
start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())
year_count = len(annual)


# =========================
# 최근 20년 회귀분석
# =========================

# 수업 기준 마지막 연도 2025년을 포함한 최근 20년
recent_start = LAST_YEAR - 19

recent = annual[
    (annual["연도"] >= recent_start) &
    (annual["연도"] <= LAST_YEAR)
].copy()

# 최근 20년도 같은 방식으로
# 1908년부터 지난 연수를 독립변수로 사용
recent["지난연수"] = (
    recent["연도"] - BASE_YEAR
)

x_recent = recent["지난연수"].to_numpy()
y_recent = recent["연평균기온"].to_numpy()

recent_slope, recent_intercept = np.polyfit(
    x_recent,
    y_recent,
    1
)

# 100년당 기울기
recent_slope_100 = recent_slope * 100


# =========================
# 핵심 결과
# =========================
st.header("📊 기온 상승 속도 비교")

st.write(
    "회귀 직선의 기울기를 **100년에 기온이 몇 ℃ 변하는가**로 "
    "바꾸어 비교했습니다."
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label=f"전체 기간 ({start_year}~{end_year})",
        value=f"{slope_all_100:.2f} ℃ / 100년"
    )

with col2:
    st.metric(
        label=f"최근 20년 ({recent_start}~{LAST_YEAR})",
        value=f"{recent_slope_100:.2f} ℃ / 100년"
    )

st.caption(
    "최근 20년의 값도 비교를 위해 회귀선의 기울기를 "
    "100년 기준으로 환산한 값입니다."
)


# =========================
# 데이터 정보
# =========================
st.header("📌 분석에 사용한 데이터")

info1, info2, info3, info4 = st.columns(4)

info1.metric(
    "직선을 만든 해의 개수",
    f"{year_count}개"
)

info2.metric(
    "시작 연도",
    f"{start_year}년"
)

info3.metric(
    "끝 연도",
    f"{end_year}년"
)

info4.metric(
    "상관계수",
    f"{correlation:.3f}"
)

st.write(
    f"2025년 이후의 데이터와 연간 평균기온 관측일이 "
    f"{MIN_DAYS}일 미만인 해는 분석에서 제외했습니다."
)


# =========================
# 산점도 + 회귀 직선
# =========================
st.header("📈 연평균기온과 회귀 직선")

fig = go.Figure()


# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        customdata=annual["관측일수"],
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f} ℃<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>"
        )
    )
)


# =========================
# 전체 기간 회귀선
# =========================
all_line_years = np.arange(
    start_year,
    end_year + 1
)

all_line_x = (
    all_line_years - BASE_YEAR
)

all_line_y = (
    slope_all * all_line_x
    + intercept_all
)

fig.add_trace(
    go.Scatter(
        x=all_line_years,
        y=all_line_y,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(width=4),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "전체 기간 회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# =========================
# 최근 20년 회귀선
# =========================
recent_line_years = np.arange(
    recent_start,
    LAST_YEAR + 1
)

recent_line_x = (
    recent_line_years - BASE_YEAR
)

recent_line_y = (
    recent_slope * recent_line_x
    + recent_intercept
)

fig.add_trace(
    go.Scatter(
        x=recent_line_years,
        y=recent_line_y,
        mode="lines",
        name="최근 20년 회귀선",
        line=dict(
            width=4,
            dash="dash"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "최근 20년 회귀선: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)


# 그래프 설정
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend_title="범례"
)

# 가로축에는 실제 연도를 표시
fig.update_xaxes(
    tickformat="d"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================
# 상관계수
# =========================
st.header("🔗 연도와 평균기온의 상관관계")

st.metric(
    "상관계수",
    f"{correlation:.3f}"
)

st.write(
    "상관계수가 양수이면 연도가 증가할수록 "
    "연평균기온도 높아지는 경향이 있다는 뜻입니다."
)


# =========================
# 연도별 기온 예측
# =========================
st.header("🔮 연도별 예상 기온")

selected_year = st.slider(
    "예상 기온을 확인할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 전체 기간 회귀선으로 예측
selected_x = selected_year - BASE_YEAR

predicted_temp = (
    slope_all * selected_x
    + intercept_all
)


# 크게 표시
st.markdown(
    f"""
    <div style="
        text-align:center;
        padding:35px;
        margin-top:15px;
        margin-bottom:20px;
        border-radius:20px;
        background-color:rgba(128,128,128,0.12);
    ">

        <div style="
            font-size:25px;
        ">
            {selected_year}년 예상 연평균기온
        </div>

        <div style="
            font-size:65px;
            font-weight:bold;
            margin-top:8px;
        ">
            {predicted_temp:.2f} ℃
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# 관측 범위 밖의 연도일 경우 안내
if (
    selected_year < start_year
    or selected_year > end_year
):
    st.warning(
        f"{selected_year}년은 실제 데이터를 사용한 기간 "
        f"({start_year}~{end_year}년) 밖입니다. "
        "따라서 전체 기간 회귀 직선을 연장해 계산한 예상값입니다."
    )


st.caption(
    "예상 기온은 과거 기온의 선형적인 변화 추세를 "
    "연장해서 계산한 값이며 실제 미래 기온과 다를 수 있습니다."
)
