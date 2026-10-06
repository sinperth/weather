import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 회귀 모델",
    page_icon="📈",
    layout="wide"
)

st.title("📈 기온 회귀 모델 평가")
st.write(
    "과거 연평균 기온을 훈련데이터로 사용해 선형회귀 모델을 만들고, "
    "최근 20년의 기온을 얼마나 잘 예측하는지 평가합니다."
)


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    df["연도"] = df["날짜"].dt.year

    # 연도별 연평균기온과 관측일수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 2025년까지만 사용하고,
    # 1년의 관측일수가 300일 이상인 연도만 사용
    yearly = yearly[
        (yearly["연도"] <= 2025) &
        (yearly["관측일수"] >= 300)
    ].copy()

    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


yearly = load_data()


# --------------------------------------------------
# 데이터 확인
# --------------------------------------------------
st.subheader("📊 분석에 사용한 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("사용 연도 수", f"{len(yearly)}년")

with col2:
    st.metric("가장 오래된 연도", f"{yearly['연도'].min()}년")

with col3:
    st.metric("가장 최근 연도", f"{yearly['연도'].max()}년")


st.caption(
    "연도별 평균기온을 계산한 뒤, 관측일수가 300일 이상인 연도만 분석에 사용했습니다."
)


# --------------------------------------------------
# 훈련 / 테스트 데이터 나누기
# --------------------------------------------------
train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()

train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()

test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


# --------------------------------------------------
# 필요한 데이터가 있는지 확인
# --------------------------------------------------
if len(train_50) == 0 or len(train_100) == 0 or len(test) == 0:
    st.error("분석에 필요한 연도 데이터가 충분하지 않습니다.")
    st.stop()


# --------------------------------------------------
# 선형회귀 함수
# --------------------------------------------------
def train_model(train_data):
    # 이전 분석과 맞추기 위해 1908년을 기준으로 연도 차이를 사용
    X = (train_data["연도"] - 1908).values.reshape(-1, 1)
    y = train_data["연평균기온"].values

    model = LinearRegression()
    model.fit(X, y)

    return model


def evaluate_model(model, test_data):
    X_test = (test_data["연도"] - 1908).values.reshape(-1, 1)
    y_test = test_data["연평균기온"].values

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return mae, mse, r2, predictions


# --------------------------------------------------
# 모델 학습
# --------------------------------------------------
model_50 = train_model(train_50)
model_100 = train_model(train_100)


# --------------------------------------------------
# 테스트 데이터 평가
# --------------------------------------------------
mae_50, mse_50, r2_50, pred_50 = evaluate_model(
    model_50,
    test
)

mae_100, mse_100, r2_100, pred_100 = evaluate_model(
    model_100,
    test
)


# --------------------------------------------------
# 기울기
# --------------------------------------------------
slope_50 = model_50.coef_[0]
slope_100 = model_100.coef_[0]

slope_50_100 = slope_50 * 100
slope_100_100 = slope_100 * 100


# --------------------------------------------------
# 학습 데이터 / 테스트 데이터 설명
# --------------------------------------------------
st.subheader("① 훈련 데이터와 테스트 데이터")

data_info = pd.DataFrame({
    "구분": [
        "최근 50년 훈련 데이터",
        "최근 100년 훈련 데이터",
        "공통 테스트 데이터"
    ],
    "기간": [
        "1956~2005",
        "1906~2005",
        "2006~2025"
    ],
    "연도 수": [
        len(train_50),
        len(train_100),
        len(test)
    ]
})

st.dataframe(
    data_info,
    use_container_width=True,
    hide_index=True
)

st.info(
    "두 모델 모두 2006~2025년을 한 번도 학습하지 않은 공통 테스트 데이터로 평가합니다."
)


# --------------------------------------------------
# 회귀선 비교
# --------------------------------------------------
st.subheader("② 50년 학습과 100년 학습의 회귀선 비교")

fig = go.Figure()

# 실제 연평균 기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        hovertemplate="%{x}년<br>평균기온: %{y:.2f}℃<extra></extra>"
    )
)

# 회귀선용 연도
line_years = np.arange(
    yearly["연도"].min(),
    2026
)

line_x = (line_years - 1908).reshape(-1, 1)

# 50년 회귀선
line_pred_50 = model_50.predict(line_x)

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_pred_50,
        mode="lines",
        name="1956~2005 학습 회귀선",
        hovertemplate="%{x}년<br>예측: %{y:.2f}℃<extra></extra>"
    )
)

# 100년 회귀선
line_pred_100 = model_100.predict(line_x)

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_pred_100,
        mode="lines",
        name="1906~2005 학습 회귀선",
        hovertemplate="%{x}년<br>예측: %{y:.2f}℃<extra></extra>"
    )
)

# 테스트 구간 표시
fig.add_vrect(
    x0=2006,
    x1=2025,
    fillcolor="gray",
    opacity=0.12,
    line_width=0,
    annotation_text="테스트 데이터",
    annotation_position="top left"
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified",
    height=550
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 기울기 비교
# --------------------------------------------------
st.subheader("③ 회귀선의 기울기 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 최근 50년 학습")

    st.metric(
        "100년에 기온이 변하는 정도",
        f"{slope_50_100:+.2f} ℃"
    )

    st.write(
        f"연간 기울기: {slope_50:+.4f} ℃/년"
    )

with col2:
    st.markdown("### 최근 100년 학습")

    st.metric(
        "100년에 기온이 변하는 정도",
        f"{slope_100_100:+.2f} ℃"
    )

    st.write(
        f"연간 기울기: {slope_100:+.4f} ℃/년"
    )


if abs(slope_50_100) > abs(slope_100_100):
    st.info(
        "최근 50년으로 학습한 회귀선이 최근 100년으로 학습한 회귀선보다 "
        "기온 변화의 기울기가 더 큽니다."
    )
elif abs(slope_50_100) < abs(slope_100_100):
    st.info(
        "최근 100년으로 학습한 회귀선이 최근 50년으로 학습한 회귀선보다 "
        "기온 변화의 기울기가 더 큽니다."
    )
else:
    st.info("두 회귀선의 기울기가 같습니다.")


# --------------------------------------------------
# 예측 성능 비교
# --------------------------------------------------
st.subheader("④ 최근 20년 테스트 데이터 예측 성능")

st.write(
    "MAE와 MSE는 작을수록 좋고, R²는 1에 가까울수록 좋습니다."
)

performance = pd.DataFrame({
    "모델": [
        "1956~2005 학습",
        "1906~2005 학습"
    ],
    "MAE": [
        mae_50,
        mae_100
    ],
    "MSE": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    performance.style.format({
        "MAE": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 성능 비교 메시지
# --------------------------------------------------
st.subheader("⑤ 두 모델의 성능 비교")

col1, col2, col3 = st.columns(3)

with col1:
    if mae_50 < mae_100:
        st.success("MAE: 50년 학습이 더 좋음")
    elif mae_50 > mae_100:
        st.success("MAE: 100년 학습이 더 좋음")
    else:
        st.info("MAE: 동일")

with col2:
    if mse_50 < mse_100:
        st.success("MSE: 50년 학습이 더 좋음")
    elif mse_50 > mse_100:
        st.success("MSE: 100년 학습이 더 좋음")
    else:
        st.info("MSE: 동일")

with col3:
    if r2_50 > r2_100:
        st.success("R²: 50년 학습이 더 좋음")
    elif r2_50 < r2_100:
        st.success("R²: 100년 학습이 더 좋음")
    else:
        st.info("R²: 동일")


# --------------------------------------------------
# 테스트 기간 실제값 vs 예측값
# --------------------------------------------------
st.subheader("⑥ 최근 20년 실제 기온과 예측 기온")

test_result = test[
    ["연도", "연평균기온"]
].copy()

test_result["50년 학습 예측"] = pred_50
test_result["100년 학습 예측"] = pred_100

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["연평균기온"],
        mode="lines+markers",
        name="실제 기온",
        hovertemplate="%{x}년<br>실제: %{y:.2f}℃<extra></extra>"
    )
)

fig2.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["50년 학습 예측"],
        mode="lines+markers",
        name="50년 학습 예측",
        hovertemplate="%{x}년<br>예측: %{y:.2f}℃<extra></extra>"
    )
)

fig2.add_trace(
    go.Scatter(
        x=test_result["연도"],
        y=test_result["100년 학습 예측"],
        mode="lines+markers",
        name="100년 학습 예측",
        hovertemplate="%{x}년<br>예측: %{y:.2f}℃<extra></extra>"
    )
)

fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# 결과표
# --------------------------------------------------
st.subheader("⑦ 테스트 데이터 상세 결과")

result_display = test_result.copy()

result_display["실제 기온"] = result_display["연평균기온"].round(2)
result_display["50년 예측"] = result_display["50년 학습 예측"].round(2)
result_display["100년 예측"] = result_display["100년 학습 예측"].round(2)

result_display = result_display[
    ["연도", "실제 기온", "50년 예측", "100년 예측"]
]

st.dataframe(
    result_display,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# 마무리
# --------------------------------------------------
st.markdown("---")

st.markdown(
    """
### 📌 해석하기

- **MAE**: 실제 기온과 예측 기온의 평균적인 차이입니다. 작을수록 좋습니다.
- **MSE**: 예측 오차를 제곱해서 평균낸 값입니다. 작을수록 좋습니다.
- **R²**: 회귀모델이 기온의 변동을 얼마나 설명하는지를 나타냅니다. 1에 가까울수록 좋습니다.
- **기울기**: 회귀선이 100년 동안 몇 ℃ 변하는지를 나타냅니다.

두 모델은 똑같이 **2006~2025년을 테스트 데이터**로 사용하기 때문에,
50년 학습과 100년 학습 중 어느 쪽이 최근 기온을 더 잘 예측하는지 직접 비교할 수 있습니다.
"""
)
