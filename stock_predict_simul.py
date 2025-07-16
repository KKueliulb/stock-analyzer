import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import Ml_mdoel  # 사용자 정의 모듈

plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide")
st.title("📈 외부 요인 기반 주가 시뮬레이터 (LSTM + Event Study)")

# --- 세션 상태 기본값 설정 ---
if 'ticker' not in st.session_state:
    st.session_state['ticker'] = "005930.KS"
if 'start_date' not in st.session_state:
    st.session_state['start_date'] = pd.to_datetime("2023-01-01")
if 'end_date' not in st.session_state:
    yesterday = datetime.now() - timedelta(days=1)
    st.session_state['end_date'] = pd.to_datetime(yesterday.date())
if 'event_date' not in st.session_state:
    today = datetime.now().date()
    st.session_state['event_date'] = pd.to_datetime(today)
if 'future_days' not in st.session_state:
    st.session_state['future_days'] = 30

if 'war' not in st.session_state:
    st.session_state['war'] = False
if 'oil' not in st.session_state:
    st.session_state['oil'] = 0.0
if 'interest' not in st.session_state:
    st.session_state['interest'] = 0.0
if 'market_cap' not in st.session_state:
    st.session_state['market_cap'] = 10000

# --- 사이드바: 티커 및 날짜 입력 파트 ---
st.sidebar.header("① 종목 및 날짜 입력")
st.session_state['ticker'] = st.sidebar.text_input("주식 코드 입력(예: 005930.KS)", value=st.session_state['ticker'])
st.session_state['start_date'] = st.sidebar.date_input("시작일", value=st.session_state['start_date'])
st.session_state['end_date'] = st.sidebar.date_input("종료일", value=st.session_state['end_date'])
st.session_state['event_date'] = st.sidebar.date_input("이벤트 날짜(미래)", value=st.session_state['event_date'])
st.session_state['future_days'] = st.sidebar.number_input("미래 예측 기간(일)", value=st.session_state['future_days'], min_value=1, max_value=180)

st.sidebar.markdown("---")

# --- 사이드바: 외부 변수 입력 파트 ---
st.sidebar.header("② 외부 변수")
if st.sidebar.button("🔄 외부 변수 초기화"):
    st.session_state['war'] = False
    st.session_state['oil'] = 0.0
    st.session_state['interest'] = 0.0
    st.session_state['market_cap'] = 10000

st.session_state['war'] = st.sidebar.checkbox("전쟁 발생", value=st.session_state['war'])
st.session_state['oil'] = st.sidebar.slider("유가 변화율(%)", -10.0, 10.0, st.session_state['oil'])
st.session_state['interest'] = st.sidebar.slider("금리 변화율(%)", -1.0, 1.0, st.session_state['interest'])
st.session_state['market_cap'] = st.sidebar.number_input("시가총액 (억 원)", value=st.session_state['market_cap'])

# --- 변수 할당 ---
ticker = st.session_state['ticker']
start_date = st.session_state['start_date']
end_date = st.session_state['end_date']
event_date = st.session_state['event_date']
future_days = st.session_state['future_days']
war_int = int(st.session_state['war'])
oil_float = st.session_state['oil'] / 100
interest_float = st.session_state['interest'] / 100
market_cap = st.session_state['market_cap']

# --- 데이터 불러오기 함수 ---
def load_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError("Empty data received.")
        return data
    except Exception as e:
        st.warning(f"{ticker} 다운로드 실패: {e}")
        st.info("📄 예제 파일 data.csv를 대신 사용합니다.")
        return pd.read_csv("data.csv", parse_dates=["Date"], index_col="Date")

# --- 예측 실행 ---
if st.button("예측 실행"):
    data = load_stock_data(ticker, start_date, end_date)
    if data.empty:
        st.error("데이터를 불러올 수 없습니다.")
        st.stop()

    if 'Close' in data.columns:
        close = data['Close'].values.reshape(-1, 1)
    elif 'Price' in data.columns:
        close = data['Price'].values.reshape(-1, 1)
    else:
        st.error("데이터에 Close 또는 Price 컬럼이 없습니다.")
        st.stop()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    if len(scaled) < 60:
        st.error(f"LSTM 모델 학습을 위한 데이터가 부족합니다. 최소 60일 이상 필요 (현재 {len(scaled)}일치)")
        st.stop()

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])
    X = np.array(X)
    y = np.array(y).reshape(-1)

    model = Sequential([
        LSTM(50, input_shape=(60, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)

    last_date = data.index[-1]
    if event_date <= last_date.date():
        st.error("이벤트 날짜는 데이터 마지막 날짜 이후의 미래 날짜여야 합니다.")
        st.stop()

    seed = scaled[-60:].copy()
    future_scaled = []

    for _ in range(future_days):
        x_input = seed.reshape(1, 60, 1)
        pred_scaled = model.predict(x_input)[0]
        future_scaled.append(pred_scaled)
        seed = np.append(seed, pred_scaled.reshape(-1, 1), axis=0)[-60:]

    future_scaled = np.array(future_scaled).reshape(-1, 1)
    future_dates = pd.date_range(start=event_date, periods=future_days)

    past_close = close.flatten()
    future_pred = scaler.inverse_transform(future_scaled).flatten()

    predicted_return = Ml_mdoel.predict_return(war_int, oil_float, interest_float)

    sign = -1 if (war_int == 1 or oil_float > 0 or interest_float > 0) else 1
    adjusted_future_pred = future_pred * (1 + sign * predicted_return)

    st.sidebar.write(f"머신러닝 예측 충격 크기: {predicted_return*100:.2f}%")
    st.sidebar.write(f"적용 방향: {'하락' if sign == -1 else '상승'}")

    df_past = pd.DataFrame({
        'Date': data.index[60:],
        'Real': past_close[60:],
        'Predicted': scaler.inverse_transform(y.reshape(-1, 1)).flatten()
    })
    df_past['Adj'] = df_past['Predicted']

    df_future = pd.DataFrame({
        'Date': future_dates,
        'Predicted': future_pred,
        'Adj': adjusted_future_pred
    })

    df_all = pd.concat([df_past, df_future], ignore_index=True)
    df_all['AR'] = df_all['Adj'] - df_all['Predicted']
    df_all['CAR'] = df_all['AR'].cumsum()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_all['Date'], np.concatenate([df_past['Real'], [np.nan]*future_days]), label='실제 주가')
    ax.plot(df_all['Date'], df_all['Predicted'], label='LSTM 예측 주가', linestyle='--')
    ax.plot(df_all['Date'], df_all['Adj'], label='외부 요인 반영 주가', linestyle='-.')
    ax.axvline(event_date, color='red', linestyle=':', label='이벤트일')
    ax.set_title(f"{ticker} 주가 예측 및 외부 충격 시뮬레이션")
    ax.legend()
    st.pyplot(fig)

    car = df_all['CAR'].iloc[-1]
    loss = car / df_all['Predicted'].mean() * market_cap

    st.metric(label="📉 누적 초과 수익률 (CAR)", value=f"{car:.2f}")
    st.success(f"💰 예상 수익: 약 {loss:.2f}억 원")

    with st.expander("📑 분석 데이터 보기"):
        st.dataframe(df_all[['Date', 'Real', 'Predicted', 'Adj', 'AR', 'CAR']])
