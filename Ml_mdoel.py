import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def load_and_prepare_data():
    df = pd.read_csv("total_data.csv", parse_dates=["Date"], index_col="Date")
    
    df['Price_Return'] = df['Price'].pct_change().shift(-1)
    df['Oil_Change'] = df['Oil'].pct_change()
    df['Rate_Change'] = df['Rate'].diff() / 100

    df = df.dropna()
    return df

def train_model():
    df = load_and_prepare_data()
    X = df[['War', 'Oil_Change', 'Rate_Change']]
    y = df['Price_Return']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "rf_model.pkl")
    print("Model training complete.")

def load_model():
    return joblib.load("rf_model.pkl")

def predict_return(war, oil_change, rate_change):
    model = load_model()
    scenario = pd.DataFrame({
        'War': [war],
        'Oil_Change': [oil_change],
        'Rate_Change': [rate_change]
    }, index=[0])

    raw = model.predict(scenario)[0]
    return abs(raw)  # ✔️ 방향 제거, 절댓값만 반환


if __name__ == "__main__":
    print("Starting training...")
    train_model()
result = predict_return(1, 0.02, -0.01)
print(result)

