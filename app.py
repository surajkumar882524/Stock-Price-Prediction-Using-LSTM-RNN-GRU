import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU, Dropout

# ---------------- UI ----------------
st.title("📈 Stock Price Prediction (LSTM, RNN, GRU)")

# ---------------- File Upload ----------------
file = st.file_uploader("Upload CSV File", type=["csv"])

# ---------------- MAIN BLOCK ----------------
if file is not None:

    # ✅ STEP 1: Load Data
    df = pd.read_csv(file)
    df = df.head(5000)   # 🔥 speed optimization

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # ✅ STEP 2: Select Correct Columns
    st.write("Columns in dataset:", df.columns)

    data = df[['PREVCLOSE', 'CLOSE']]
    data.dropna(inplace=True)

    # ✅ STEP 3: Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # ✅ STEP 4: Create Sequences
    def create_dataset(data, time_step=50):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data)

    # ✅ STEP 5: Train-Test Split
    train_size = int(len(X) * 0.8)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    y_train_open = y_train[:, 0]
    y_test_open = y_test[:, 0]

    y_train_close = y_train[:, 1]
    y_test_close = y_test[:, 1]

    # ✅ STEP 6: Train Models
    with st.spinner("⏳ Training models... Please wait"):

        # 🔵 LSTM
        model_lstm = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mse')
        model_lstm.fit(X_train, y_train_open, epochs=2, batch_size=32, verbose=0)

        # 🔴 RNN
        model_rnn = Sequential([
            SimpleRNN(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            SimpleRNN(64),
            Dropout(0.2),
            Dense(1)
        ])
        model_rnn.compile(optimizer='adam', loss='mse')
        model_rnn.fit(X_train, y_train_close, epochs=2, batch_size=32, verbose=0)

        # 🟢 GRU
        model_gru = Sequential([
            GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            GRU(64),
            Dropout(0.2),
            Dense(1)
        ])
        model_gru.compile(optimizer='adam', loss='mse')
        model_gru.fit(X_train, y_train_close, epochs=2, batch_size=32, verbose=0)

    st.success("✅ Models trained successfully!")

    # ✅ STEP 7: Predictions
    pred_lstm = model_lstm.predict(X_test).reshape(-1)
    pred_rnn = model_rnn.predict(X_test).reshape(-1)
    pred_gru = model_gru.predict(X_test).reshape(-1)

    # ✅ STEP 8: Evaluation
    st.subheader("📉 Model Performance")

    lstm_mse = mean_squared_error(y_test_open, pred_lstm)
    rnn_mse = mean_squared_error(y_test_close, pred_rnn)
    gru_mse = mean_squared_error(y_test_close, pred_gru)

    st.write("🔵 LSTM MSE (Open):", lstm_mse)
    st.write("🔴 RNN MSE (Close):", rnn_mse)
    st.write("🟢 GRU MSE (Close):", gru_mse)

    # ✅ STEP 9: Graph
    st.subheader("📊 LSTM Prediction vs Actual")

    fig = plt.figure()
    plt.plot(y_test_open, label='Actual')
    plt.plot(pred_lstm, label='Predicted (LSTM)')
    plt.legend()

    st.pyplot(fig)

else:
    st.warning("⚠️ Please upload a CSV file to proceed")
    
    



    