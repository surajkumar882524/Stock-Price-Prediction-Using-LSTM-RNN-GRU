# 📈 Stock Price Prediction Using LSTM, RNN & GRU

A Deep Learning project that predicts stock prices using Sequence Models such as Long Short-Term Memory (LSTM), Recurrent Neural Network (RNN), and Gated Recurrent Unit (GRU). This project compares the performance of all three models to identify the most effective approach for stock price forecasting.

---

# 🚀 Project Overview

Stock price prediction is one of the most challenging tasks in financial analysis due to the dynamic nature of the stock market.

This project applies Deep Learning techniques to historical stock market data to forecast future stock prices. Three sequence models—LSTM, RNN, and GRU—are trained and evaluated to compare their prediction performance.

---

# 🎯 Objectives

- Predict future stock prices using Deep Learning.
- Compare the performance of LSTM, RNN, and GRU models.
- Analyze prediction accuracy using evaluation metrics.
- Visualize actual vs predicted stock prices.

---

# 📂 Project Structure

```
Stock-Price-Prediction-Using-LSTM-RNN-GRU/
│
├── Dataset/
├── Models/
├── Notebook/
├── Images/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit (if applicable)

---

# 📊 Dataset

The project uses historical stock market data containing:

- Open Price
- High Price
- Low Price
- Close Price
- Volume
- Date

The dataset is preprocessed before training the models.

---

# ⚙ Data Preprocessing

- Data Cleaning
- Handling Missing Values
- Feature Scaling using MinMaxScaler
- Sequence Generation
- Train-Test Split

---

# 🧠 Deep Learning Models

## 1️⃣ Recurrent Neural Network (RNN)

The RNN model captures sequential information from historical stock prices.

---

## 2️⃣ Long Short-Term Memory (LSTM)

LSTM overcomes the vanishing gradient problem and captures long-term dependencies in stock price sequences.

---

## 3️⃣ Gated Recurrent Unit (GRU)

GRU is a lightweight alternative to LSTM that trains faster while maintaining strong prediction performance.

---

# 📈 Model Workflow

Historical Stock Data

⬇

Data Preprocessing

⬇

Feature Scaling

⬇

Sequence Creation

⬇

Train RNN

⬇

Train LSTM

⬇

Train GRU

⬇

Model Evaluation

⬇

Prediction

⬇

Visualization

---

# 📊 Model Evaluation

The models are evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Loss Curves
- Prediction Graphs

---

# 📷 Project Screenshots

## Dataset

<img width="1192" height="199" alt="image" src="https://github.com/user-attachments/assets/43bf3fe7-e799-4d6d-9049-bad5f6109924" />
<img width="1752" height="702" alt="Screenshot 2026-07-30 182615" src="https://github.com/user-attachments/assets/eb338d66-5cd8-424e-ad26-c81821e15a0a" />




---


## Model Training

<img width="685" height="754" alt="Screenshot 2026-07-30 183742" src="https://github.com/user-attachments/assets/8bc21dfd-f553-4a4a-acbe-33e72ce67d21" />
<img width="699" height="781" alt="Screenshot 2026-07-30 183827" src="https://github.com/user-attachments/assets/621a8714-5de8-4608-8bb7-5c91a75b8d6d" />
<img width="632" height="749" alt="Screenshot 2026-07-30 183848" src="https://github.com/user-attachments/assets/8cc96686-14e0-405e-9bd0-d765da1de4dd" />




---

## Prediction Graph
<img width="994" height="528" alt="output" src="https://github.com/user-attachments/assets/222b5cb9-959d-4789-b852-86b7bab9f5bc" />


---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Stock-Price-Prediction-Using-LSTM-RNN-GRU.git
```

Go to the project folder

```bash
cd Stock-Price-Prediction-Using-LSTM-RNN-GRU
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📌 Future Improvements

- Real-time stock prediction
- Live stock market API integration
- Multi-stock prediction
- Hyperparameter optimization
- Deployment on Streamlit Cloud
- News sentiment analysis

---


---

# 👨‍💻 Author

**Suraj **

GitHub: https://github.com/surajkumar882524

LinkedIn: (https://www.linkedin.com/in/suraj-kumar-0b340b363/)

---
