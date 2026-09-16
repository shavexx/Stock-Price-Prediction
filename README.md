# 📈 Stock Price Prediction Web Application

A simple, beginner-friendly machine learning web application built using Python, Pandas, Scikit-learn, and Streamlit. This project predicts the future closing price of a stock using historical stock market features (`Open`, `High`, `Low`, `Volume`).

---

## 📌 Project Overview
Stock price prediction is a classic machine learning problem. In this project, we use a simple **Linear Regression** algorithm to understand the linear relationship between daily stock indicators and the final closing price. 

This application features an interactive **Streamlit** dashboard where users can:
- View summary statistics (Latest, Highest, Lowest, and Average prices).
- Analyze historical closing price trends via interactive charts.
- Train a machine learning model on uploaded or default historical stock data.
- Evaluate model accuracy using **Mean Absolute Error (MAE)** and **R² Score**.
- Interactively input future stock metrics (`Open`, `High`, `Low`, `Volume`) to predict the expected `Close` price.

---

## 📁 Project File Structure
```text
stock-price-prediction/
│
├── app.py                  # Main Streamlit web application code
├── requirements.txt        # List of required Python packages
├── data/
│   └── stock_data.csv      # Sample historical stock market dataset
└── README.md               # Project documentation and guide
```

---

## 📊 Dataset Information
The dataset uses standard historical stock market columns:
- **Date**: The trading date (converted to proper datetime).
- **Open**: Opening price of the stock for the day.
- **High**: Highest price reached during the trading session.
- **Low**: Lowest price reached during the trading session.
- **Close**: Final closing price of the stock (**Target Variable**).
- **Volume**: Total number of shares traded during the day.

---

## 🛠️ Technologies Used
- **Python**: Primary programming language.
- **Pandas**: For reading CSV files, data cleaning, and dataset manipulation.
- **NumPy**: For numerical array operations.
- **Scikit-learn**: For splitting data, training the Linear Regression model, and evaluating metrics.
- **Streamlit**: Framework for building the interactive web user interface.
- **Plotly**: For interactive line charts and visualizations.

---

## 🧠 Machine Learning Algorithm: Linear Regression

### How Linear Regression Works
Linear Regression is a simple supervised learning algorithm used for predicting continuous numerical values. It assumes a linear relationship between input features ($X$) and the target output ($y$).

The mathematical equation for Linear Regression is:

$$ \text{Close} = w_1 \cdot \text{Open} + w_2 \cdot \text{High} + w_3 \cdot \text{Low} + w_4 \cdot \text{Volume} + b $$

Where:
- $w_1, w_2, w_3, w_4$ are the learned weights (coefficients) for each feature.
- $b$ is the intercept (bias term).

---

## ⚡ How Prediction Works Step-by-Step

1. **Load Data**: Load `stock_data.csv` using Pandas.
2. **Clean & Sort**: Drop any missing rows and sort chronologically by `Date`.
3. **Feature Selection**:
   - **Features ($X$)**: `Open`, `High`, `Low`, `Volume`
   - **Target ($y$)**: `Close`
4. **Train-Test Split**: Divide data into 80% training data (to train the model) and 20% testing data (to measure model performance).
5. **Model Fitting**: Train `LinearRegression()` on the training set.
6. **Evaluation**: Compare predicted test values against actual test values using **MAE** (lower is better) and **R² Score** (closer to 100% is better).
7. **User Prediction**: The user enters Open, High, Low, and Volume in the UI. The trained model applies the learned formula to calculate the predicted closing price.

---

## 💻 Installation & How to Run

### Step 1: Install Dependencies
Open your command prompt or terminal in the project directory and run:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Streamlit Application
Execute the following command:
```bash
streamlit run app.py
```

### Step 3: Open in Browser
Streamlit will automatically launch the web interface in your browser at `http://localhost:8501`.

---

## ⚠️ Educational Disclaimer
This prediction application is created for **educational and college project demonstration purposes only**. Stock market prices are influenced by complex external factors (news, economic events, market sentiment) and cannot be predicted accurately using simple linear models alone. Do **not** use this app for real financial trading or investment decisions.

## 🚀 Live Demo

🔗 **Live Application:**  
https://stock-price-prediction-ngte9snevdsk5rerkum2vq.streamlit.app/
