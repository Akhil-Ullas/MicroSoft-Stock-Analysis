# Microsoft Stock Price Predictor

A machine learning project that predicts Microsoft (MSFT) stock closing prices using an **LSTM (Long Short-Term Memory)** neural network, with an interactive **Streamlit** web app and a **Power BI** dashboard for visual analysis.

---

## Project Structure

```
MicroSoft-Stock-Analysis/
├── Dashboard/          # Dashboard screenshots or exports
├── Data/               # MSFT stock dataset (MSFT_data.csv)
├── Models/             # Trained LSTM model & scaler
│   ├── LSTM.keras
│   └── scaler.pkl
├── Power Bi/           # Power BI report (.pbix)
├── app.py              # Streamlit web application
├── msft_main.ipynb     # Jupyter notebook (EDA + model training)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Features

-  **Exploratory Data Analysis** of historical MSFT stock prices
-  **LSTM model** trained to predict closing prices
-  **Date-based prediction** — input any date (past or future)
-  **Auto-handles** weekends & holidays by snapping to nearest trading day
-  **Power BI dashboard** for interactive stock trend visualization
-  **Streamlit app** for a user-friendly prediction interface

---

##  How the Model Works

1. Takes the last **60 days** of closing prices as input
2. Scales the data using a pre-fitted `MinMaxScaler`
3. Feeds the sequence into an **LSTM neural network**
4. Outputs the predicted closing price for the target date

**For future dates**, the model iteratively predicts day-by-day using its own predictions as input.

---

##  Streamlit App

The app allows users to:
- Select any date using a date picker
- Get the **predicted closing price**
- See the **actual price** (if it's a historical date)
- View the **difference** and **percentage error** between actual and predicted

```
Predicted Price: $391.90
Actual Price:    $407.39
Difference:      $15.49  (3.80%)
```

---

##  Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Akhil-Ullas/MicroSoft-Stock-Analysis.git
cd MicroSoft-Stock-Analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| **Python** | Core language |
| **TensorFlow / Keras** | LSTM model training & inference |
| **Streamlit** | Interactive web app |
| **Pandas / NumPy** | Data processing |
| **Scikit-learn** | Data scaling (MinMaxScaler) |
| **Matplotlib / Plotly** | Visualizations |
| **yfinance** | Stock data fetching |
| **Power BI** | Business intelligence dashboard |
| **Jupyter Notebook** | EDA and model development |

---

##  Dataset

- **Source:** Yahoo Finance via `yfinance`
- **Ticker:** `MSFT` (Microsoft Corporation)
- **Key Column Used:** `Close` (Closing Price)

---

##  Contact

**Akhil Ullas**
- GitHub: [@Akhil-Ullas](https://github.com/Akhil-Ullas)
- Linkedin:[Akhil V U](https://www.linkedin.com/in/akhil-v-u/)
---

