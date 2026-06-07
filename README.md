# ML-Portfolio-Optimizer-Project

This project focuses on building a machine learning-based portfolio optimization system that uses historical market data to rank stocks and simulate portfolio performance over time.

The idea behind the project was to explore whether machine learning models can help improve stock selection by combining market features, ranking algorithms, and portfolio validation techniques.

## Features

* Collects market data automatically
* Uses feature engineering for stock analysis
* Portfolio ranking using ML models
* Walk-forward validation for testing strategies
* Portfolio simulation and performance tracking
* Interactive dashboard using Streamlit

---

## Technologies Used

* Python
* Streamlit
* Pandas / NumPy
* Scikit-learn
* XGBoost
* CatBoost
* Yahoo Finance API
* SciPy

---

## Project Structure

```bash
ML-Portfolio-Optimizer-Project/
│
├── app.py                # Streamlit dashboard
├── data_pipeline.py      # Data collection and preprocessing
├── model.py              # ML models and ranking logic
├── validation.py         # Walk-forward validation and testing
├── requirements.txt      # Dependencies
├── README.md
├── LICENSE
└── .gitignore
```

---

## How It Works

### 1. Data Collection

Market data is collected from external sources and processed for analysis.

### 2. Feature Engineering

Some features used include:

* Momentum indicators
* Volatility measures
* Volume trends
* Market returns
* Moving average distance
* Breakout signals
* Mean reversion metrics

### 3. Model Training

The project uses ranking models such as:

* XGBoost Ranker
* CatBoost Ranker

These models rank stocks instead of simply predicting prices.

### 4. Validation

Strategies are tested using walk-forward validation to reduce overfitting and simulate real-world conditions.

### 5. Portfolio Construction

Top-ranked stocks are selected and portfolio performance is tracked over multiple periods.

---

## Installation

Clone the repository:

```bash
git clone <repo-url>
cd ML-Portfolio-Optimizer-Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

---

## Results

The project focuses more on stock ranking and portfolio construction rather than predicting exact prices. Performance is evaluated based on how well the selected portfolio performs over time.

---

## Limitations

* Market behavior changes over time
* Historical performance does not guarantee future returns
* Results depend heavily on feature selection and market conditions

---

## Future Improvements

Things that can be added later:

* More portfolio optimization methods
* Better risk management techniques
* More asset classes
* Hyperparameter tuning
* Live market integration

---

## Author

Built as a project to explore machine learning applications in portfolio optimization and quantitative finance.
