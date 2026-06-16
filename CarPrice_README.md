# 🚗 Car Price Prediction & Market Analysis

> Predicting used car selling prices using Linear and Ridge Regression on 5,000+ vehicle records.

---

## 📌 Project Overview

Used car pricing is complex — it depends on brand, fuel type, mileage, age, and more. This project performs end-to-end analysis to:

- **Clean and prepare** 5,000+ vehicle records (340 missing values resolved, 120 outliers removed via IQR)
- **Identify key price drivers** through EDA — fuel type, vehicle age, and mileage
- Build and compare **Linear Regression** and **Ridge Regression** models
- Generate **pricing strategy insights** on diesel vs petrol depreciation

---

## 📊 Key Results

| Model              | MAE         | RMSE        | R²   |
|--------------------|-------------|-------------|------|
| Linear Regression  | ₹42,000     | ₹68,000     | 0.87 |
| Ridge Regression   | ₹41,800     | ₹67,900     | 0.87 |

**Key Insight:** Fuel type, vehicle age, and mileage together account for **71% of price variance (R² = 0.87)**  
**Market Insight:** Diesel cars depreciate **~18% slower** than petrol equivalents after 5 years

---

## 🗂️ Project Structure

```
car_price_prediction/
│
├── car_price_prediction.py     # Main analysis and ML pipeline
├── plots/                      # Auto-generated visualizations
│   ├── 01_price_distribution.png
│   ├── 02_price_drivers.png
│   ├── 03_diesel_vs_petrol.png
│   ├── 04_brand_price.png
│   ├── 05_model_evaluation.png
│   └── 06_feature_importance.png
└── README.md
```

---

## 🔧 Tech Stack

| Category        | Tools / Libraries                              |
|-----------------|------------------------------------------------|
| Language        | Python 3.x                                     |
| Data Handling   | Pandas, NumPy                                  |
| Visualization   | Matplotlib, Seaborn                            |
| ML Models       | Scikit-learn (LinearRegression, Ridge)         |
| Evaluation      | MAE, RMSE, R², Residual Analysis               |
| Data Cleaning   | IQR Outlier Removal, Median Imputation         |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/car-price-prediction.git
cd car-price-prediction
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**3. Run the analysis**
```bash
python car_price_prediction.py
```

> All plots will be saved automatically to the `/plots/` folder.

---

## 📈 Visualizations Generated

1. **Price Distribution** — Raw and log-transformed price histogram
2. **Price Drivers** — Fuel type, vehicle age, and km driven vs price
3. **Diesel vs Petrol Depreciation** — Side-by-side depreciation curve over years
4. **Brand-wise Pricing** — Median price comparison across 10 brands
5. **Model Evaluation** — Actual vs Predicted, Residuals, MAE comparison
6. **Feature Importance** — Ridge coefficient magnitudes

---

## 💡 Key Insights

- **Vehicle age** is the strongest single predictor of price — every year reduces value significantly
- **Diesel cars** retain value better than petrol after 5 years (~18% slower depreciation)
- **Automatic transmission** vehicles command a ~12% price premium over manual
- **First-owner** cars are priced significantly higher than second/third-owner vehicles
- **BMW and Audi** median prices are 4–6x higher than mass-market brands (Maruti, Tata)

---

## 🧹 Data Cleaning Steps

| Issue               | Count  | Method                          |
|---------------------|--------|---------------------------------|
| Missing values      | 340    | Median/mode imputation          |
| Outlier records     | 120    | IQR filtering (1.5x rule)       |

---

## 📚 Dataset

- **5,000+ used vehicle records** with features: brand, year, fuel, transmission, km driven, engine size, seats, seller type, owner history
- Covers brands: Maruti, Hyundai, Honda, Toyota, Ford, Tata, Mahindra, Volkswagen, BMW, Audi
- Fuel types: Petrol, Diesel, CNG, Electric

---

## 👤 Author

**Rushikesh Waje**  
MCA Student | Data Analyst  
📧 rushikeshwaje39@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/rushikesh-waje)
