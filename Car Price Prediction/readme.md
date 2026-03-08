# 🚗 Car Price Prediction Using Machine Learning

## 📌 Project Overview

This project builds a machine learning model to predict the selling
price of cars based on vehicle attributes such as present price, fuel type,
transmission, ownership history, driven kilometers, and more.

The workflow covers:

-   Exploratory Data Analysis (EDA)
-   Feature Engineering
-   Outlier Treatment
-   Multicollinearity Check (VIF)
-   Data Transformation & Scaling
-   Model Training & Hyperparameter Tuning
-   Model Comparison & Selection

The final objective is to identify the most accurate and generalizable
model for car price prediction.

------------------------------------------------------------------------

## 📊 Dataset Description

The dataset (`car data.csv`) contains historical information about car sales.

### Key Features

  Feature         Description
  --------------- --------------------------------
  **Car_Name:**         Name of the car model\
  **Year:**             Manufacturing year\
  **Present_Price:**    Current ex-showroom price\
  **Selling_Price:**    Target variable (resale price)\
  **Driven_kms:**       Total kilometers driven\
  **Fuel_Type:**        Petrol / Diesel / CNG\
  **Selling_type:**     Dealer / Individual\
  **Transmission:**     Manual / Automatic\
  **Owner:**            Number of previous owners

------------------------------------------------------------------------

## 🔍 Key Findings

-   **City** is the most sold car model in the dataset.
-   **2015** recorded the highest number of car purchases.
-   Petrol vehicles dominate the dataset compared to Diesel and CNG.
-   Most cars are sold through **Dealers** rather than Individuals.
-   Manual transmission cars are significantly more common than
    Automatic.
-   Diesel cars generally have higher selling prices.
-   Dealer-listed cars tend to be priced higher than Individual
    listings.
-   Automatic transmission vehicles are usually priced higher than
    Manual.
-   First-owner vehicles command higher prices.
-   There is a negative relationship between driven kilometers and
    selling price.
-   `Present_Price` is the strongest predictor of `Selling_Price`.

------------------------------------------------------------------------

## 🛠 Feature Engineering & Preprocessing

### Car Age Feature

`car_age = current_year - Year`

### Outlier Handling

-   IQR method for `Selling_Price`
-   99th percentile capping for numerical variables

### Encoding

-   One-hot encoding using `pd.get_dummies()`

### Multicollinearity

-   Variance Inflation Factor (VIF) used to detect collinearity issues.

### Target Transformation

Log transformation applied to reduce skewness.

### Scaling

StandardScaler applied to normalize features.

------------------------------------------------------------------------

## 🤖 Models Implemented

-   Decision Tree Regressor
-   Random Forest Regressor
-   Gradient Boosting Regressor
-   XGBoost Regressor

Hyperparameter tuning performed using GridSearchCV and
RandomizedSearchCV.

------------------------------------------------------------------------

## 📊 Evaluation Metrics

-   R² Score (Primary metric)
-   Mean Squared Error (MSE)
-   Root Mean Squared Error (RMSE)
-   Mean Absolute Error (MAE)

------------------------------------------------------------------------

## 🏆 Final Model

Selected Model: **Tuned XGBoost Regressor**

Performance: - Training R² ≈ 0.99 - Testing R² ≈ 0.92

The model demonstrates strong predictive performance and generalization
ability.

------------------------------------------------------------------------

## 📁 Project Structure

├── car data.csv\
├── car_price_prediction.ipynb\
├── trained_models/\
|     └── dt_model.joblib\
|     └── gb_model.joblib\
|     └── rf_model.joblib\
|     └── tuned_xgb.joblib\
|     └── xgb_model.joblib\
├── readme.md

------------------------------------------------------------------------

## 🎯 Conclusion

This project demonstrates a complete machine learning pipeline for
predicting used car prices. Through comprehensive data exploration,
preprocessing, and model comparison, Tuned XGBoost Regressor emerged as the most
reliable predictor.

The insights provide a clear understanding of factors influencing used
car prices and showcase practical machine learning application in the
automotive domain.
