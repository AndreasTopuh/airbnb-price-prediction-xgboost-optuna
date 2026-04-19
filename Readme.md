# Airbnb Price Prediction Model

This notebook implements an end-to-end machine learning pipeline for predicting Airbnb listing prices with three tree-based models: **Random Forest**, **XGBoost**, and **LightGBM**.

---

## Pipeline Overview

### 1. Setup and Imports
- **Data manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`
- **ML models:** `RandomForestRegressor`, `XGBRegressor`, `LGBMRegressor`
- **Evaluation metrics:** `mean_squared_error`, `mean_absolute_error`, `r2_score`
- **Preprocessing:** `LabelEncoder`
- **Hyperparameter optimization:** `optuna`
- **Cross-validation:** `KFold`, `train_test_split`
- **Model export:** `joblib`

### 2. Data Loading and Initial Exploration
- Load the Airbnb dataset (261,894 rows × 55 columns)
- Inspect data types and missing values
- Review dataset shape and basic structure

### 3. Data Cleaning
- Drop columns with more than 50% missing values
- Drop text and identifier columns that are not used for modeling
- Fill missing numerical values with the median
- Fill missing categorical values with the mode
- Remove duplicate rows

### 4. Outlier Removal
- Remove invalid prices (`price <= 0`)
- Filter extreme price outliers using the 1st and 99th percentiles
- Compare price statistics before and after filtering

### 5. Exploratory Data Analysis
- Create price categories: **Budget**, **Mid-Range**, **Premium**, **Luxury**
- Visualize the original and log-transformed price distribution

### 6. Feature Engineering
- Add interaction and ratio features
- Add `amenities_count` from the raw amenities field
- Add location features such as `lat_long_interaction` and `distance_to_center`

### 7. Target Transformation
- Apply `np.log1p(price)` to reduce target skewness
- Convert predictions back with `np.expm1()` during evaluation

### 8. Categorical Encoding
- Ordinal encode `price_class`
- Label encode binary categorical columns
- One-hot encode low-cardinality categorical columns
- Target encode high-cardinality categorical columns after the split

### 9. Train-Test Split
- Split data into 80% training and 20% testing
- Keep features unscaled because tree-based models are threshold-based

### 10. Baseline Model Training
- Train default Random Forest, XGBoost, and LightGBM models
- Evaluate with RMSE, MAE, R², and RMSLE

### 11. Cross-Validation Evaluation
- Run 5-fold cross-validation with fold-safe target encoding
- Report mean CV RMSE in log space

### 12. Hyperparameter Tuning
- Use Optuna with fold-safe target encoding inside each validation fold
- Tune all three models and evaluate on the holdout test set

### 13. Model Comparison and Visualization
- Combine baseline and tuned results in one table
- Visualize RMSE, R², and MAE for tuned models
- Select the best model using the highest R² score

### 14. Model Export
- Map the selected model name to the trained model object
- Save the best model with `joblib.dump()`

### 15. Model Verification
- Predict 50 holdout samples
- Compare actual vs predicted prices in tabular and graphical form
