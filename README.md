# Predicting Home Prices Using Economic Indicators
This project aims to predict home prices using various economic indicators from the Federal Reserve Economic Data (FRED). The project involves data collection, data preparation, model building, and analysis of the results.

## Introduction
This project uses machine learning models to predict home prices based on several economic indicators. The models used include Linear Regression, Random Forest Regression, Decision Tree Regression, Gradient Boosting Regression, and XGBoost Regression.

## Data Collection
The data is collected from the Federal Reserve Economic Data (FRED) using the FRED API. The following economic indicators are used:

 - Case-Shiller Home Price Index
 - Unemployment Rate
 - Inflation (CPI)
 - Gross Domestic Product (GDP)
 - 30-Year Fixed Mortgage Rate
 - Median Household Income
 - Housing Starts
 - Population
 - Interest Rates

## Data Preparation
The collected data is merged into a single DataFrame, with the 'date' column as the common key. Missing values are handled using forward filling, and rows with any remaining missing values are dropped.

```bash
# Dictionary of series IDs and their descriptive column names
series_dict = {
    'CSUSHPINSA': 'Case_Shiller_Home_Price_Index',
    'UNRATE': 'Unemployment_Rate',
    'CPIAUCSL': 'Inflation',
    'GDP': 'Gross_Domestic_Product',
    'MORTGAGE30US': '30_Year_Fixed_Mortgage_Rate',
    'MEHOINUSA672N': 'Median_Household_Income',
    'HOUST': 'Housing_Starts',
    'POPTHM': 'Population',
    'FEDFUNDS': 'Interest_Rates'
}

# Fetch and merge data
dataframes = []
for series_id, column_name in series_dict.items():
    df = fetch_fred_data(series_id, api_key, column_name)
    dataframes.append(df)

combined_df = dataframes[0]
for df in dataframes[1:]:
    combined_df = combined_df.merge(df, on='date', how='outer')

combined_df.ffill(inplace=True)
combined_df.dropna(inplace=True)
```

## Model Building
The data is split into training and testing sets. Several machine learning models are trained and evaluated, including:

1. Linear Regression
2. Random Forest Regression
3. Decision Tree Regression
4. Gradient Boosting Regression
5. XGBoost Regression

```bash
# Splitting the data into training and testing sets
X = combined_df.drop(columns=['date', 'Case_Shiller_Home_Price_Index'])
y = combined_df['Case_Shiller_Home_Price_Index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization and training
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42),
    'Decision Tree Regression': DecisionTreeRegressor(random_state=42),
    'Gradient Boosting Regression': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost Regression': XGBRegressor(n_estimators=100, random_state=42)
}

results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mse)
    results.append({"Model": model_name, "Mean Squared Error": mse, "R-squared": r2, "Root Mean Squared Error": rmse})
```

## Results and Analysis
The performance of each model is evaluated based on Mean Squared Error (MSE), R-squared, and Root Mean Squared Error (RMSE). The Random Forest Regression model performed the best with the lowest RMSE and the highest R² score.

```bash
from tabulate import tabulate

# Print results in a tabular format
print(tabulate(results, headers="keys", tablefmt="grid"))
```

Feature importance is analyzed for the Random Forest model, and the coefficients are analyzed for the Linear Regression model.

```bash
# Feature Importances from Random Forest
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf_model.feature_importances_})
print(feature_importances.sort_values(by='Importance', ascending=False))

# Linear Model Coefficients
coefficients = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': linear_model.coef_})
print(coefficients.sort_values(by='Coefficient', ascending=False))
```
## Usage
To run this project:

1. Clone the repository.
2. Install the required dependencies.
3. Obtain an API key from FRED and store it in a secure manner.
4. Run the Jupyter notebook to fetch data, train models, and analyze results.

```bash
git clone https://github.com/your-username/Home-Price-Prediction-Economic-Indicators.git
cd Home-Price-Prediction-Economic-Indicators
pip install -r requirements.txt
```

## Conclusion
The Random Forest Regression model is the best performing model, with GDP, Inflation, and Population being the most significant predictors of home prices. The relationships between features and home prices were analyzed, with some counterintuitive results likely due to multicollinearity.

## License
This project is licensed under the [MIT License](LICENSE.txt).

