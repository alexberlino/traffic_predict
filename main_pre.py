import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step 1: Load your traffic data
data = pd.read_csv('./data/mbtraffic.csv')

# Convert 'Date' column to datetime
# 'coerce' will turn errors into NaT (Not a Time)
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
# Convert to the number of days since the first date
data['Date'] = (data['Date'] - data['Date'].min()).dt.days

# Convert 'CTR' column to numeric, forcing errors to NaN
data['CTR'] = pd.to_numeric(data['CTR'], errors='coerce')

# Debug: Check for NaN values after initial transformations
print("Initial data with NaNs:")
print(data.isna().sum())  # This will show the count of NaNs per column

# Handle missing values (fill NaN with the mean for now)
# For 'CTR' and 'Position', fill NaNs with the mean
data['CTR'] = data['CTR'].fillna(data['CTR'].mean())
data['Position'] = data['Position'].fillna(data['Position'].mean())
data['Impressions'] = data['Impressions'].fillna(data['Impressions'].mean())
data['Clicks'] = data['Clicks'].fillna(data['Clicks'].mean())

# Debug: Check for NaN values after filling
print("\nData after filling NaNs:")
# This will show the count of NaNs per column after filling
print(data.isna().sum())

# Step 2: Convert 'Date' to datetime format (already done earlier)
data['Date'] = pd.to_datetime(data['Date'])

# Step 3: Feature Engineering
data['day_of_week'] = data['Date'].dt.dayofweek
data['month'] = data['Date'].dt.month
data['week_of_year'] = data['Date'].dt.isocalendar().week

# Debug: Check for NaNs after feature engineering
print("\nData after feature engineering:")
# This will show the count of NaNs per column after feature engineering
print(data.isna().sum())

# Step 4: Prepare Features (X) and Target (y)
X = data[['Impressions', 'CTR', 'Position',
          'day_of_week', 'month', 'week_of_year']]

# Dropping columns not needed for the prediction
X = data.drop(columns=['Date', 'Clicks', 'CTR', 'Position'])

y = data['Clicks']

# Debug: Check for NaNs before splitting into train/test
print("\nX before train-test split:")
print(X.isna().sum())  # This will show the count of NaNs in features

# Step 5: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Debug: Check for NaNs in the train/test split
print("\nX_train NaNs:")
print(X_train.isna().sum())  # Check for NaNs in training data
print("\nX_test NaNs:")
print(X_test.isna().sum())  # Check for NaNs in testing data

# Step 6: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Debug: Check for NaNs after scaling
print("\nX_train_scaled NaNs:")
# Check for NaNs in the scaled training data
print(np.isnan(X_train_scaled).sum())
print("\nX_test_scaled NaNs:")
# Check for NaNs in the scaled testing data
print(np.isnan(X_test_scaled).sum())

# Step 7: Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 8: Predict and Evaluate the Model
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print evaluation metrics
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# Step 9: Save the Predicted Results to CSV
results = pd.DataFrame({
    'Date': data['Date'].iloc[y_test.index],
    'Actual Clicks': y_test,
    'Predicted Clicks': y_pred
})
results.to_csv('./data/traffic_predictions_pre.csv', index=False)

# Step 10: Save the Evaluation Metrics to a Text File
with open('./data/evaluation_metrics_pre.txt', 'w') as f:
    f.write(f"Mean Absolute Error: {mae}\n")
    f.write(f"Root Mean Squared Error: {rmse}\n")

# Step 11: Save the Visualization as a PNG Image
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Actual Clicks", color='blue')
plt.plot(y_pred, label="Predicted Clicks", color='red', linestyle='dashed')
plt.title("Actual vs Predicted Clicks")
plt.xlabel("Time")
plt.ylabel("Clicks")
plt.legend()
plt.savefig('./data/actual_vs_predicted_clicks_pre.png')
plt.close()
