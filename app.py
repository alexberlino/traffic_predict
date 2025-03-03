import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.io as pio

# Define file paths
data_directory = './data'
rankings_path = os.path.join(data_directory, 'rankings.csv')
queries_path = os.path.join(data_directory, 'queries.csv')
categories_path = os.path.join(data_directory, 'categories.csv')

# Function to check file existence


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} was not found.")


# Check if files exist
check_file_exists(rankings_path)
check_file_exists(queries_path)
check_file_exists(categories_path)

# Load the provided CSV files with error handling
try:
    rankings_df = pd.read_csv(rankings_path)
    queries_df = pd.read_csv(queries_path)
    categories_df = pd.read_csv(categories_path)
except Exception as e:
    print(f"An error occurred while reading the files: {e}")
    raise

# Display the first few rows of each dataframe to understand their structure
print("Rankings DataFrame Columns:", rankings_df.columns)
print(rankings_df.head())
print("Queries DataFrame Columns:", queries_df.columns)
print(queries_df.head())
print("Categories DataFrame Columns:", categories_df.columns)
print(categories_df.head())

# Clean and preprocess the rankings data
rankings_df.columns = rankings_df.columns.str.strip()

# Verify column names after stripping
print("Cleaned Rankings DataFrame Columns:", rankings_df.columns)

# Convert 'Rank' column to numeric, coercing errors to NaN
rankings_df['Rank'] = pd.to_numeric(rankings_df['Rank'], errors='coerce')

# Handle NaN values in 'Rank' column before conversion
rankings_df['Rank'] = rankings_df['Rank'].fillna(100).astype(int)

# Verify the data type and handle cases where 'Rank' could be greater than 100
rankings_df['Rank'] = rankings_df['Rank'].apply(
    lambda x: 100 if x > 100 else x)

# Clean and preprocess the traffic data
queries_df.columns = queries_df.columns.str.strip()
queries_df['Top queries'] = queries_df['Top queries'].str.strip()

# Convert CTR to numeric by removing the '%' and dividing by 100
queries_df['CTR'] = queries_df['CTR'].str.rstrip('%').astype('float') / 100

# Verify the sum of clicks before dividing by 3
total_clicks_3_months = queries_df['Clicks'].sum()
print(f'Total clicks for 3 months: {total_clicks_3_months}')

# Divide the Clicks data by 3 to get monthly traffic
queries_df['Clicks'] = queries_df['Clicks'] / 3

# Verify the sum of clicks after dividing by 3
total_clicks_monthly = queries_df['Clicks'].sum()
print(f'Total clicks per month: {total_clicks_monthly}')

# Merge the datasets on the keywords
merged_df = pd.merge(rankings_df, queries_df, how='left',
                     left_on='Keyword', right_on='Top queries')

# Adjust the merge to include the condition for impressions
merged_df['Rank'] = merged_df.apply(
    lambda row: row['Rank'] if row['Impressions'] >= 1000 else row['Rank'], axis=1)
merged_df['CTR'] = merged_df.apply(
    lambda row: row['CTR'] if row['Impressions'] >= 1000 else np.nan, axis=1)
merged_df['Position'] = merged_df.apply(
    lambda row: row['Position'] if row['Impressions'] >= 1000 else np.nan, axis=1)

# Rename columns in categories_df
categories_df.columns = categories_df.columns.str.strip()
categories_df.rename(columns={
    'Main Keyword': 'Keyword',
    'Brand': 'Category',
    'Clicks': 'Category_Clicks',
    'Impressions': 'Category_Impressions',
    'Average Position': 'Category_Average_Position',
    'Search Volume': 'Volume'
}, inplace=True)

# Merge with categories data
merged_df = pd.merge(merged_df, categories_df, how='left', on='Keyword')

# Display the adjusted merged dataframe
print(merged_df.head())

# Handle missing values by filling NaNs with 0
merged_df['Clicks'].fillna(0, inplace=True)

# Estimate clicks for terms without GSC data based on category-specific CTR


def estimate_ctr(rank, keyword_type):
    if keyword_type == 'brand':
        if rank == 1:
            return 0.01
        elif rank <= 3:
            return 0.01
        elif rank <= 10:
            return 0.005
        elif rank <= 20:
            return 0.0025
        else:
            return 0.001
    elif keyword_type == 'semibrand':
        if rank == 1:
            return 0.05
        elif rank <= 3:
            return 0.05
        elif rank <= 10:
            return 0.03
        elif rank <= 20:
            return 0.015
        else:
            return 0.005
    else:  # other
        if rank == 1:
            return 0.30
        elif rank <= 3:
            return 0.15
        elif rank <= 10:
            return 0.10
        elif rank <= 20:
            return 0.05
        else:
            return 0.01


merged_df['Estimated_CTR'] = merged_df.apply(
    lambda row: row['CTR'] if not pd.isna(row['CTR']) else estimate_ctr(row['Rank'], row['Category']), axis=1)

# Ensure to use the correct volume column
merged_df['Volume'] = merged_df['Volume_x'].combine_first(
    merged_df['Volume_y'])

merged_df['Estimated_Clicks'] = merged_df.apply(
    lambda row: row['Clicks'] if row['Clicks'] > 0 else row['Volume'] * row['Estimated_CTR'], axis=1)

# Convert Estimated_Clicks to numeric
merged_df['Estimated_Clicks'] = pd.to_numeric(merged_df['Estimated_Clicks'])

# Verify the sum of current estimated clicks
current_clicks_estimated = merged_df['Estimated_Clicks'].sum()
print(f'Current estimated clicks per month: {current_clicks_estimated}')

# Prepare the data for the model
X = merged_df[['Rank', 'Volume']]
y = merged_df['Estimated_Clicks']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Root Mean Squared Error: {rmse}')

# Function to simulate traffic increase based on rank improvements


def simulate_traffic_increase(data, improvement_factor):
    simulated_data = data.copy()
    if improvement_factor == 0:
        simulated_data['New_Rank'] = simulated_data['Rank']
        simulated_data['New_Estimated_CTR'] = simulated_data['Estimated_CTR']
        simulated_data['New_Estimated_Clicks'] = simulated_data['Estimated_Clicks']
        simulated_data['Predicted_Clicks'] = simulated_data['Estimated_Clicks']
    else:
        rank_improvement = simulated_data['Rank'] * improvement_factor
        simulated_data['New_Rank'] = simulated_data['Rank'] - rank_improvement
        simulated_data['New_Rank'] = simulated_data['New_Rank'].apply(
            lambda x: max(x, 1))  # Ensure rank doesn't go below 1
        simulated_data['New_Estimated_CTR'] = simulated_data.apply(
            lambda row: estimate_ctr(row['New_Rank'], row['Category']), axis=1)
        simulated_data['New_Estimated_Clicks'] = simulated_data['Volume'] * \
            simulated_data['New_Estimated_CTR']

        # Use the trained model to predict clicks based on the new rank
        simulated_data_temp = simulated_data.copy()
        simulated_data_temp['Rank'] = simulated_data_temp['New_Rank']
        X_pred = simulated_data_temp[['Rank', 'Volume']]
        simulated_data['Predicted_Clicks'] = simulated_data['New_Estimated_Clicks']

    return simulated_data[['Keyword', 'Rank', 'New_Rank', 'Volume', 'Estimated_CTR', 'New_Estimated_CTR', 'Estimated_Clicks', 'Predicted_Clicks', 'New_Estimated_Clicks']]

# Function to plot traffic increase with Plotly


def plot_traffic_increase(improvement_factor):
    simulated_data = simulate_traffic_increase(merged_df, improvement_factor)
    current_clicks = merged_df['Estimated_Clicks'].sum()
    total_clicks = simulated_data['Predicted_Clicks'].sum()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=['Current Monthly Clicks',
            f'Monthly Clicks with Improvement Factor {improvement_factor}'],
        y=[current_clicks, total_clicks],
        marker_color=['blue', 'green']
    ))

    fig.update_layout(
        title='Traffic Prediction Based on Rank Improvement',
        yaxis_title='Total Monthly Clicks',
        yaxis_tickformat='.2f'  # Format y-axis labels to 2 decimal places
    )

    return fig

# Create frames for the animation
frames = []
for i in np.arange(0, 1.01, 0.01):
    frames.append(go.Frame(data=plot_traffic_increase(i).data, name=str(i)))

# Create the initial figure
fig = plot_traffic_increase(0)
fig.frames = frames

# Add slider only (remove play/pause buttons)
fig.update_layout(
    sliders=[{
        "steps": [{"args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                   "label": str(i),
                   "method": "animate"} for i in np.arange(0, 1.01, 0.01)]
    }]
)

# Save the animation as an HTML file
pio.write_html(fig, file='traffic_prediction_animation.html', auto_open=True)

# Simulate traffic increase for factor 1
simulated_data_factor_1 = simulate_traffic_increase(merged_df, 1)
print(simulated_data_factor_1[['Keyword', 'Rank', 'New_Rank', 'Volume',
      'Estimated_Clicks', 'Predicted_Clicks', 'New_Estimated_Clicks']])

# Export the results to a CSV file
simulated_data_factor_1.to_csv('simulated_traffic_increase.csv', index=False)
print("Exported the results to 'simulated_traffic_increase.csv'")
