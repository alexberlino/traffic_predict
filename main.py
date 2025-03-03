import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Load the results data (assuming results are already generated)
# If not, we'll create sample data
try:
    results = pd.read_csv('./data/traffic_predictions.csv')
    # Ensure Date is in datetime format
    results['Date'] = pd.to_datetime(results['Date'])
except:
    # Sample data for demonstration
    dates = pd.date_range(start='2023-01-01', periods=100)
    actual = np.random.normal(100, 20, 100)
    predicted = actual + np.random.normal(0, 10, 100)
    results = pd.DataFrame({
        'Date': dates,
        'Actual Clicks': actual,
        'Predicted Clicks': predicted
    })

# Sort results by date for proper time series visualization
results = results.sort_values('Date')

# Calculate the error/difference
results['Difference'] = results['Actual Clicks'] - results['Predicted Clicks']
results['Absolute Difference'] = abs(results['Difference'])

# Create a figure with two panels
fig, axes = plt.subplots(2, 1, figsize=(
    12, 10), gridspec_kw={'height_ratios': [1, 1]})

# 1. Linear comparison graph showing actual and predicted side by side
indexes = np.arange(len(results))
bar_width = 0.35

axes[0].bar(indexes - bar_width/2, results['Actual Clicks'],
            bar_width, label='Actual Clicks', color='#1f77b4')
axes[0].bar(indexes + bar_width/2, results['Predicted Clicks'],
            bar_width, label='Predicted Clicks', color='#ff7f0e')

axes[0].set_title('Actual vs Predicted Clicks Comparison', fontsize=16)
axes[0].set_ylabel('Number of Clicks', fontsize=12)
axes[0].set_xticks(indexes)

# Format x-axis to show dates at reasonable intervals
if len(results) > 20:
    num_dates_to_show = min(20, len(results))
    step_size = max(1, len(results) // num_dates_to_show)
    indices_to_show = indexes[::step_size]
    dates_to_show = results['Date'].iloc[::step_size].dt.strftime('%Y-%m-%d')
    axes[0].set_xticks(indices_to_show)
    axes[0].set_xticklabels(dates_to_show, rotation=45, ha='right')
else:
    axes[0].set_xticklabels(results['Date'].dt.strftime(
        '%Y-%m-%d'), rotation=45, ha='right')

axes[0].grid(True, alpha=0.3, axis='y')
axes[0].legend(fontsize=12)

# 2. Difference plot as a line chart
axes[1].plot(indexes, results['Difference'], marker='o', markersize=4,
             linewidth=2, color='#d62728', label='Difference (Actual - Predicted)')
axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[1].set_title(
    'Difference Between Actual and Predicted Clicks', fontsize=16)
axes[1].set_ylabel('Difference', fontsize=12)
axes[1].set_xlabel('Date', fontsize=12)

# Match x-axis ticks to the first plot
axes[1].set_xticks(axes[0].get_xticks())
axes[1].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=12)

# Add statistics to the second plot
mean_difference = results['Difference'].mean()
median_difference = results['Difference'].median()
max_over_prediction = results['Difference'].min()
max_under_prediction = results['Difference'].max()
mean_absolute_diff = results['Absolute Difference'].mean()

stats_text = (f"Mean Difference: {mean_difference:.2f}\n"
              f"Median Difference: {median_difference:.2f}\n"
              f"Mean Absolute Difference: {mean_absolute_diff:.2f}\n"
              f"Max Over-prediction: {-max_over_prediction:.2f}\n"
              f"Max Under-prediction: {max_under_prediction:.2f}")

axes[1].annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3",
                           fc="white", ec="gray", alpha=0.8),
                 fontsize=10)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('./data/linear_comparison.png', dpi=300, bbox_inches='tight')

print("Linear comparison visualization created and saved as 'linear_comparison.png'")

# Create a second visualization showing the cumulative difference over time
plt.figure(figsize=(12, 6))

# Calculate cumulative actual and predicted
results['Cumulative Actual'] = results['Actual Clicks'].cumsum()
results['Cumulative Predicted'] = results['Predicted Clicks'].cumsum()
results['Cumulative Difference'] = results['Cumulative Actual'] - \
    results['Cumulative Predicted']

plt.plot(results['Date'], results['Cumulative Actual'],
         label='Cumulative Actual', linewidth=2)
plt.plot(results['Date'], results['Cumulative Predicted'],
         label='Cumulative Predicted', linewidth=2, linestyle='--')
plt.plot(results['Date'], results['Cumulative Difference'],
         label='Cumulative Difference', linewidth=2, color='red')

plt.title('Cumulative Clicks and Difference Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Clicks', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

# Format x-axis for readability
if len(results) > 20:
    num_dates_to_show = 10
    date_indices = np.linspace(0, len(results)-1, num_dates_to_show, dtype=int)
    plt.xticks(results['Date'].iloc[date_indices],
               results['Date'].iloc[date_indices].dt.strftime('%Y-%m-%d'),
               rotation=45, ha='right')

# Save this figure as well
plt.savefig('./data/cumulative_comparison.png', dpi=300, bbox_inches='tight')
plt.tight_layout()

print("Cumulative comparison visualization created and saved as 'cumulative_comparison.png'")
