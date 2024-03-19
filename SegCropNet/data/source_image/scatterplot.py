import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import linregress

# Load your data from the xlsx file
df = pd.read_excel(r'D:\PythonProject\lanenet-lane-detection-pytorch-main\data\source_image\pre.xlsx')

# Extract the real and predicted values
real_values = df.iloc[:, 0]
predicted_values = df.iloc[:, 1]

# Scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=real_values, y=predicted_values, color='blue', alpha=0.7)

# Add a regression line with confidence bands
slope, intercept, r_value, p_value, std_err = linregress(real_values, predicted_values)
line = slope * real_values + intercept
plt.plot(real_values, line, color='red', linewidth=2)

# Calculate R^2 and RMSE
r_squared = r_value**2
rmse = np.sqrt(mean_squared_error(real_values, predicted_values))

# Add labels and title
plt.title('Scatterplot with Regression Line and Confidence Bands', fontname='Times New Roman', fontsize=16)
plt.xlabel('Real Values', fontname='Times New Roman', fontsize=14)
plt.ylabel('Predicted Values', fontname='Times New Roman', fontsize=14)

# Add fit coefficients information
text = f'R^2: {r_squared:.4f}\nRMSE: {rmse:.4f}'
plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontname='Times New Roman', fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Show the plot
plt.show()
