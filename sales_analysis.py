# sales_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# 1. Load Dataset
df = pd.read_csv("Superstore.csv", encoding='latin-1')

# 2. Data Preprocessing
df['Order Date'] = pd.to_datetime(df['Order Date'])
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# 3. Descriptive Business Analytics
print("Total Sales: ", df['Sales'].sum())
print("Total Profit: ", df['Profit'].sum())

# Monthly sales trend
monthly_sales = df.groupby(df['Order Date'].dt.to_period('M')).sum(numeric_only=True)['Sales']
monthly_sales.plot(kind='line', figsize=(10,5), title='Monthly Sales Trend')
plt.show()

# Top selling products
top_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
top_products.plot(kind='bar', figsize=(10,5), title='Top 10 Selling Products')
plt.show()

# Region-wise sales
region_sales = df.groupby('Region')['Sales'].sum()
region_sales.plot(kind='pie', autopct='%1.1f%%', figsize=(6,6), title='Sales by Region')
plt.show()

# 4. Customer Segmentation
customer_sales = df.groupby('Customer Name')['Sales'].sum()
df['Customer Segment'] = df['Customer Name'].map(
    lambda x: 'High Value' if customer_sales[x] > 5000 else ('Medium Value' if customer_sales[x] > 2000 else 'Low Value')
)
print(df[['Customer Name', 'Customer Segment']].drop_duplicates().head())

# 5. Predictive Analytics - Sales Forecast
sales_ts = monthly_sales.to_timestamp()
model = ARIMA(sales_ts, order=(2,1,2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=6)

plt.figure(figsize=(10,5))
plt.plot(sales_ts, label='Historical Sales')
plt.plot(pd.date_range(sales_ts.index[-1] + pd.DateOffset(months=1), periods=6, freq='M'), forecast, label='Forecast')
plt.title('6-Month Sales Forecast')
plt.legend()
plt.show()

# 6. Business Insights
print("\n--- Business Insights ---")
print("1. Focus on top 3 regions as they contribute the most to revenue.")
print("2. Promote top-selling products and bundle them with slower movers.")
print("3. Target 'Medium Value' customers with special discounts to move them to 'High Value'.")
print("4. Prepare inventory for forecasted high-demand months.")
