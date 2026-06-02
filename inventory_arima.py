import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
weeks = 52
week_indices = np.arange(weeks)

historical_sales = {
    'CLW001': np.random.randint(40, 60, weeks),
    'CLW002': np.round(np.linspace(20, 70, weeks) + np.random.randint(-5, 5, weeks)).astype(int),
    'CLW003': np.random.randint(5, 15, weeks),
    'CLW004': np.round(20 + 15 * np.sin(2 * np.pi * week_indices / 52) + np.random.randint(-3, 3, weeks)).astype(int),
    'CLW005': np.random.randint(80, 120, weeks),
    'CLW006': np.round(np.linspace(60, 20, weeks) + np.random.randint(-5, 5, weeks)).astype(int),
    'CLW007': np.round(30 + 20 * np.sin(2 * np.pi * week_indices / 26) + np.random.randint(-3, 3, weeks)).astype(int),
    'CLW008': np.random.randint(25, 45, weeks),
    'CLW009': np.round(np.linspace(10, 80, weeks) + np.random.randint(-8, 8, weeks)).astype(int),
    'CLW010': np.random.randint(15, 25, weeks),
}

sku_attributes = {
    'CLW001': {'vendor': 'Elfin',  'order_lead_time': 4, 'oh_inventory': 180},
    'CLW002': {'vendor': 'Honey',  'order_lead_time': 6, 'oh_inventory': 80},
    'CLW003': {'vendor': 'Elfin',  'order_lead_time': 5, 'oh_inventory': 300},
    'CLW004': {'vendor': 'Honey',  'order_lead_time': 7, 'oh_inventory': 120},
    'CLW005': {'vendor': 'Elfin',  'order_lead_time': 4, 'oh_inventory': 250},
    'CLW006': {'vendor': 'Honey',  'order_lead_time': 6, 'oh_inventory': 400},
    'CLW007': {'vendor': 'Elfin',  'order_lead_time': 5, 'oh_inventory': 90},
    'CLW008': {'vendor': 'Honey',  'order_lead_time': 8, 'oh_inventory': 160},
    'CLW009': {'vendor': 'Elfin',  'order_lead_time': 4, 'oh_inventory': 50},
    'CLW010': {'vendor': 'Honey',  'order_lead_time': 7, 'oh_inventory': 200},
}

GOAL_WOC = 4
EASYCOM = 1
MOQ = 50
TRAIN_WEEKS = 44
TEST_WEEKS = 8

results = []
forecast_results = []
error_results = []

for sku, sales in historical_sales.items():

    attrs = sku_attributes[sku]
    vendor = attrs['vendor']
    order_lead_time = attrs['order_lead_time']
    oh_inventory = attrs['oh_inventory']
    total_lead_time = order_lead_time + EASYCOM

    # --- split data into train and test ---
    train_data = sales[:TRAIN_WEEKS]
    test_data = sales[TRAIN_WEEKS:]

    # --- fit model on training data only ---
    model = auto_arima(
        train_data,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )

    # --- forecast for test period (8 weeks) ---
    test_forecast = model.predict(n_periods=TEST_WEEKS)
    test_forecast = [max(1, round(f)) for f in test_forecast]

    # --- calculate error metrics ---
    actual = list(test_data)
    errors = [actual[i] - test_forecast[i] for i in range(TEST_WEEKS)]
    absolute_errors = [abs(e) for e in errors]
    squared_errors = [e**2 for e in errors]
    pct_errors = [abs(errors[i]) / actual[i] * 100 for i in range(TEST_WEEKS)]

    mae = round(sum(absolute_errors) / TEST_WEEKS, 2)
    rmse = round((sum(squared_errors) / TEST_WEEKS) ** 0.5, 2)
    mape = round(sum(pct_errors) / TEST_WEEKS, 2)

    if mape < 15:
        model_status = 'Good'
    elif mape < 25:
        model_status = 'Review'
    else:
        model_status = 'Retrain'

    # --- add error rows for each test week ---
    for i in range(TEST_WEEKS):
        error_results.append({
            'SKU': sku,
            'Week': TRAIN_WEEKS + i + 1,
            'Forecasted': test_forecast[i],
            'Actual': actual[i],
            'Error': errors[i],
            'Absolute Error': absolute_errors[i],
            'Squared Error': squared_errors[i],
            'Percentage Error (%)': round(pct_errors[i], 2),
            'MAE': mae if i == 0 else '',
            'RMSE': rmse if i == 0 else '',
            'MAPE (%)': mape if i == 0 else '',
            'Model Status': model_status if i == 0 else '',
        })

    # --- refit on full 52 weeks for live forecast ---
    full_model = auto_arima(
        sales,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )

    # --- sheet 1 forecast ---
    forecast_1week = full_model.predict(n_periods=1)
    forecasted_weekly_demand = max(1, round(forecast_1week[0]))

    woc = round(oh_inventory / forecasted_weekly_demand, 2)
    overage_underage = round(woc - GOAL_WOC, 2)
    order_sell = round((GOAL_WOC - woc) * forecasted_weekly_demand)
    if order_sell > 0:
        order_sell = max(MOQ, round(order_sell / MOQ) * MOQ)

    results.append({
        'SKU': sku,
        'Last 7 Days': int(sales[-1]),
        'OH Inventory': oh_inventory,
        'WOC': woc,
        'Goal WOC': GOAL_WOC,
        'Overage / Underage': overage_underage,
        'Order / Sell': order_sell,
        'Vendor': vendor,
        'Order Lead Time': order_lead_time,
        'Easy / .com': EASYCOM,
        'Total Lead Time': total_lead_time,
        'MOQ': MOQ,
    })

    # --- sheet 2 forecast ---
    forecast_4weeks = full_model.predict(n_periods=4)
    forecast_4weeks = [max(1, round(f)) for f in forecast_4weeks]

    week1, week2, week3, week4 = forecast_4weeks
    total_4week_demand = sum(forecast_4weeks)
    avg_weekly_demand = round(total_4week_demand / 4)

    stock_at_arrival = oh_inventory - (avg_weekly_demand * total_lead_time)
    ideal_stock_at_arrival = avg_weekly_demand * GOAL_WOC
    order_qty_needed = ideal_stock_at_arrival - max(0, stock_at_arrival)
    if order_qty_needed > 0:
        order_qty_needed = max(MOQ, round(order_qty_needed / MOQ) * MOQ)
    else:
        order_qty_needed = 0

    woc_at_arrival = round(stock_at_arrival / avg_weekly_demand, 2) if avg_weekly_demand > 0 else 0

    if stock_at_arrival <= 0:
        status = 'CRITICAL - stockout before order arrives'
    elif woc_at_arrival < GOAL_WOC:
        status = 'WARNING - order immediately'
    elif woc_at_arrival > 6:
        status = 'OVERSTOCKED - focus on selling down'
    else:
        status = 'HEALTHY'

    forecast_results.append({
        'SKU': sku,
        'Vendor': vendor,
        'OH Inventory': oh_inventory,
        'Week 1 Forecast': week1,
        'Week 2 Forecast': week2,
        'Week 3 Forecast': week3,
        'Week 4 Forecast': week4,
        'Total 4 Week Demand': total_4week_demand,
        'Avg Weekly Demand': avg_weekly_demand,
        'Total Lead Time (weeks)': total_lead_time,
        'Stock at Arrival': stock_at_arrival,
        'Ideal Stock at Arrival': ideal_stock_at_arrival,
        'Order Qty Needed': order_qty_needed,
        'WOC at Arrival': woc_at_arrival,
        'Status': status,
    })

    print(f"done: {sku} — MAPE: {mape}% — Model Status: {model_status}")

df = pd.DataFrame(results)
df_forecast = pd.DataFrame(forecast_results)
df_error = pd.DataFrame(error_results)
df_historical = pd.DataFrame(historical_sales)
df_historical.index = df_historical.index + 1
df_historical.index.name = 'Week'

with pd.ExcelWriter('inventory_report.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Inventory Status', index=False)
    df_forecast.to_excel(writer, sheet_name='Demand Forecast', index=False)
    df_error.to_excel(writer, sheet_name='Model Accuracy', index=False)
    df_historical.to_excel(writer, sheet_name='Historical Sales', index=True)

print("\nExcel file created successfully!")