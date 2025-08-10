# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 20:32:35 2025

@author: EBUNOLUWASIMI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from prophet import Prophet

data = pd.read_csv(r"C:\Users\EBUNOLUWASIMI\Dropbox\Study Materials\Python\JP Morgan Chase\Nat_Gas.csv")
data['Dates'] = pd.to_datetime(data['Dates'])
data = data.sort_values('Dates').reset_index(drop=True)

print(data.to_string())
print(data.describe())

plt.figure(figsize=(10, 6))
plt.plot(data['Dates'], data['Prices'], marker='o')
plt.title("Natural Gas Monthly Prices")
plt.xlabel("Dates")
plt.ylabel("Prices")
plt.grid(True)
plt.show()

prophet_df = data.rename(columns={'Dates': 'ds', 'Prices': 'y'})
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False)
model.fit(prophet_df)

future = model.make_future_dataframe(periods=12, freq='ME')
forecast = model.predict(future)
forecast_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'Dates', 'yhat': 'Prices'})

historical_df = forecast_df[forecast_df['Dates'] <= data['Dates'].max()]
future_df = forecast_df[forecast_df['Dates'] > data['Dates'].max()]

def estimate_past_price(target_date):
    target_date = pd.to_datetime(target_date)
    return np.interp(
        x=pd.Timestamp(target_date).toordinal(),
        xp=data['Dates'].map(datetime.toordinal),
        fp=data['Prices']
    )

combined_df = pd.concat([historical_df, future_df], ignore_index=True)

def get_price(date):
    date = pd.to_datetime(date)
    if date <= data['Dates'].max():
        # Interpolate for past dates
        return estimate_past_price(date)
    elif date <= combined_df['Dates'].max():
        # Look up from forecast
        return np.interp(
            x=date.toordinal(),
            xp=combined_df['Dates'].map(datetime.toordinal),
            fp=combined_df['Prices']
        )
    else:
        return "Date out of forecast range"

plt.figure(figsize=(12, 6))
plt.plot(data['Dates'], data['Prices'], label='Historical', marker='o')
plt.plot(forecast_df['Dates'], forecast_df['Prices'], label='Forecast', marker='x', linestyle='--')
plt.axvline(data['Dates'].max(), color='gray', linestyle=':', label='Forecast Start')
plt.title("Natural Gas Price Forecast")
plt.xlabel("Dates")
plt.ylabel("Prices")
plt.legend()
plt.grid(True)
plt.show()


date = input("input a date value in the format 'YYYY-MM-DD'")
edate = get_price(date)
print("Price on " + date +":", round(edate, 2))











