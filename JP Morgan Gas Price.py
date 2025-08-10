# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 20:32:35 2025

@author: EBUNOLUWASIMI
"""
# Import needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from prophet import Prophet
import math

#Load CSV data
data = pd.read_csv(r"C:\Users\EBUNOLUWASIMI\Dropbox\Study Materials\Python\JP Morgan Chase\Nat_Gas.csv")

#Ensure data type consistency
data['Dates'] = pd.to_datetime(data['Dates'])
data = data.sort_values('Dates').reset_index(drop=True)

# Check and visualise raw data
print(data.to_string())
print(data.describe())

plt.figure(figsize=(10, 6))
plt.plot(data['Dates'], data['Prices'], marker='o')
plt.title("Natural Gas Monthly Prices")
plt.xlabel("Dates")
plt.ylabel("Prices")
plt.grid(True)
plt.show()

start_date = datetime(2020,10,31)
end_date = datetime(2024,9,30)
months = []
year = start_date.year
month = start_date.month + 1
while True:
    current = datetime(year, month, 1) + timedelta(days=-1)
    months.append(current)
    if current.month == end_date.month and current.year == end_date.year:
        break
    else:
        month = ((month + 1) % 12) or 12
        if month == 1:
            year += 1
        
days_from_start = [(day - start_date ).days for day in months]
time = np.array(days_from_start)

plt.figure(figsize=(10, 6))
plt.plot(time, data['Prices'], marker='o')
plt.xlabel('Days from start date')
plt.ylabel('Price')
plt.title('Linear Trend of Monthly Input Prices')
plt.grid(True)
plt.show()

# Use ARIMA and Prophet to handle seasonality
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

def valuer(indates, costprice, outdates, sellprice, iwrate, storagecost, maxvolume, iwcost,hcost):
    volume = 0
    cashout = 0
    cashin = 0
    
    # Ensure dates are in sequence
    dates = sorted(set(indates + outdates))
    
    for i in range(len(dates)):
        # processing code for each date
        startdate = dates[i]

        if startdate in indates:
            # Inject on these dates and sum up cash flows
            if volume <= maxvolume - iwrate:
                volume += iwrate

                # Cost to purchase gas
                cashout += iwrate * costprice[indates.index(startdate)]
                # Injection cost
                injectioncost = (iwrate * iwcost) + hcost
                cashout += injectioncost
                print('Injected gas on %s at a price of %s'%(startdate, costprice[indates.index(startdate)]))

            else:
                # We do not want to inject when rate is greater than total volume minus volume
                print('Injection is not possible on date %s as there is insufficient space in the storage facility'%startdate)
        elif startdate in outdates:
            #Withdraw on these dates and sum cash flows
            if volume >= iwrate:
                volume -= iwrate
                cashin += iwrate * sellprice[outdates.index(startdate)]
                # Withdrawal cost
                withdrawalcost = (iwrate * iwcost) + hcost
                cashin -= withdrawalcost
                print('Extracted gas on %s at a price of %s'%(startdate, sellprice[outdates.index(startdate)]))
            else:
                # we cannot withdraw more gas than is actually stored
                print('Extraction is not possible on date %s as there is insufficient volume of gas stored'%startdate)
                
    storecost = math.ceil((max(outdates) - min(indates)).days // 30) * storagecost
    return cashin - storecost - cashout

indates = []
costprice = []
outdates = []
sellprice = []

while True:
    print("Please, input date value in the format 'YYYY-MM-DD'")
    prompt = input("input date or type end to continue")
    if prompt in ("end","e","\n"," "): 
        break
    else:
        date = pd.to_datetime(prompt)
        status = input("Input transaction type: injection or withdrawal")
        if status in ("injection","in","inj"):
            indates.append(date)
            costprice.append(float(get_price(date)))
        elif status in ("withdrawal","with","wd"):
            outdates.append(date)
            sellprice.append(float(get_price(date)))

iwrate = float(input("Injection/Withdrawal Rate?"))    # rate of gas in cubic feet per day
storagecost = float(input("Storage cost?"))  # cost of storing total volume
iwcost = float(input("Injection/Withdrawal Cost per iwrate?")) # $/cf
hcost = float(input("Roundtrip Haulage Cost?")) #cost of haulage
maxvolume = float(input("Maximum Storage Volume?")) # maximum storage capacity of the storage facility

result = valuer(indates, sellprice, outdates, costprice, iwrate, storagecost, maxvolume, iwcost,hcost)
print()
print(f"The value of the contract is: ${result}")

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















