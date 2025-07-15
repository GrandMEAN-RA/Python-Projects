# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 00:15:13 2025

@author: EBUNOLUWASIMI
"""

import pandas as pd
import pymysql
from sqlalchemy import create_engine
from tqdm import tqdm

tqdm.pandas()

df = pd.read_excel(r'C:\Users\EBUNOLUWASIMI\Dropbox\Study Materials\SQL\Practise Data\covid\WHO-COVID-19-global-data.csv.xlsx')
engine = create_engine('mysql+pymysql://root:israelofGOD@localhost/covid_19_updated')

chuncksize = 500
total = len(df)

print("Rows: ",total)

df.columns = (
    df.columns
    .str.strip()                 # remove leading/trailing spaces
    .str.replace(' ', '_')      # replace spaces with underscores
    .str.replace(r'\W+', '', regex=True)  # remove non-alphanumeric
)

df.head(0).to_sql('weeklydata', con=engine, if_exists='replace', index=False)


for i in tqdm(range(0,total,chuncksize),desc="Uploading in chuncks..."):
    chunck = df.iloc[i:i+chuncksize]
    chunck.to_sql('weeklydata',con=engine,if_exists='append',index=False)

print('Data import successful!')
