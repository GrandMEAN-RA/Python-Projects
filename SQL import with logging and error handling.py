# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 14:04:52 2025

@author: EBUNOLUWASIMI
"""

import pandas as pd
import pymysql
from sqlalchemy import create_engine
from tqdm import tqdm
import logging

# Logging setup
logging.basicConfig(filename='data_import.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Database connection
engine = create_engine('mysql+pymysql://USERNAME:PASSWORD@localhost/DATABASE')

# Read Excel file
df = pd.read_excel(r'C:\Users\EBUNOLUWASIMI\Dropbox\Study Materials\Alex the analyst\PortfolioProjects-main\PortfolioProjects-main\Nashville Housing Data for Data Cleaning (reuploaded).xlsx')

df.columns = (
    df.columns
    .str.strip()                 # remove leading/trailing spaces
    .str.replace(' ', '_')      # replace spaces with underscores
    .str.replace(r'\W+', '', regex=True)  # remove non-alphanumeric
)

# Basic validation function
def validate_row(row):
    try:
        # Example checks (customize as needed)
        assert pd.notnull(row['Country'])              # Country name must exist
        assert pd.notnull(row['IndexValue'])           # No null index values
        assert isinstance(row['IndexValue'], (int, float))  # Value must be numeric
        return True
    except Exception as e:
        logging.warning(f"Invalid row skipped: {row.to_dict()} | Error: {e}")
        return False

# Filter valid rows
valid_df = df[df.apply(validate_row, axis=1)].copy()
print(f"Total rows: {len(df)} | Valid rows: {len(valid_df)}")

# Create table structure once
valid_df.head(0).to_sql('countryindices', con=engine, if_exists='replace', index=False)

# Insert in chunks with append and error handling
chunk_size = 500
inserted = 0
failed = 0

for i in tqdm(range(0, len(valid_df), chunk_size), desc="Uploading in chunks..."):
    chunk = valid_df.iloc[i:i + chunk_size]
    try:
        chunk.to_sql('countryindices', con=engine, if_exists='append', index=False)
        inserted += len(chunk)
    except Exception as e:
        failed += len(chunk)
        logging.error(f"Failed to insert chunk starting at row {i}: {e}")

print(f"✅ Data import complete! Inserted: {inserted} | Failed chunks: {failed}")
