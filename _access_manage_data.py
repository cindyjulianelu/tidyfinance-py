# _access_manage_data.py
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import sqlite3

# Define date variables, range of data
start_date = "1960-01-01"
end_date = "2022-12-31"

# Fama French Data
factors_ff3_monthly_raw = pdr.DataReader(
  
)
factors_ff3_monthly = (factors_ff3_monthly_raw

)
factors_ff5_monthly_raw = pdr.DataReader(
  
)
factors_ff5_monthly = (factors_ff5_monthly_raw

)
factors_ff3_daily_raw = pdr.DataReader(
  
)[0]
factors_ff3_daily = (factors_ff3_daily_raw

)
industries_ff_monthly_raw = pdr.DataReader(
  
)[0]
industries_ff_monthly = (industries_ff_monthly_raw

)
pdf.famafrench.get_available_datasets()

# Q Factors
factors_q_monthly_link = (
  
)
factors_q_monthly = (pd.read_csv(factors_q_monthly_link)

)

# Macro predictors
sheet_id = ""
sheet_name = ".xlsx"
macro_predictors_link = (
  
)

# Now transform the data for macro predictors
macro_predictors = (
  
)

# Other Macro data
cpi_monthly = (pdr.DataReader(
  
))

# Setting up database
tidy_finance = sqlite3.connect(
  database = ""
)
(factors_ff3_monthly.to_sql(
  
))
pd.read_sql_query(
  sql = "", 
  con = tidy_finance, 
  parse_dates = {"month"}
)

# Storing all other data in database
data_dict = {
  "factors_ff5_monthly": factors_ff5_monthly, 
  "factors_ff3_daily" : factors_ff3_daily, 
  "industries_ff_monthly" : industries_ff_monthly, 
  "factors_q_montly" : factors_q_monthly, 
  "macro_predictors" : macro_predictors, 
  "cpi_monthly" : cpi_monthly
}
for key, value in data_dict.items():
  value.to_sql(name = key, 
  con = tidy_finance, 
  if_exists = "replace", index=False)

# These are the steps to follow after setup
import pandas as pd
import sqlite3
tidy_finance = sqlite.connect(
  database = "data/tidy_finance_python.sqlite"
)
factors_q_monthly = pd.read_sql_query(
  sql = "", 
  con = tidy_finance,
  parse_dates = {"month"}
)

# To optimize database 
tidy_finance.execute("VACUUM")

# Exercises
