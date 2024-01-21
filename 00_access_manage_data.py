# 00_access_manage_data.py
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import sqlite3

# Define date variables, range of data
start_date = "1960-01-01"
end_date = "2022-12-31"

# Fama French Data
factors_ff3_monthly_raw = pdr.DataReader(
  name = "F-F_Research_Data_Factors", 
  data_source = "famafrench", 
  start = start_date, 
  end = end_date)[0]
factors_ff3_monthly = (factors_ff3_monthly_raw.divide(
  100
).reset_index(names = "month"
).assign(month = lambda x : pd.to_datetime(
  x["month"].astype(str))
).rename(str.lower, axis = "columns"
).rename(columns = {"mkt-rf" : "mkt_excess"})
)
factors_ff5_monthly_raw = pdr.DataReader(
  name = "F-F_Research_Data_5_Factors_2x3", 
  data_source = "famafrench", 
  start = start_date,
  end = end_date)[0]
factors_ff5_monthly = (factors_ff5_monthly_raw.divide(100
).reset_index(
  names = "month"
).assign(
  month = lambda x: pd.to_datetime(x['month'].astype(str))
).rename(
  str.lower, axis = 'columns'
).rename(
  columns = {"mkr-rf" : "mkt_excess"}
)
)
factors_ff3_daily_raw = pdr.DataReader(
  name = "F-F_Research_Data_Factors_daily",
  data_source = "famafrench",
  start = start_date, end = end_date)[0]
factors_ff3_daily = (factors_ff3_daily_raw.divide(
  100
).reset_index(
  names = "date"
).rename(
  str.lower, axis = "columns"
).rename(
  columns = {"mkt-rf": "mkt_excess"}
)
)
industries_ff_monthly_raw = pdr.DataReader(
  name = "10_Industry_Portfolios",
  data_source = "famafrench", 
  start = start_date,
  end = end_date)[0]
industries_ff_monthly = (industries_ff_monthly_raw.divide(
  100
).reset_index(
  names = "month"
).assign(
  month = lambda x: pd.to_datetime(x['month'].astype(str))
).rename(str.lower, axis = "columns")
)
pdr.famafrench.get_available_datasets()

# Q Factors
factors_q_monthly_link = (
  "https://global-q.org/uploads/1/2/2/6/122679606/"
  "q5_factors_monthly_2022.csv"
)
factors_q_monthly = (pd.read_csv(factors_q_monthly_link).assign(
  month = lambda x: (
    pd.to_datetime(x['year'].astype(str) + "-" + 
    x["month"].astype(str) + "-01")
  )
).drop(
  columns = ["R_F", "R_MKT", "year"]
).rename(
  columns = lambda x: x.replace("R_", "").lower()
).query(f"month >= '{start_date}' and month <= '{end_date}'"
).assign(
 **{col : lambda x: x[col]/100 for col in 
 ["me", "ia", "roe", "eg"]} 
)
)

# Macro predictors
sheet_id = "1g4LOaRj4TvwJr9RIaA_nwrXXWTOy46bP"
sheet_name = "macro_predictors.xlsx"
macro_predictors_link = (
  f"https://docs.google.com/spreadsheets/d/{sheet_id}" 
  f"/gviz/tq?tqx=out:csv&sheet={sheet_name}"
)

# Now transform the data for macro predictors
macro_predictors = (
  pd.read_csv(macro_predictors_link, thousands = ",").assign(
    month = lambda x: pd.to_datetime(x["yyyymm"], 
    format = "%Y%m"),
    dp = lambda x: np.log(x["D12"]) - np.log(x["Index"]),
    dy = lambda x: np.log(x["D12"]) - np.log(x["D12"].shift(1)),
    ep = lambda x: np.log(x["E12"]) - np.log(x["Index"]),
    de = lambda x: np.log(x["D12"]) - np.log(x["E12"]),
    tms = lambda x: x["lty"] - x["tbl"],
    dfy = lambda x: x["BAA"] - x["AAA"]
  ).rename(
    columns = {"b/m" : "bm"}
  ).get(
    ["month", "dp", "dy", "ep", "de", "svar", 
    "bm", "ntis", "tbl", "lty", "ltr", "tms", "dfy", 
    "infl"]
  ).query(
    "month >= @start_date and month <= @end_date"
  ).dropna()
)

# Other Macro data
cpi_monthly = (pdr.DataReader(
  name = "CPIAUCNS", 
  data_source = "fred", 
  start = start_date, end = end_date
).reset_index(names = "month"
).rename(
  columns = {"CPIAUCNS" : "cpi"}
).assign(
  cpi = lambda x: x['cpi']/x["cpi"].iloc[-1]
))

# Set database - make sure to create /data folder in same directory before
tidy_finance = sqlite3.connect(
  database = "data/tidy_finance_python.sqlite"
)
(factors_ff3_monthly.to_sql(
  name = "factors_ff3_monthly", 
  con = tidy_finance, 
  if_exists = "replace",
  index = False
))
pd.read_sql_query(
  sql = "SELECT month, rf FROM factors_ff3_monthly", 
  con = tidy_finance, 
  parse_dates = {"month"}
)

# Storing all other data in database
data_dict = {
  "factors_ff5_monthly": factors_ff5_monthly, 
  "factors_ff3_daily" : factors_ff3_daily, 
  "industries_ff_monthly" : industries_ff_monthly, 
  "factors_q_monthly" : factors_q_monthly, 
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
tidy_finance = sqlite3.connect(
  database = "data/tidy_finance_python.sqlite"
)
factors_q_monthly = pd.read_sql_query(
  sql = "SELECT * FROM factors_q_monthly", 
  con = tidy_finance,
  parse_dates = {"month"}
)

# To optimize database 
tidy_finance.execute("VACUUM")
