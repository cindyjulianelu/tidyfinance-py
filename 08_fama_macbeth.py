# 08_fama_macbeth.py
import pandas as pd
import numpy as np
import sqlite3 
import statsmodels.formula.api as smf

# Data preparation
tidy_finance = sqlite3.connect(
  database = "data/tidy_finance_python.sqlite"
)
crsp_monthly = pd.read_sql_query(
  sql = ("SELECT permno, gvkey, month, ret_excess, "
  "mktcap FROM crsp_monthly"), 
  con = tidy_finance, 
  parse_dates = {"month"}
)
compustat = pd.read_sql_query(
  sql = "SELECT datadate, gvkey, be from compustat", 
  con = tidy_finance, 
  parse_dates = {"datadate"}
)
beta = pd.read_sql_query(
  sql = "SELECT month, permno, beta_monthly FROM beta", 
  con = tidy_finance, 
  parse_dates = {"month"}
)
characteristics = (compustat.assign(
  month = lambda x: 
    x["datadate"].dt.to_period("M").dt.to_timestamp()
).merge(
  crsp_monthly, how = "left", 
  on = ["gvkey", "month"]
).merge(
  beta, how = "left", 
  on = ["permno", "month"]
).assign(
  bm = lambda x: x["be"] / x["mktcap"], 
  log_mktcap = lambda x: np.log(x["mktcap"]), 
  sorting_date = lambda x: x["month"] + pd.DateOffset(months = 6)
).get(
  ["gvkey", "bm", "log_mktcap", "beta_monthly", 
  "sorting_date"]
).rename(
  columns = {"beta_monthly" : "beta"}
))
data_fama_macbeth = (crsp_monthly.merge(
  characteristics, 
  how = "left", 
  left_on = ["gvkey", "month"], 
  right_on = ["gvkey", "sorting_date"]
)
)
