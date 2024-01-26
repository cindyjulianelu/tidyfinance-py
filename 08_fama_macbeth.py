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
).sort_values(
  ["month", "permno"]
).groupby(
  "permno"
).apply(
  lambda x: x.assign(
    beta = x["beta"].fillna(method == "ffill"), 
    bm = x["bm"].fillna(method == "ffill"), 
    log_mktcap = x["log_mktcap"].fillna(
      method == "ffill"
    )
  ).reset_index(
    drop = True
  )
)
data_fama_macbeth_lagged = (data_fama_macbeth.assign(
  month = lambda x: x["month"] - pd.DateOffset(months = 1)
).get(
  ["permno", "month", "ret_excess"]
).rename(
  columns = {"ret_excess" : "ret_excess_lead"}
))
data_fama_macbeth = (data_fama_macbeth.merge(
  data_fama_macbeth_lagged, 
  how = "left", 
  on = ["permno", "month"]
).get(
  ["permno", "month", "ret_excess_lead", 
  "beta", "log_mktcap", "bm"]
).dropna())

# Cross section regressions
risk_premiums = (data_fama_macbeth.groupby(
  "month"
).apply(
  lambda x: smf.ols(
    formula = "ret_excess_lead ~ beta + log_mktcap + bm",
    data = x
  ).fit().params
).reset_index()
)

# Time series aggregation
price_of_rise = (risk_premiums.melt(
  id_vars = "month", 
  var_name = "factor", 
  value_name = "estimate"
).groupby("factor")["estimate"].apply(
  lambda x: pd.Series({
    "risk_premium" : 100 * x.mean(), 
    "t_statistics" : x.mean() / x.std() * np.sqrt(len(x))
  })
).reset_index().pivot(
  index = "factor", 
  columns = "level_1", 
  value = "estimate"
).reset_index())

# Adjust for autocorrelation
price_of_risk_newey_west = (risk_premiums.melt(
  id_vars = "month", 
  var_name = "factor", 
  value_name = "estimate"
).groupby(
  "factor"
).apply(
  lambda x: (
    x["estimate"].mean() / 
    smf.ols("estimate ~ 1", x)
  ).fit(
    cov_type = "HAC", 
    cov_kwds = {"maxlags" : 6}
  ).bse
)).reset_index().rename(
  columns = {"Intercept" : "t_statistic_new_west"}
)
)
(price_of_risk.merge(
  price_of_risk_newey_west, on = "factor"
).round(3))
