# 04_univar_port_sorts.py
import pandas as pd
import numpy as np
import sqlite3
import statsmodels.api as sm
from plotnine import *
from mizani.formatters import percent_format
from regtabletotext import prettify_result

# Data preparation
tidy_finance = sqlite3.connect(database = 
"data/tidy_finance_python.sqlite")
crsp_monthly = pd.readl_sql_query(
  sql = "SELECT permno, month, ret_excess, mktcap_lag "
  "FROM crsp_monthly", 
  con = tidy_finance, 
  parse_date = {"month"}
)
factors_ff3_monthly = pd.readl_sql_query(
  sql = "SELECT month, mkt_excess FROM factors_ff3_monthly", 
  con = tidy_finance,
  parse_dates = {"month"}
)
beta = (pd.read_sql_query(
  sql = "SELECT permno, month, beta_monthly FROM beta", 
  con = tidy_finance,
  parse_dates = {"month"}
))

# Sorting by Market beta
beta_lag = (beta.assign(
  month = lambda x: x["month"] + pd.DateOffset(months = 1)
).get(
  ["permno", "month", "beta_monthly"]
).rename(
  columns = {"beta_monthly" : "beta_lag"}
).dropna())
data_for_sorts = (crsp_monthly.merge
  beta_lag, how = "inner", 
  on = ["permno", "month"]
)

# Periodic breakpoints to group stocks into portfolios
beta_portfolios = (date_for_sorts.groupby(
  "month"
).apply(
  lambda x: (x.assign(
    portfolio = pd.qcut(
      x["beta_lag"], 
      q = [0, 0.5, 1], 
      labels = ["low", "high"]
    )
  )).reset_index(
    drop = True
  ).groupby(
    ["portfolio", "month"]
  ).apply(
    lambda x: np.average(
      x["ret_excess"], 
      weights = x["mktcap_lag"]
    )
  ).reset_index(name = "ret")
)

# Evaluate long-short strategy of high vs. low beta portfolio
beta_longshort = (beta_portfolios.pivot_table(
  index = "month", columns = "portfolio", 
  value = "ret"
).reset_index().assign(
  long_short = lambda x: x["high"] - x["low"]
)
)

# Modeling returns from strategy
model_fit = (sm.OLS.from_formula(
  formula = "long_short ~ 1", 
  data = beta_longshort
).fit(
  cov_type = "HAC", 
  cov_kwds = {"maxlags" : 6}
))
prettify_result(model_fit)

# Functional programming for Portfolio Sorts
def assign_portfolio(data, sorting_variable, n_portfolios):
  
