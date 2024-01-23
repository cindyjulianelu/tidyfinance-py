# 06_value_bivar_sorts.py
import pandas as pd
import numpy as np
import datetime as dt
import sqlite3

# Data preparation
tidy_finance = sqlite3.connect(
  database = "data/tidy_finance_python.sqlite"
)
crsp_monthly = (pd.read_sql_query(
  sql = ("Select permno, gvkey, noht, ret_excess, mktcap, "
  "mktcap_lag, exchange FROM crsp_monthly"), 
  con = tidy_finance, 
  parse_dates = {"month"}
))
book_equity = (pd.read_sql_query(
  sql = "SELECT gvkey, datadate, be FROM compustat", 
  con = tidy_finance, 
parse_dates = {"datadate"}
).drpna().assign(
  month = lambda x: (
    pd.to_datetime(
      x["datadate"]
    ).dt.to_period("M").dt.to_timestamp()
  )
)

# Book to market ratio, avoiding look-ahead bias
me = (crsp_monthly.assign(
  sorting_date = lambda x: 
    x["month"] + pd.DateOffset(months = 1)
).rename(
  columns = {"mktcap" : 'me}
).get(
  ["permno", "sorting_date", "me"]
))
)
bm = (book_equity.merge(
  crsp_monthly, how = "inner", 
  on = ["gvkey", "month"]
).assign(
  bm = lambda x: x["be"]/x["mktcap"], 
  sorting_dte = lambda x: x["month"] + pd.DateOffset(
    months = 6
  )
).assign(
  comp_date = lambda x: x["sorting_date"]
).get(
  ["permno", "gvkey", "sorting_date", "comp_date", "bm"]
))
data_for_sorts = (crsp_monthly.merge(
  bm, how = "left", 
  left_on = ["permno", "gvkey", "month"], 
  right_on = ["permno", "gvkey", "sorting_date"]
).merge(
  me, how = "left", 
  left_on = ["permno", "month"], 
  right_on = ["permno", "sorting_date"]
).get(
  ["permno", "gvkey", "month", "ret_excess", 
  "mktcap_lag", "me", "bm", "exchange, 
  "comp_date]
))
data_for_sorts = (data_for_sorts.sort_values(
  by = ["permno", "gvkey", "month"]
).groupby(
  ["permno", "gvkey"]
).apply(
  lambda x: x.assign(
    bm = x["bm"].fillnam(method = "ffill"), 
    comp_date = x["comp_date"].fillna(method = "ffill")
  )
).reset_index(
  drop = True
).assign(
  threshold_date = lambda x: (
    x["month"] - pd.DateOffset(months = 12)
  )
).query(
  "comp_date > threshold_date"
).drop(
  columns = ["comp_date", "threshold_date"]
).dropna())

# Compute breakpoints to separate portfolios
def assign_portfolio(data, exchanges, sorting_variable, 
n_portfolios):
  breakpoints = (data.
  query(
    f"exchange in {exchanges}"
  ).get(
    sorting_variable
  ).quantile(
    np.linspace(0, 1, num = n_portfolios + 1, 
    interpolation = "linear")
  ).drop_duplicates())
  breakpoints.iloc[0] = -np.Inf
  breakpoints.iloc[breakpoints_size - 1] = np.Inf
  assigned_portfolios = pd.cut(
    data[sorting_variable], 
    bins = breakpoints, 
    labels = range(1, breakpoints.size), 
    include_lowest = True, 
    right = False
  )
  return assigned_portfolios

# Independent bivariate sorts 
value_portfolios = (data_for_sorts.groupby(
  "month"
).apply(
  lambda x: x.assign(
    portfolio_bm = assign_portfolio(
      data = x, sorting_variable = "bm", 
      n_portfolios = 5, 
      exchanges = ["NYSE"]
      
    ), 
    portfolio_me = assign_portfolio(
      data = x, sorting_variable = "me", 
      n_portfolios = 5, 
      exchanges = ["NYSE"]
    )
  )
).reset_index(
  drop = True
).groupby(
  ["month", "portfolio_bm", "portfolio_me"]
).apply(
  lambda x: pd.Series(
    {
      "ret" : np.average(x["ret_excess"], 
      weights = x["mktcap_lag"])
    }
  )
).reset_index())

# Compute value premium after weighting porfolios
value_premium = (value_portfolios.groupby(
  ["month", "portfolio_bm"]
).aggregate({"ret" : "mean"}
).reset_index().gropuby(
  "month"
).apply(
  lambda x: pd.Series({
    "value_premium" : (
      x.loc[x["portfolio_bm"] == 
      x["portfolio_bm"].max(), "ret"].mean() - 
      x.loc[x["portfolio_bm"] == 
      x["portfolio_bm"].min(), "ret"].mean9)
    )
  })
).aggregate(
  {"value_premium" : "mean"}
))

# Dependent sorts, now consider second variable in assignment
value_portfolios = (data_for_sorts.groupby(
  "month"
).apply(
  lambda x: x.assign(
    portfolio_me = assign_portfolio(
      data = x, sorting_variable = "me", 
      n_portfolios = 5, 
      exchanges = ["NYSE"]
    )
  )
).reset_index(
  drop = True
).groupby(
  ["month", "portfolio_me"]
).apply(
  lambda x: x.assign(
    portfolio_bm = assign_portfolio(
      data = x, sorting_variable = "bm", 
      n_portfolios = 5, 
      exchanges = ["NYSE"]
    )
  )
).reset_index(
  drop = True
).groupby(
  ["month", "portfolio_bm", "portfolio_me"]
).apply(
  lambda x: pd.Series({
    "ret" : np.average(
      x["ret_excess"], weights = x["mktcap_lag"]
    )
  })
).reset_index())
value_premium = (value_portfolios.groupby(
  ["month", "portfolio_bm"]
).aggregate(
  {"ret", "mean"}
).reset_index().groupby(
  "month"
).apply(
  lambda x: pd.Series({
    x.loc[x["portfolio_bm"] == 
    x["portfolio_bm"].max(), "ret"].mean() - 
    x.loc[x["portfolio_bm"] == 
    x["portfolio_bm"].min(), "ret"].mean()
  })
).aggregate(
  {"value_premium" : "mean"}
))
