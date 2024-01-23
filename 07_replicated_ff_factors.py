# 07_replicating_ff_factors.py
import pandas as pd
import numpy as np
import sqlite3
import statsmodels.formula.api as smf
from regtabletotext import prettify_Result

# Prepare and load data sources
tidy_finance = sqlite3.connect(
  database = "data/tidy_finance_python.sqlite"
)
crsp_monthly = (pd.read_sql_query(
 sql = ("SELECT permno, gvkey, month, ret_excess, mktcap, "
 "mktcap_lag, exchange FROM crsp_monthly"), 
 con = tidy_finance, 
 parse_dates = {"month"}
).dropna()
)
compustat = (pd.read_sql_query(
  sql = "SELECT gvkey, datadate, be, op, "
  "inv FROM compustat", 
  con = tidy_finance, 
  parse_dates = {"datadate"}
).dropna())
factors_ff3_monthly = pd.read_sql_query(
  sql = "SELECT month, smb, hml "
  "FROM factors_ff3_monthly", 
  con = tidy_finance, 
  parse_dates = {"month"}
)
factors_ff5_monthly = pd.read_sql_query(
  sql = "SELECT month, smb, hml, rmw, "
  "cma FROM factors_ff5_monthly", 
  con = tidy_finance, 
  parse_dates = {"month"}
)
size = (crsp_monthly.query(
  "month.dt.month == 6"
).assign(
  sorting_date = lambda x: (
    x["month"] + pd.DateOffset(months = 1)
  )
).get(
  ["permno", "exchange", "sorting_date", "mktcap"]
).rename(
  columns = {"mktcap" : "size"}
)
)
market_equity = (crsp_monthly.query(
  "month.dt.month == 12"
).assign(
  sorting_date = lambda x : (
    x["month"] + pd.DateOffset(months = 7)
  )
).get(
  ["permno", "gvkey", "sorting_date", "mktcap"]
).rename(
  columns = {"mktcap" : "me"}
))
book_to_market = (compustat.assign(
  sorting_date= lambda x: (pd.to_datetime(
    x["datadate"].dt.year + 1
  ).astype(str) + "0701", 
  format = "%Y%m%d")
).merge(
  market_equity, 
  how = "inner", 
  on = ["gvkey", "sorting_date"]
).assign(
  bm = lambda x: x["be"] / x["me"]
).get(
  ["permno", "sorting_date", "me", "bm"]
)
)
sorting_variables = (size.merge(
  book_to_market, 
  how = "inner", 
  on = ["permno", "sorting_date"]
).dropna().drop_duplicates(
  subset = ["permno", "sorting_date"]
))

# Portfolio construction with exchange-based breakpoints
def assign_portfolio(data, sorting_variable, percentiles):
  breakpoints = (data.query(
    "exchage == 'NYSE'"
  ).get(
    sorting_variable
  ).quantile(
    percentiles, interpolation = "linear"
  ))
  breakpoints.iloc[0] = -np.Inf
  breakpoints.iloc[breakpoints.size-1] = np.Inf
  assigned_portfolios = pd.cut(
    data[sorting_variable], 
    bins = breakpoints, 
    labels = pd.Series(range(1, breakpoints.size)), 
    include_lowest = True, 
    right = False
  )
  return assigned_portfolios
portfolios = (sorting_variables.groupby(
  "sorting_date"
).apply(
  lambda x: x.assign(
    portfolio_size = assign_portfolio(
      x, "size", [0, 0.5, 1]
    ), 
    portfolio_bm = assign_portfolio(
      x, "bm", [0, 0.3, 0.7, 1]
    )
  )
).reset_index(
  drop = True
).get(
  ["permno", "sorting_date", "portfolio_size", 
  "portfolio_bm"]
))

# Merge portfolios to return data for rest of the year
portfolios = (crsp_monthly.assign(
  sorting_date = lambda x: (
    pd_to_datetime(
      x["month"].apply(
        lambda x: str(x.year - 1) + "0701" if x.month <= 6
        else xtr(x.year) + "0701"
      )
    )
  )
).merge(
  portfolios, how = "inner", 
  on = ["permno", "sorting_date"]
))

# Fama French 3 factor model
factors_replicated = (portfolios.groupby(
  ["portfolio_size", "portfolio_bm", "month"]
).apply(
  lambda x: pd.Series(
    {"ret" : np.average(
      x["ret_excess"], weights = x["mktcap_lag"]
    )}
  )
).reset_index().groupby(
  "month"
).apply(
  lambda x: pd.Series({
    "smb_replicated" : (
      x["ret"][x["portfolio_size"] == 1].mean() - 
      x["ret"][x["portfolio_size"] == 2].mean()
    ), 
    "hml_replicated" : (
      x["ret"][x["portfolio_bm"] == 3].mean() - 
      x["ret"][x["portfolio_bm"] == 1].mean()
    )
  })
).reset_index()
)
factors_replicated = (factors_replicated.merge(
  factors_ff3_monthy, 
  how = "inner", 
  on = "month"
).round(4))

# Replicating evaluation of the size and value premium
model_smb = (smf.ols(
  formula = "smb ~ smb_replicated", 
  data = factors_replicated
).fit())
prettify_result(model_smb)
model_hml = (smf.ols(
  formula = "mhl ~ hml_replicated", 
  data = factors_replicated
).fit())
prettify_result(model_hml)

# Fama French five factor model
other_sorting_variables = (compustat.assign(
  sorting_date = lambda x: (pd.to_datetime(
    (
      x["datadate"].dt.year + 1
    ).astype(str) + "0701", format = "%Y%m%d"
  ))
).merge(
  market_equity, how = "inner", 
  on = ["gvkey", "sorting_date"]
).assign(
  bm = lambda x: x["be"] / x["me"]
).get(
  ["permno", "sorting_date", "me", "bm", "op", "inv"]
))
sorting_variables = (size.merge(
  other_sorting_variables, 
  how = "inner", 
  on = ["permno", "sorting_date"]
).dropna().drop_duplicates(
  subset = ["permno", "sorting_date"]
))

# Each month, sort all stocks into 2 size portolios
portfolios = (sorting_variables.groupby(
  "sorting_date"
).apply(
  lambda x: x.assign(
    portfolio_size = assign_portfolio(x, "size", 
    [0, 0.5, 1])
  )
).reset_index(
  drop = True
).groupby(
  ["sorting_date", "portfolio_size"]
).apply(
  lambda x: x.assign(
    portfolio_bm = assign_portfolio(
      x, "bm", [0, 0.3, 0.7, 1]
    ), 
    portfolio_op = assign_portfolio(
      x, "op", [0, 0.3, 0.7, 1]
    ), 
    portfolio_inv = assign_portfolio(
      x, "inv", [0, 0.3, 0.7, 1]
    )
  )
).reset_index(
    drop = True
).get(
  ["permno", "sorting_date", "portfolio_size", 
  "portfolio_bm", "portfolio_op", "portfolio_inv"]
)
)
portfolios = (crsp_monthly.assign(
  sorting_date = lambda x: (pd.to_datetime(
    x["month"].apply(
      lambda x: str(x.year-1) + "0701" if x.month <= 6
      else str(x.year) + "0701"
    )
  ))
).merge(
  portfolios, how = "inner", 
  on = ["permno", "sorting_date"]
))

# Construct all factors, saving size for the last 
portfolios_value = (portfolios.groupby(
  ["portfolio_size", "portfolio_bm", "month"]
).apply(
  lambda x: pd.Series(
    {
      "ret" : np.average(
        x["ret_excess"], weights = x["mktcap_lag"]
      )
    }
  )
).reset_index())
factors_value = (portfolios_value.groupby(
  "month"
).apply(
  lambda x: pd.Series({
    "hml_replicated" : (
      x["ret"][x["portfolio_bm"] == 3].mean() - 
      x["ret"][x["portfolio_bm"] == 1].mean()
    )
  })
).reset_index())
portfolios_profitability = (portfolios.groupby(
  ["portfolio_size", "portfolio_op", "month"]
).apply(
  lambda x: pd.Series({
    "ret" : np.average(x["ret_excess"], 
    weights = x["mktcap_lag"])
  })
).reset_index())
factors_profitability = (portfolios_profitability.groupby(
  "month"
).apply(
  lamba x: pd.Series({
    "rmw_replicated" : (
      x["ret"][x["portfolio_op"] == 3].mean() - 
      x["ret"][x["portfolio_op"] == 1].mean()
    )
  })
).reset_index())
portfolios_investment = (portfolios.groupby(
  ["portfolio_size", "portfolio_inv", "month"]
).apply(
  lambda x: pd.Series({
    "ret" : np.average(
      x["ret_excess"], weights = x["mktcap_lag"]
    )
  })
).reset_index())
factors_investment = (portfolios_investment.groupby(
  "month"
).apply(
  lambda x: pd.Series({
    "cma_replicated" : (
      x["ret"][x["portfolio_inv"] == 1].mean() - 
      x["ret"][x["portfolio_inv"] == 3].mean()
    )
  })
).reset_index())

# Size factor constructed long 6 small portfolios
factors_size = (
  pd.concat(
    [portfolios_Value, portfolios_profitability, 
    portfolios_investment], 
    ignore_index = True
  ).groupby(
    "month"
  ).apply(
    lambda x: pd.Series({
      "smb_replicated" : (
        x["ret"][x["portfolio_size"] == 1].mean() - 
        x["ret"][x["portfolio_size"] == 2].mean()
      )
    })
  ).reset_index()
)

# Join all factors into one data frame
factors_replicated = (factors_size.merge(
  factors_value, how = "outer", on = "month"
).merge(
  factors_profibatility, 
  how = "outer", on = "month"
).merge(
  factors_investment, how = "outer", on = "month"
)
)
factors_replicated = (factors_replicated.merge(
  factors_ff5_monthly, 
  how = "inner", on = "month"
).round(4))

# Replication evaluation again
model_smb = (smf.ols(
  formula = "smb ~ smb_replicated", 
  data = factors_replicated
).fit())
prettify_result(model_smb)
model_hml = (smf.ols(
  formula = "hml ~ fml_replicated", 
  data = factors_replicated
).fit())
prettify_result(model_hml)
model_rmw = (smf.ols(
  formula = "rmw ~ rmw_replicated", 
  data = factors_replicated
).fit())
prettify_result(model_rmw)
model_cma (smf.ols(
  formula = "cma ~ cma_replicated", 
  data = factors_replicated
).fit())
prettify_result(model_cma)
