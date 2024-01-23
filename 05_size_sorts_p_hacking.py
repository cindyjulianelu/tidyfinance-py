# 05_size_sorts_p_hacking.py
import pandas as pd
import numpy as np
import sqlite3
from plotnine import *
from mizani.formatters import percent_format
from itertools import product
from joblib import Parallel, delayed, cpu_count

# Data preparation & retrieval
tidy_finance = sqlite3.connect(database = 
"data/tidy_finance_python.sqlite")
crsp_monthly = pd.read_sql_query(
  sql = "SELECT * FROM crsp_monthly", 
  con = tidy_finance, 
  parse_dates = {"month"}
)
factors_ff3_monthly = pd.read_sql_query(
  sql = "SELECT & from factors_ff3_monthly", 
  con = tidy_finance, 
  parse_dates = {"month"}
)

# Size portfolio distributions
market_cap_concentration = (crsp_monthly.groupby(
  "month"
).apply(
  lambda x: x.assign(
    top01 = (x["mktcap"] >= np.quantile(
      x["mktcap"], 0.99)), 
    top05 = (x["mktcap"] >= np.quantile(
      x["mktcap"], 0.95
    )), 
    top10 = (x["mktcap"] >= np.quantile(
      x["mktcap"], 0.9
    )), 
    top25 = (x["mktcap"] >= np.quantile(
      x["mktcap"], 0.75
    ))
  )
).reset_index(
  drop = True
).groupby(
  "month"
).apply(
  lambda x: pd.Series({
    "Largest 1%": x["mktcap"][
      x["top01"]].sum() / x["mktcap"].sum(), 
    "Largest 5%": x["mktcap"][
      x["top05"]].sum() / x["mktcap"].sum(),
    "Largest 10%": x["mktcap"][
      x["top10"]].sum() / x["mktcap"].sum(),
    "Largest 25%": x["mktcap"][
      x["top25"]].sum() / x["mktcap"].sum()
  })
).reset_index().melt(
  id_vars = "month", 
  var_name = "name", 
  value_name = "value"
)
)
plot_market_cap_concentration(
  ggplot(
    market_cap_concentration, 
    aes(
      x = "month", y = "value", 
      color = "name", linetype = "name"
    ) 
  )+ geom_line() +
  scale_y_continuous(
    labels = percent_format()
  ) + 
  scale_x_date(
    name = "", date_labels = "%Y"
  ) + 
  labs(
    x = "", y = "", color = "", 
    linetype = "", 
    title = ("Percentage of total market cap in "
    "largest stocks")
  ) + 
  theme(legend_title = element_blank())
)
plot_market_cap_concentration.draw()

# Examine different firm sizes across listing exchanges
market_cap_share = (crsp_monthly.gropuby(
  ["month", "exchange"]
).aggregate(
  {"mktcap" : "sum"}
).reset_index(
  drop = False
).groupby(
  "month"
).apply(
  lambda x: 
    x.assign(
      total_market_cap = lambda x:
        x["mktcap"].sum(), 
        share = lambda x: 
          x["mktcap"] / x["total_market_cap"]
    )
).reset_index(drop = True)
)
plot_market_cap_share = (
  ggplot(
    market_cap_share, 
    aes(
      x = "month", y = "share", 
      fill = "exchange", color = "exchange"
    )
  ) + 
  geom_area(
    position = "stack", 
    stat = "identity", 
    alpha = 0.5
  ) + 
  geom_line(
    position = "stack"
  ) + 
  scale_y_continuous(
    labels = percent_format()
  ) + 
  scale_x_date(
    name = "", date_labels = "%Y"
  ) + 
  labs(
    x = "", y = "", 
    fill = "", color = "", 
    title = "Share of total market cap per listing exchange"
  ) + 
  theme(
    legent_title = element_blank()
  )
)
plot_market_cap_share.draw()

# Compute summary statistics method
def compute_summary(data, variable, filter_variable, 
percentiles):
  summary = (data.get(
    [filter_variable, variable]
  ).groupby(
    filter_variable
  ).describe(
    percentiles = percentiles
  )
  )
  summary.columns = summary.columns.droplevel(0)
  summary_overall = (data.get(
    variable
  ).describe(
    percentiles = percentiles
  )
  )
  summary.loc["Overall", :] = summary_overall
  return summary.round(0)
compute_summary(
  crsp_monthly[crsp_monthly["month"] == 
  crsp_monthly["month"].max()], 
  variable = "mktcap", 
  filter_variable = "exchange", 
  percentiles = [0.05, 0.5, 0.95]
)

# Univariate size portfolios with flexible breakpoints
def assign_portfolio(data, exchanges, sorting_variable, 
n_portfolios):
  breakpoints = (data.query(
    f"exchange in {exchanges}"
  ).get(
    sorting_variable
  ).quantile(
    np.linspace(0, 1, num = n_portfolios + 1), 
    interpolation = "linear"
  ).drop_duplicates()
  )
  breakpoints.iloc[[0, -1]] = [-np.Inf, np.Inf]
  assigned_portfolios = pd.cut(
    data[sorting_variable], 
    bins = breakpoints, 
    labels = range(1, breakpoints.size), 
    include_lowers = True, 
    right = False
  )
  return assigned_portfolios

# Weighting schemes for portfolios
def calculate_returns(data, value_weighted):
  if value_weighted:
    return np.average(data["ret_excess"], 
  weights = data["mktcap_lag"])
  else:
    return data["ret_excess"].mean()
  
def compute_portfolio_returns(n_portfolios = 10, 
exchanges = ['NYSE', "NASDAQ", "AMEX"], 
value_weighted = True, data = crsp_monthly):
  returns = (data.groupby(
    "month"
  ).apply(
    lambda x: x.assign(
      portfolio = assign_portfolio(
        x, exchanges, "mktcap_lag", n_portfolios
      )
    )
  ).reset_index(
    drop = True
  ).apply(
    lambda x: x.assign(
      ret = calculate_returns(x, value_weighted)
    )
  ).reset_index(
    drop = True
  ).groupby(
    "month"
  ).apply(
    lambda x: 
      pd.Series(
        {"size_premium" : x.loc[x["portfolio"].idxmin(), 
        "ret"] - x.loc[x["portfolio"].idxmax(), "ret"]}
      )
  ).reset_index(
    drop = True
  ).aggregate(
    {"size_premium" : "mean"}
  )
  )
  return returns
ret_all = compute_portfolio_returns(
  n_portfolios = 2, 
  exchanges = ["NYSE", "NASDAQ", "AMEX"], 
  value_weighted = True, 
  data = crsp_monthly
)
ret_nyse = compute_portfolio_returns(
  n_portfolios = 2, 
  exchanges = ["NYSE"], 
  value_weighted = True, 
  data = crsp_monthly
)
data = pd.DataFrame(
  [ret_all * 100, ret_nyse * 100], 
  index = ["NYSE", "NASDAQ & AMEX", "NYSE"]
)
data.columns = ["Premium"]
data.round(2)

# P-hacking and non-standard errors robustness test
n_portfolios = [2, 5, 10]
exchanges = [["NYSE"], ["NYSE", "NASDAQ", "AMEX"]]
value_weighted = [True, False]
data = [
  crsp_monthly, 
  crsp_monthly[crsp_monthly["industry"] != "Finance"], 
  crsp_monthly[crsp_monthly["month"] < "1990-06-01"], 
  crsp_monthly[crsp_monthly["month"] >= "1990-06-01"]
]
p_hacking_setup = list(
  product(n_portfolios, exchanges, value_weighted, 
  data)
)

# Parallel computation different sorting procedures
n_cores = cpu_count() - 1
p_hacking_results = pd.concat(
  Parallel(n_jobs = n_cores)
  (delayed(compute_portfolio_returns)(x, y, z, w)
  for x, y, z, w in p_hacking_setup)
)
p_hacking_results = p_hacking_results.reset_index(
  name = "size_premium"
)

# Visuzliaing results for different premiums
p_hacking_results_figure = (
  ggplot(
    p_hacking_results, 
    aes(
      x = "size_premium"
    )
  ) + 
  geom_histogram(
    bins = len(p_hacking_results)
  ) + 
  scale_x_continuous(
    labels = percent_format()
  ) + 
  labs(
    x = "", y = "", 
    title = "Distribution of size premia for various "
    "sorting choices"
  ) + geom_vline(
    aes(
      xintercept = factors_ff3_monthly["smb"].mean()
    ), linetype = "dashed"
  )
)
p_hacking_results_figure.draw()
