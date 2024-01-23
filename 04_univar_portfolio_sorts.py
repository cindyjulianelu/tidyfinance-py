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
  breakpoints = np.quantile(
    data[sorting_variable].dropna(), 
    np.linspace(0, 1, n_portfolios + 1),
    method = "linear"
  )
  assigned_portfolios = pd.cut(
    data[sorting_variable], 
    bins = breakpoints, 
    labels = range(1, breakpoints.size), 
    include_lowest = True, 
    right = False
  )
  return assigned_portfolios

# Use top function to sort stocks into 10 portfolios each month
beta_portfolios = (data_for_sorts.groupby(
  "month"
).apply(
  lambda x: x.assign(
    portfolio = assign_portfolio(x, "beta_lag", 10)
  )
).reset_index().groupby(
  ["portfolio", "month"]
).apply(
  lambda x: x.assign(
    ret = np.average(
      x["ret_excess"], weights = x["mktcap_lag"]
    )
  )
).reset_index().merge(
  factors_ff3_monthly, how = "left", 
  on = "month"
))

# More performance evaluation, CAPM-adjusted alphas
beta_portfolios_summary = (beta_portfolios.groupby(
  "portfolio"
).apply(
  lambda x: x.assign(
    alpha = sm.OLS.from_formula(
      formula = "ret ~ 1 + mkt_excess", 
      data = x
    ).fit().params[0], 
    beta = sm.OLS.from_formula(
      formula = "ret ~ 1 + mkt_excess", 
      data = x
    ).fit().params[1],
    ret = x["ret"].mean()
).tail(1)
).reset_index(
  drop = True
).get(
  ["portfolio", "alpha", "beta", "ret"]
)
)

# Illustrate CAPM alphas of beta sorted portfolios
plot_beta_portfolios_summary(
  ggplot(
    beta_portfolios_summary, 
    aes(
      x = "portfolio", 
      y = "alpha", 
      fill = "portfolio"
    )
  ) + 
  geom_bar(
    stat = "identity"
  ) + 
  labs(
    x = "Portfolio", 
    y = "CAPM alpha", fill = "Portfolio", 
    title = "CAPM alphas of beta-sorted portfolios"
  )
).scale_y_continuous(
  labels = percent_format()
).theme(
  legend_position = "none"
)
plot_beta_portfolios_summary.draw()

# The security market line and beta portfolios
sml_capm = (sm.OLS.from_formula(
  formula = "ret ~ 1 + beta", 
  data - beta_portfolios_summary
).fit().params)
plot_sml_capm = (
  ggplot(
    beta_portfolios_summary, 
    aes(
      x = "beta", y = "ret", 
      color = "portfolio"
    )
  ) + 
  geom_point() + 
  geom_abline(
    intercept = 0, 
    slope = factors_ff3_monthly["mkt_excess"].mean(), 
    linetype = "solid"
  ) + 
  geom_abline(
    intercept = sml_capm["Intercept"], 
    slope = sml_capm["beta"], 
    linetype = "dashed"
  ) + 
  labs(
    x = "Beta", y = "Excess Return", 
    color = "Portfolio", 
    title = "Average portfolio excess returns and beta estimates"
  ) + 
  scale_x_continuous(
    limits = (0, 2)
  ) + 
  scale_y_continuous(
    labels = percent_format(), 
    limits = (0, factors_ff3_monthly["mkt_excess"].mean())
  )
) 
plot_sml_capm.draw()

# Prove more evidence against CAPM predictions
beta_longshort = (beta_portfolios.assign(
  portfolio = lambda x: (
    x["portfolio"].apply(
      lambda y: high if y == x["portfolio"].max()
      else ("low" if y == x["portfolio"].min()
      else y)
  )
  )
).query(
  "portfolio in ['low', 'high']"
).pivot_table(
  index = "month", 
  columns = "portfolio", 
  values = "ret"
).assign(
  long_short = lambda x: x["high"] - x["low"]
).merge(
  factors_ff3_monthly, 
  how = "left", on = "month"
)
)

# There's no statistically significant returns
model_fit = (sm.OLS.from_formula(
  formula = "long_short ~ 1",
  data = beta_longshort
).fit(cov_type = "HAC", 
cov_kwds = {"maxlags" : 1}))
prettify_result(model_fit)

# Annual returns of extreme beta portfolios
beta_longshort_year = (beta_longshort.assign(
  year = lambda x: x["month"].dt.year
).groupby(
  "year"
).aggregate(
  
).reset_index().melt(
  id-vars = "year", var_name = "name", 
  value_name = "value"
))
plot_beta_longshort_year = (
  ggplot(
    beta_longshort_year, 
    aes(x = "year", 
    y = "value", fill = "name")
  ) + 
  geom_col(
    position = "dodge"
  ) + 
  facet_wrap(
    "~name", ncol = 1
  ) + 
  labs(
    x = "", y = "", 
    title = "Annual returns of beta portfolios"
  ) + 
  scale_color_discrete(
    guide = False
  ) + 
  scale_y_continuous(
    labels = percent_format()
  ) + 
  theme(
    legend_position = "none"
  )
)
plot_beta_longshort_year.draw()
