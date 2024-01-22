# 03_beta_estimation.py
import pandas as pd
import numpy as np
import sqlite3
import statsmodels.formula.api as smf
from regtabletotext import prettify_result
from statsmodels.regression.rolling import RollingOLS
from plotnine import *
from mizani.breaks import date_breaks
from mizani.formatters import percent_format, date_format
from joblib import Parallel, delayed, cpu_count
from itertools import product

# Estimate beta from monthly return
tidy_finance = sqlite3.connect(
  database = "data/tidy_finance_python.sqlite"
)
crsp_monthly = (pd.read_sql_query(
  sql = "SELECT permno, month, industry, ret_excess "
  "FROM crsp_monthly", 
  con = tidy_finance, 
  parse_dates = {"month"},
).dropna())
factors_ff3_monthly = pd.read_sql_query(
  sql = "SELECT month, mkt_excess " 
  "FROM factors_ff3_monthly",
  con = tidy_finance, 
  parse_dates = {"month"}
)
crsp_monthly = (crsp_monthly.merge(
  factors_ff3_monthly, how = "left",
  on = "month"
))

# Regress stock excess return on market portfolio excess return
model_beta = (smf.ols(
  formula = "ret_excess ~ mkt_excess", 
  data = crsp_monthly.query("permno == 14593")
).fit())
prettify_result(model_beta)

# Rolling Window Estimation
window_size = 60
min_obs = 48
valid_permnos = (crsp_monthly.dropna().groupby(
  "permno"
)["permno"].count().reset_index(
  name = "counts"
).query(f"counts > {window_size}+1"))

# Need to change implicit missing rows to explicit
permno_information = (crsp_monthly.merge(
  valid_permnos, how = "inner", on = "permno"
).groupby(["permno"]
).aggregate(
  first_month = ("month", "min"), 
  last_month = ("month", "max")
).reset_index()
)

# Complete the missing observations in CRSP sample
unique_permno = crsp_monthly["permno"].unique()
unique_month = factors_ff3_monthly["month"].unique()
all_combinations = pd.DataFrame(
  product(unique_permno, unique_month), 
  columns = ["permno", "month"]
)

# Expand CRSP sample, include row with missing excess return
returns_monthly = (all_combinations.merge(
  crsp_monthly.get(["permno", "month", 
  "ret_excess"]), 
  how = "left", 
  on = ["permno", "month"]
).merge(
  permno_information, how = "left", 
  on = "permno"
).query(
  "(month >= first_month) & (month <= last_month)"
).drop(columns = ["first_month", "last_month"]
).merge(
  crsp_monthly.get(["permno", "month", "industry"]), 
  how = "left", 
  on = ["permno", "month"]
).merge(
  factors_ff3_monthly, 
  how = "left", on = "month"
)
)

# CAPM regression for data containing minimum observations
def roll_capm_estimation(data, window_size, min_obs):
  data = data.sort_values("month")
  result = (RollingOLS.from_formula(
    formula = "ret_excess ~ mkt_excess", 
    data = data,
    window = window_size,
    min_nobs = min_obs, 
    missing = "drop"
  ).fit().params.get("mkt_excess")
  )
  result.index = data.index
  return result

# Test cases before running whole CRSP sample
examples = pd.DataFrame({
  "permno" : [14593, 10107, 93436, 17778],
  "company" : ["Apple", "Microsoft", "Tesla", 
  "Berkshire Hathaway"]
})

# Perform roll window estimation and visualize
beta_example = (returns_monthly.merge(
  examples, how = "inner", on = "permno"
).groupby(["permno"]
).apply(
  lambda x: x.assign(
    beta = roll_capm_estimation(x, window_size, min_obs)
  )
).reset_index(drop = True).dropna()
)
plot_beta = (
  ggplot(beta_example, 
  aes(x = "month", 
  y = "beta", color = "company", 
  linetype = "company")) + 
  geom_line() + 
  labs(x = "", y = "", color = "", linetype = "", 
  title = "Monthly beta estimates for example stocks") + 
  scale_x_datetime(breaks = date_breaks("5 year"), 
  labels = date_format("%Y"))
)
)
plot_beta.draw()

# Estimate beta using all monthly returns
def roll_capm_estimation_for_joblib(permno, group):
  if "date" in group.columns:
    group = group.sort_values(by = "date")
  else:
    group = group.sort_values(by = "month")
  beta_values = (RollingOLS.from_formula(
    formula = "ret_excess ~ mkt_excess", 
    data = group, 
    window = window_size, 
    min_nobs = min_obs, 
    missing = "drop"
  ).fit().params.get(
    "mkt_excess"
  ))
  result = pd.DataFrame(beta_values)
  result.columns = ["beta"]
  result["month"] = group["month"].values
  result["permno"] = permno
  try:
    result["date"] = group["date"].values
    result = result[
      (result.groupby("month")["date"].transform(
        "max"
      )) == result["date"]
    ]
  except(KeyError):
    pass
  return result
permno_groups = (returns_monthly.merge(
  valid_permnos, how = "inner", 
  on = "permno"
).groupby("permno", group_keys = False))
n_cores = cpu_count() - 1
beta_monthly = (
  pd.concat(
    Parallel(n_jobs = n_cores)
    (delayed(roll_capm_estimation_for_joblib)(name, group)
    for name, group in permno_groups)
  ).dropna().rename(
    columns = {"beta": "beta_monthly"}
  )
)

# Estimating beta using daily returns
factors_ff_daily = pd.read_sql_query(
  sql = "SELECT date, mkt_excess FROM factors_ff3_daily",
  con = tidy_finance, 
  parse_dates = {"date"}
)
unique_Date = factors_ff3_daily["date"].unique()

# Consider 3 months of data as window
window_size = 60
min_obs = 50
permnos = list(
  crsp_monthly["permno"].unique.astype(str))
batch_size = 500
batches = np.ceil(
  len(permnos) / batch_size
).astype(int)

# Same steps as monthly CRSP data
beta_daily = []
for j in range(1, batches + 1):
  permno_batch = permnos[
    ((j-1) * batch_size) : (
      min(j * batch_size, len(permnos)))
  ]
  permno_batch_formatted = (
    ", ".join(
      f"'{permno}'" for permno in permno_batch
    )
  )
  permno_string = f"({permno_batch_formatted})"
  crsp_daily_sub_query(
    "SELECT permno, month, date, ret_excess "
    "FROM crsp_daily "
    f"WHERE permno IN {permno_string}"
  )
  crsp_daily_sub = pd.read_sql_query(
    sql = crsp_daily_sub_query, 
    con = tidy_finance, 
    dtype = {"permno" : int},
    parse_dates = {"date", "month"}
  )
  valid_permnos = (crsp_daily_sub.groupby(
    "permno"
  )["permno"].count().reset_index(
    name = "counts"
  ).query(
    r"counts > {window_size} + 1"
  ).drop(
    columns = "counts"
  ))
  permno_information = (crsp_daily_sub.merge(
    valid_permnos, 
    how = "inner", on = "permno"
  ).groupby(
    ["permno"]
  ).aggregate(
    first_date = ("date", "min"), 
    last_date = ("date", "max")
  ).reset_index()
  )
  unique_permno = permno_information["permno"].unique()
  all_combinations = pd.DataFrame(
    product(unique_permno, unique_date), 
    colunms = ["permno", "date"]
  )
  returns_daily = (crsp_daily_sub.merge(
    all_combinations, how = "right", 
    on = ["permno", "date"]
  ).merge(
    permno_information, how = "left", 
    on = "permno"
  ).query(
    "(date >= first_date) & (date <= last_date)"
  ).drop(
    columns = ["first_date", "last_date"]
  ).merge(
    factors_ff3_daily, how = "left", on = "date"
  ))
  permno_groups = (returns_daily.groupby(
    "permno", group_keys = False
  ))
  beta_daily_sub = (
    pd.concat(
      Parallel(n_jobs = n_cores)
      (delayed(roll_capm_estimation_for_joblib)(name, group)
      for name, group in permno_groups)
    ).dropna().rename(
      columns = {"beta" : "beta_daily"}
    )
  )
  beta_daily.append(beta_daily_sub)
  print(
    f"Batch {j} out of {batches} done ({(j / batches) * 100:.2f}%)\n")
beta_daily = pd.concat(beta_daily)

# Comparing beta estimates
beta_industries = (beta_monthly.merge(
  crsp_monthly, how= "inner", 
  on = ["permno", "month"]
).dropna(
  subset = "beta_monthly"
).groupby(
  ["industry", "permno"])["beta_monthly"]
).aggregate(
  "mean"
).reset_index()
)
industry_order = (beta_industries.groupby(
  "industry"
)["beta_monthly"].aggregate(
  "median"
).sort_values().index.tolist())
plot_beta_industries = (
  ggplot(beta_industries, aes(
    x = "industry", y = "beta_monthly"
  )) + 
  geom_boxplot() + 
  coord_flip() + 
  labs(x = "", y = "", 
  title = "Firm-specific beta distributions by industry"
  ) + 
  scale_x_discrete(
    limits = industry_order
  )
)
plot_beta_industries.draw()

# Time variation in cross section of estimated betas
beta_quantiles = (beta_monthly.groupby(
  "month"
)["beta_monthly"].quantile(
  q = np.arange(0.1, 1.0, 0.1)
).reset_index().rename(
  columns = {"level_1" : "quantiles"}
).assign(
  quantile = lambda x: (
    x['quantile'] * 100).astype(int)
).dropna())
linetypes = ["-", "--", "-.", ":"]
n_quantiles = beta_quantiles[
  "quantile"
].nunique()
plot_beta_quantiles = (
  ggplot(
    beta_quantiles, aes(
      x = "month", y = "beta_monthly", 
      color = "factor(quantile)", 
      linetype = "factor(quantile)"
    )
  ) + 
  geom_line() + 
  labs(
    x = "", y = "", color = "", linetype = "", 
    title = "Monthly deciles of estimated betas"
  ) + 
  scale_x_datetime(
    breaks = date_breaks("5 year"), 
    labels = date_format("%Y")
  ) + 
  scale_linetype_manual(
    values = [linetypes[1 % len(linetypes)] for l in 
    range(n_quantiles)]
  )
)
plot_beta_quantiles.draw()

# Comparing daily and monthly data, combine both data
beta = (beta_monthly.get(
  ["permno", "month", "beta_monthly"]
).merge(
  beta_daily.get(["permno", "month", "beta_daily"]), 
  how = "outer", 
  on = ["permno", "month"]
))
beta_comparison = (beta.merge(
  examples, on = "permno"
).melt(
  id_vars = ["permno", "month", "company"], 
  var_name = "name", 
  value_vars = ["beta_monthly", "beta_daily"], 
  value_name = "value"
).dropna()
)
plot_beta_comparison = (
  ggplot(
    beta_comparison, 
    aes(
      x = "month", y = "value", 
      color = "name"
    )
  ) + 
  geom_line() + 
  facet_wrap("~company", ncol = 1) +
  labs(
    x = "", y = "", color = "",
    title = "Comparing beta estimates, monthly vs. daily"
  ) + 
  scale_x_datetime(breaks = date_breaks("10 years"), 
  labels = date_format("%Y")) + 
  theme(figure_size = (6.4, 6.4))
)
plot_beta_comparison.draw()

# Write the estimates to database in future chapters
(beta.to_dql(
  name = "beta", 
  con = tidy_finance, 
  if_exists = "replace", 
  index = False
))

# Plausibility tests, share of stocks with estimates 
beta_long = (crsp_monthly.merge(
  beta, how = "left", 
  on = ["permno", "month"]
).melt(
  id_vars = ["permno", "month"], 
  var_name= "name", 
  value_vars = ["beta_monthly", "beta_daily"], 
  value_name = "value"
))
beta_shares = (beta_long.groupby(
  ["month", "name"]
).aggregate(
 share = ("value", 
 lambda x: sum(~x.isna())/ len(x)) 
).reset_index()
)
plot_beta_long = (
  ggplot(beta_shares, 
  aes(
    x = "month", y = "share", 
    color = "name", 
    linetype = "name"
  ) + 
  geom_line() + 
  labs(
    x = "", y = "", 
    color = "", linetype = "", 
    title = 
    "End of month share of securities with beta estimates"
  ) + 
  scale_y_continuous(
    labels = percent_format()
  ) + 
  scale_x_datetime(
    breaks = date_breaks("10 year""), 
    labels = date_format("%Y"))
  )
)
plot_beta_long.draw()

# Distributional summary statistics of variables
beta_long.groupby("name")['value'].describe.round(2)

# Positively correlated estimators
beta.get(
  ['beta_monthly', "beta_daily"]
).corr().round(2)
