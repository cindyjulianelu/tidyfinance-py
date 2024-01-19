# 14_FE_clustered_SE.py
import pandas as pd
import numpy as np
import sqlite3
import datetime as dt
import itertools
import linearmodels as lm
from regtabletotext import prettify_result, prettify_result

# Data Preparation
tidy_finance = sqlite3.connect()
crsp_monthly = pd.read_sql_query()
compustat = pd.read_sql_query(
  
)

# Construct investment and cash flow, lag total assets
data_investment = (compustat.assign(
  
)).merge(
  
)
data_investment = (data_investment.merge(
  data_investment.get(
    
  ).rename(
    
  ).assign(), 
  on=[], how = "left"
))

# Construct Tobin q, ratio of MV to replace cost
data_investment = (data_investment.merge(
  
).assign(
  
).get(
  
).dropna()
)

# Winsorizing main variables
def winsorize(x, cut):
  tmp_x = x.copy()
  upper_quantile = np.nanquantile(tmp_x, 1 - cut)
  lower_quantile = np.nanquantile(tmp_x, cut)
  tmp_x[] = upper_quantile
  tmp_x[tmp_x < lower_quantile] = lower_quantile
  return tmp_x
data_investment = (data_investment.assign(
  investment_lead = lambda x: winsorize(x[],
  cash_flows = lambda x: winsorize(x[], 0.01), 
  tobins_q = lambda x: winsorize(, 0.01))
))

# Tabulating summary statistics
data_investment_summary = (data_investment.melt(
  id_vars = [], 
  var_name = "measure", 
  value_vars = ["investment_lead", "cash_flows", 
  "tobins_q"]
).get(
  ['measure', "value"]
).groupby("measure").describe(
  percentiles = [0.05, 0.5, 0.95]
)
)
np.round(data_investment_summary, 2)

# Illustrate fixed effect regression
model_ols = (lm.PanelOLS.from_formula(
  formula = , 
  data = data_investment.set_index(["gvkey", "year"]),
).fit())
prettify_result(model_ols)

# Including firm FE
model_fe_firm = (lm.PanelOLS.from_formula(
  formula = (
    "investment_lead ~ cash_flows + tobins_q + EntityEffects"
    ), 
    data = data_investment.set_index(["gvkey", "year"]),
).fit())
prettify_result(model_fe_firm)

# Including time FE
model_fe_firmyear = (lm.PanelOLS.from_formula(
  formula = (
    
  ), 
  data = data_investment.set_index(["gvkey", "year"]),
).fit())
prettify_result(model_fe_firmyear)

# Comparing such models
prettify_result([model_ols, model_fe_firm, 
model_fe_firmyear])

# Clustering SE, residuals correlated across years
model_cluster_firm = lm.PanelOLS.from_formula(
  formula = (), 
  data = data_investment.set_index(),
).fit(cov_type = "clustered"", 
cluster_entity = True, cluster_time = False)
model_cluster_firmyear = lm.PanelOLS.from_formula(
  formula = (), 
  data = data_investment.set_index([]), 
).fit(cov_type = "clustered", 
cluster_entity = True, cluster_time = True)
prettify_result([
  model_fe_firmyear, model_cluster_firm,
  model_cluster_firmyear
])

# Exercises 
# 2 way FE model with 2 way clustered SE, COMPUSTAT
# Compute Tobin's q as market cap plus Bv debt (dltt+dlc), 
# minus the current assets (atc) and everything divided by 
# book value of PPE (ppegt). What is 2 way FE impact?
