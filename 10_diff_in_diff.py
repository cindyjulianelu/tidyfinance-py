# _diff_in_diff
import pandas as pd
import numpy as np
import sqlite3
import linearmodels as lm
import statsmodels.formula.api as smf
from plotnine import *
from scipy.stats import norm
from mizani.breaks import date_breaks
from mizani.formatters import date_format
from regtabletotext import prettify_result

# Data preparation
tidy_finance = sqlite3.connect(database = 
"")
fisd = (pd.read_sql_query(
  sql = , 
  con = tidy_finance, 
  parse_dates = {"maturity"}).dropna()
)
trace_enhanced = (pd.read_sql_query(
  sql= ,
  con = tidy_finance, 
  parse_dates = {"trd_exctn_dt"}
).dropna()
)

# Further prepare bonds datasets
treatment_date = pd.to_datetime("2015-12-12")
polluting_industries = [
  
]
bonds = (fisd.query(
  
).assign(
  
).query(
  "time_to_maturity >= 1"
).rename(columns = {}
).get(
  []
).assign(polluter = lambda x: x["sic_code"].isin()
).reset_index(drop = True)
)

# Aggregating individual transaction in bond yields
trace_enhanced = (trace_enhanced.query(
  
).assign(
  
).assign(
  
)
)
trace_aggregated = (trace_enhanced.groupby(
  []
  ).aggregate(
    weighted_yield_sum = (), 
    weight_sum = ("weight", "sum"), 
    trades = ("rptd_pr", "count")
  ).reset_index().assign(
    
  ).dropna(subset = ["avg_yield"]
  ).query("trades >= 5"
  ).assign(
    
  ).assign(
    
  )
)
date_index = (trade_aggregated.groupby(
  
).idmax()
)
trace_aggreagted = (
  trace_aggregated.loc[date_index].get([
    "cusip_id", "month", "avg_yield"])
)
bonds_panel = (bonds.merge(
  trace_aggregated, how = "inner", 
  on = "cusip_id"
).dropna())

# Create the treated indicator variable
bonds_panel = (bonds_panel.assign(
  post_period = lambda x: (x["month"] >= (
    
  )).assign(
    treated = lambda x: x["polluter"] & x["post_period"]
  ).assign(
   month_cat = lambda x: pd.Categorical(
     x["month"], ordered = True
   )
  )
))

# Tabulate summary statistics
bonds_panel_summary = bonds_panel.melt(
  var_name = "measure", 
  value_vars = ["avg_yield", "time_to_maturity", 
  "log_offering_amount"]
).groupby("measure").describe(
  percentiles = [.05, .5, .95]
)
np.round(bonds_panel_summary, 2)

# Panel Regressions

# Visualizing Parallel trends

# FE model and focus on polluter response to signing
bonds_panel_alt = (bonds_panel
)
variables = (bonds_panel_alt.get(
  
).reset_index(drop = True)
)
formula = "avg_yield ~ 1+ "
for j in range(variables.shape[0]):
  if variables["diff_to_treatment"].iloc[j] != 0:
    old_names = list()
    bonds_panel_alt["new_var"] = (
      
    ) & bonds_panel_alt["polluter"]
    
    diff_to_treatment_value = variables[
      "diff_to_treatment"].iloc[j]
    direction = "lag" if 
    diff_to_treatment_value <0 else "lead"
    abs_diff_to_treatment = int()
    new_var_name = f""
    variables.at[j, "variable_name"] = new_var_name
    bonds_panel_alt[] = bonds_panel_alt[]
    formula += (f"")
formula = formula+" + EntityEffects + TimeEffects"
model_with_fe_time = (lm.PanelOLS.from_formula(
  formula = formula, 
  data = bonds_panel_alt.set_index(
    ["cusip_id", "month"]
  ).fit().summary)

# Collect regression results, with CI
lag0_row = pd.DataFrame(
  
)
model_with_fe_time_coefs = (
  
)
model_with_fe_time_coefs = pd.concat(
  [], 
  ignore_index = True
)

# Plot figure with coefficient estimates
polluter_plot = (
  ggplot()
)
polluter_plot.draw()
