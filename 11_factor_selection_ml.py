# 11_factor_selection_ml.py
import pandas as pd
import numpy as np
import sqlite3
from plotnine import *
from mizani.formatters import percent_format, date_format
from mizani.breaks import date_breaks
from itertools import product
from sklearn.model_selection import(
  train_test_split, 
  GridSeachCV, TimeSeriesSplit, 
  cross_val_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, Lasso, Ridge

# Data preparation
tidy_finance = sqlite3.connect(
  database = "data/tidy_finance_python.sqlite"
)
factors_ff3_monthly = (pd.read_sql_query(
  sql = "SELECT * FROM factors_ff3_monthly", 
  con = tidy_finance, 
  parse_date = {"month"}
).add_prefix("factor_ff_"))
factors_q_monthly = (pd.read_sql_query(
  sql = "SELECT * FROM factors_q_monthly", 
  con = tidy_finance, 
  parse_dates = {"month"}
).add_prefix("factor_q_"))
macro_predictors = (pd.read_sql_query(
  sql = "SELECT *FROM macro_predictors", 
  con = tidy_finance, 
  parse_dates = {"month"}
).add_prefix("macro_"))
industries_ff_monthly = (pd.read_sql_query(
  sql = "SELECT * FROM industries_ff_fmonthly", 
  con = tidy_finance, 
  parse_dates = {"month"}
).melt(
  id_vars = "month", 
  var_name = "industry", 
  value_name = "ret"
))

# Combine all monthly observations into one data frame
data = (industries_ff_monthly.merge(
  factors_ff3_monthly, 
  how = "left", 
  left_on = "month", 
  right_on = "factor_ff_month"
).merge(
  factors_q_monthly, 
  how = "left", 
  left_on = "month", 
  right_on = "factor_q_month"
).merge(
  macro_predictors, 
  how = "left", 
  left_on = "month", 
  right_on = "factor_q_month"
).assign(
  ret_excess = lambda x: x["ret"] - x["factor_ff_rf"]
).drop(
  columns = ["ret", "ractor_ff_month", "factor_q_month", 
  "macro_month"]
).dropna())

# Summary statistics for 10 monthly excess returns in percent
data_plot = (ggplot(
  data, 
  aes(x = "industry", y = "ret_excess")
) + 
geom_boxplot() + 
corrd_flip() + 
labs(x = "", y = "", 
title = "Excess return distributions by industry in percent") + 
scale_y_consitnuous(
  labels = percent_format()
))
data_plot.draw()

# The ML workflow
macro_variables = data.filter(
  like = "macro"
).columns
factor_variables = data.filter(
  like = "factor"
).columns
column_combinations = list(
  product(
    macro_variables, factor_variables
  )
)
new_column_values = []
for macro_column, factor_column in column_combinations:
  new_column_values.append(
    data[macro_column] * data[factor_column]
  )
column_names = [" x ".join(t) for t in column_combinations]
new_columns = pd.DataFrame(
  dict(zip(
    column_names, new_column_values
  ))
)
data = pd.concat([data, new_columns], axis = 1)
preprocessor = ColumnTransformer(
  transormers = [
    ("scale", StandardScaler(), 
    [col for col in data.columns 
    if col not in ["ret_excess", "month", "industry"]])
  ], 
  remainder = "drop",
  verbose_feature_names_out = False
)

# Build a model
lm_model = ElasticNet(
  alpha = 0.007, 
  l1_ratio = 1, 
  lmax_iter = 5000, 
  fit_intercept = False
)
lm_pipeline = Pipeline([
  ("preprocessor", proprocessor), 
  ("regressor", lm_model)
])

# Fitting model
data_manufacturing = data.query(
  "industry == 'manuf'"
)
training_date = "2011-12-01"
data_manufacturing_training = (data_manufacturing.query(
  f"month < '{training_date'"
))
lm_fit = lm_pipeline.fit(
  data_manufacturing_training, 
  data_manufacturing_training.get("ret_excess")
)
predicted_values = (pd.DataFrame({
  "Fitted value" : lm_fit.predict(data_manufacturing), 
  "Realization": data_manufacturing.get("ret_excess")
}).assign(
  month = data_manufacturing["month"]
).melt(
  id_vars = "month", 
  var_name = "Variable", 
  value_name = "return"
))
predicted_values_plot = (
  ggplot(
    predicted_values, 
    aes(
      x = "month", y = "return", 
      color = "Variable", 
      linetype = "Variable"
    )
  ) + 
  annotate(
    "rect", 
    xmin = dataa_manufacturing_training["month"].max(), 
    xmax = data_manufacturing["month"].max(), 
    ymin = -np.Inf, 
    ymax = np.Inf, 
    alpha = 0.25, 
    fill = "#808080"
  ) + 
  geom_line() + 
  labs(
    x = "", y = "", color = "", 
    linetype = "",
    title = "Monthly realized and fitted manufacturing risk premia"
  ) + 
  scale_x_datetime(
    breaks = date_breaks("5 years"), 
    labels = date_format("%Y")
  ) + 
  scale_y_continuous(
    labels = percent_format()
  )
)
predicted_values_plot.draw()

# What do estimated coefficients look like? 
x = preprocessor.fit_transform(data_manufacturing)
y = data_manufacturing["ret_excess"]
alphas = np.logspace(-5, 5, 100)
coefficient_lasso = []
for a in alphas:
  
