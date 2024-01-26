# 12_option_pricing_ml.py
import pandas as pd
import numpy as np
from plotnine import *
from itertools import product
from scipy.stats import norm 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso

# Options Pricing
def black_scholes_price(S, K, r, T, sigma):
  d1 = (np.log(S/K) + (r + sigma**2 / 2) * T)/(
    sigma * np.sqrt(T)
  )
  d2 = d1 - sigma * np.sqrt(T)
  price = S * norm.cdf(d1) - K * np.ext(
    -r * T
  ) * norm.cdf(d2)
  return price

# Data simulation
random_state = 42
np.random.seed(random_state)
S = np.arange(40, 61)
K = np.arange(20, 91)
r = np.arange(0, 0.051, 0.01)
T = np.arange(3/12, 2.01, 1/12)
sigma = np.arange(0.1, 0.81, 0.1)
option_prices = pd.DataFrame(
  product(S, K, r, T, sigma), 
  columns = ["S", "K", "r", "T", 
  "sigma"]
)
option_prices["black_scholes"] = black_scholes_price(
  option_prices["S"].values, 
  option_prices["K"].values, 
  option_prices["r"].values, 
  option_prices["T"].values, 
  option_prices["sigma"].values
)
option_prices = (option_prices.assign(
   observed_price = lambda x: (
     x["black_scholes"] + np.random.normal(
       scale = 0.15
     )
   )
)
)

# Train and testing data
train_data, test_data = train_test_split(
  option_prices, 
  test_size = 0.01, 
  random_state = random_state
)
preprocessor = ColumnTransformer(
  transformers = [(
    "mnormalize_predictors", 
    StandardScaler(), 
    ["S", "K", "r", "T", "sigma"]
  )], 
  remainder = "drop"
)

# Single layer networks and random forests
max_iter = 1000
nnet_model = MLPRegressor(
  hidden_layer_sizes = 10, 
  max_iter = max_iter, 
  random_state = random_state
)
nnet_pipeline = Pipeline([
  ("preprocessor", preprocessor),
  ("regressor", nnet_model)
])
nnet_fit = nnet_pipeline.fit(
  train_data.drop(
    columns = ["observed_price"]
  ), 
  train_data.get("observed_price")
)
rf_model = RandomForestRegressor(
  n_estimators = 50, 
  min_samples_leaf = 2000, 
  random_state = random_state
)
rf_pipeline = Pipeline([
  ("preprocessor", preprocessor), 
  ("regressor", rf_model)
])
rf.fit = rf_pipeline.fit(
  train_data.drop(
    columns = ["observed_price"]
  ), 
  train_data.get("observed_price")
)

# Deep neural net
deepnnet_model = MLPRegressor(
  hidden_layer_sizes = (10, 10, 10), 
  activation = "logistic", 
  solver = "lbfgs", 
  max_iter = max_iter, 
  random_state = random_state
)
deepnnet_pipeline = Pipeline([
  ("preprocessor", preprocessor), 
  ("regressor", deepnnet_model)
])
deepnnet_fit = deepnnet_pipeline.fit(
  train_data.drop(
    columns = ["observed_price"]
  ), 
  train_data.get("observed_price")
)

# Universal approximation
lm_pipeline = Pipeline([
  ("polynomial", PolynomialFeatures(
    degree = 5, 
    interaction_only = False, 
    include_bias = True
  )), 
  ("scaler", StandardScaler()), 
  ("regressor", Lasso(alpha = 0.01))
])
lm_fit = lm_pipeline.fit(
  train_data.get([
    "S", "K", "r", "T", "sigma"
  ]), 
  train_data.get("observed_price")
)

# Prediction evaluation
test_X = test_data.get(
  ["S", "K", "r", "T", "sigma"]
)
test_y = test_data.get("observed_price")
predictive_performance = (pd.concat(
  [test_data.reset_index(drop = True), 
  pd.DataFrame({
    "Random forest" : rf_fit.predict(test_X), 
    "Single layer" : nnet_fit.predict(test_X), 
    "Deep NN" : deepnnet_fit.predict(test_X), 
    "Lasso" : lm_fit.predict(test_X)
  })], axis = 1
).melt(
  id_vars = ["S", "K", "r", "T", "sigma", 
  "black_scholes", "observed_price"], 
  var_name = "Model", 
  value_name = "Predicted"
).assign(
  moneyness = lambda x: x["S"] - x["K"],
  pricing_error = lambda x: np.abs(
    x["Predicted"] - x["black_scholes"]
  )
))

# Show the results graphically pricing accuracy
predictive_performance_plot = (
  ggplot(predictive_performance, 
  aes(
    x = "moneyness", 
    y = "pricing_error"
  )) + 
  geom_point(
    alpha = 0.05
  ) + 
  facet_wrap("Model") + 
  labs(
    x = "Moneyness, S-K", 
    y = "Absolute prediction error", 
    title = "Prediction errors of call options, different models"
  ) + 
  theme(legend_position = "")
)
prediction_performance_plot.draw()
