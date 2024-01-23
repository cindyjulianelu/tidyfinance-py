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
from klearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, Lasso, Ridge

# Data preparation
