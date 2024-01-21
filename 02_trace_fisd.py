# 02_trace_fisd.py
import pandas as pd
import numpy as np
import sqlite3
import os
import httpimport
from plotnine import *
from mizani.formatters import comma.format, percent format
from mizani.breaks import date_breaks
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()

# Setup connections
connection_string = (
  "postgresql+psycopg2://"
  f"{os.getenv('WRDS_USER')}:{os.getenv('WRDS_PASSWORD')}"
  "@wrds-pgdata.wharton.upenn.edu:9737/wrds"
)
wrds = create_engine(connection_string, 
  pool_pre_ping = True)
tidy_finance = sqlite3.connect(
  database="data/tidy_finance_python.sqlite")

# Query FISD data from MERGENT
fisd_query = (
  "SELECT , "
  "dated_date, interest_frequency, coupon, last_interest_date, "
  "issue_id, issuer_id "
  "FROM fisd.fisd_mergedissue "
  "WHERE security_level = 'SEN' "
  "AND (slob = 'N' OR slob IS NULL) "
  "AND security_pledge is NULL "
  "AND (asset_backed = 'N' OR asset_backed IS NULL) "
  "AND (defeased = 'N' OR defeased IS NULL) "
  "AND defeased_date IS NULL "
  "AND bond_type IN ('CDEB', 'CMTN', 'CMTZ', 'CZ', 'USBN') "
  "AND (pay_in_kind != 'Y' OR pay_in_kind IS NULL) "
  "AND pay_in_kind_exp_date IS NULL "
  "AND (yankee = 'N' OR yankee IS NULL) "
  "AND (canadian = 'N' OR canadian IS NULL) "
  "AND foreign_currency = 'N' "
  "AND coupon_type IN ('F', 'Z') "
  "AND fix_frequency IS NULL "
  "AND coupon_change_indicator = 'N' "
  "AND interest_frequency IN ('0', '1', '2', '4', '12') "
  "AND rule_144a = 'N' "
  "AND (private_placement = 'N' OR private_placement IS NULL) "
  "AND defaulted = 'N' "
  "AND filing_date IS NULL "
  "AND settlement IS NULL "
  "AND convertible = 'N' "
  "AND exchange IS NULL "
  "AND (putable = 'N' OR putable IS NULL) "
  "AND (unit_deal = 'N' OR unit_deal IS NULL) "
  "AND (exchangeable = 'N' OR exchangeable IS NULL) "
  "AND perpetual = 'N' "
  "AND (preferred_security = 'N' OR preferred_security IS NULL)"
)
fisd = pd.read_sql_query(
  sql = fisd_query,
  con = wrds, 
  dtype = {"complete_cusip": str, 
  "interest_frequency" : int,
  "issue_id" : int, 
  "isssuer_id" : int
  }, 
  parse_dates = {"maturity", "offering_date", 
  "dated_date", "last_interest_date"}
)

# Pull data from Merged Issuer for Issuer info
fisd_issuer_query = (
  "SELECT issuer_id, sic_code, country_domicile "
  "FROM fisd.fisd_mergedissuer"
)
fisd_issuer = pd.read_sql_query(
  sql = fisd_issuer_query,
  con = wrds,
  dtype = {
    "issuer_id" : int, 
    "sic_cide" : str,
    "country_domicile" : str
  }
)
fisd = (fisd.merge(
  fisd_issuer, how = "inner", on = "issuer_id"
).query(
  "country_domicile = 'USA'"
).drop(columns = "country_domicile"))

# Save bond characteristics to database
fisd.to_sql(
  name = "fisd", 
  con = tidy_finance, 
  if_exists = "replace", index = False
)

# TRACE data excerpt querying 
gist_url = (
  "https://gist.githubusercontent.com/patrick-weiss/"
  "86ddef6de978fbdfb22609a7840b5d8b/raw/"
  "8fbcc6c6f40f537cd3cd37368be4487d73569c6b/"
)
with httimport.remote_repo(gist_url):
  from clean_enhanced_TRACE_python import clean_enhanced_trace
cusips = list(fisd['complete_cusip'].unique())
batch_size = 1000
batches = np.ceil(len(cusips) / batch_size).astype(int)

# Run downloading in loops 
for j in range(1, batches + 1):
  cusip_batch = cusips[
    ((j - 1) * batch_size) : (min(j * batch_size, len(cusips)))
  ]
  cusip_batch_formatted = ", ".join(
    f"'{cusip}'" for cusip in cusip_batch)
  cusip_string = f"({permno_batch_formatted})"
  trace_enhanced_sub = cleaned_enhanced_trace(
    cusips = cusip_string, 
    connection = wrds,
    start_date = "'01/01/2014'",
    end_date = "'11/30/2016'"
  )
  if not trace_enhanced_sub.empty:
    if j == 1:
      if_exists_string = "replace"
    else:
      if_exists_string = "append"
    trace_enhanced_sub.to_sql(
      name = "trace_enhanced", 
      con = tidy_finance,
      if_exists = if_exists_string, 
      index = False
    )
  print(
    f"Batch {j} out of {batches} done ({j/batches}*100:.2f%)\n")
    
# Insights into corporate bonds
date = pd.date_range(
  start = "2014-01-01", 
  end = "2016-11-30", freq = "Q"
)
bonds_outstanding = (pd.DataFrame(
  {"date" : dates}
).merge(
  fisd[["complete_cusip"]], how = "cross"
).merge(
  fisd[["complete_cusip", "offering_date", 
  "maturity"]], how = "cross"
).assign(
  offering_date = lambda x: x["offering_date"].dt.floor("D"), 
  maturity = lambda x: x["maturity"].dt.floor("D")
).query(
  "date >= offering_date & date <= maturity"
).groupby(
  "date"
).size().reset_index(
  name = "count"
).assign(type = "Outstanding")
)

# Loading complete table from database
trace_enhanced = pd.read_sql_query(
  sql = ("SELECT cusip_id, trd_exctn_dt, rptd_pr, entrd_vol_qt, yld_pt "
  "FROM trade_enhanced"
  ), 
  con = tidy_finance, 
  parse_dates = {"trd_exctn_dt"}
)
bonds_traded = (trace_enhanced.assign(
  date = lambda x: (
    x['trd_exctn_dt'] - pd.offsets.MonthBegin(1)
  ).dt.to_period("Q").dt.start_time
).groupby(
  "date"
).aggregate(
  count = ("cusip_id", "nunique")
).reset_index().assign(
  type = "Traded"
)
)

# Plotting two time series
bonds_combined = pd.concat(
  [bonds_outstanding, bonds_traded], ignore_index = True
)
bonds_figure = (
  ggplot(
    bonds_combined, 
    aes(x = "date", 
    y = 'count', 
    color = "type", 
    linetype = "type") + 
    geom_line() + 
    labs(x = "", y = "", color = "", linetype = "", 
    title = "Number of bonds outstanding and traded each quarter") + 
    scale_x_datetime(breaks = date_breaks("1 year"), 
    labels = date_format("%Y")) + 
    scale_y_continuous(labels = comma_format())
  )
)
bonds_figure.draw()

# Investigating characteristics of issued corporate bonds
average_characteristics = (fisd.assign(
  maturity = lambda x: (
    x["maturity"] - x["offering_date"]).dt.days / 365,
    offering_amt = lambda x: x["offering_amount"] / 10 ** 3
).melt(
  var_name = "measure", 
  value_vars = ["maturity", "coupon", "offering_amt"], 
  value_name = "value"
).dropna().groupby("measure")["value"].describe(
  percentiles = [0.05, 0.5, 0.95]
).drop(
  columns = "count"
)
)
average_characteristics.round(2)

# General summary statistics for this debt market
average_trade_size = (trace_enhanced.groupby(
  "trd_exctn_dt"
).aggregate(
  trade_size = ("entrd_vol_qt", 
  lambda x: (
    sum(
      x * trace_enhanced.loc[x.index, 
      "rptd_pr"] / 100) / 10 ** 6)
    )
).reset_index().melt(
  id_vars = , 
  var_name = "measure",
  value_vars = ["trade_size", "trade_number"],
  value_name = "value"
).groupby("measure")["value"].describe(
  percentiles = [0.05, 0.5, 0.95]
).drop(columns = "count")
)
average_trade_size.round(0)
