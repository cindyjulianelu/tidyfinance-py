# 01_wrds_crsp_compustat.py
import pandas as pd
import numpy as np
import sqlite3
import os
from plotnine import *
from mizani.formatters import comma.format, percent format
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()

# Starting and end query data dates
start_date = "01/01/1960"
end_date = "12/31/2022"

# With push multi-factor authentication, since May 2023
connection_string = (
  "postgresql+psycopg2://"
 f"{os.getenv('WRDS_USER')}:{os.getenv('WRDS_PASSWORD')}"
  "@wrds-pgdata.wharton.upenn.edu:9737/wrds"
)
wrds = create_engine(
  connection_string, pool_pre_ping = True)

# Downloading Monthly CRSP data
