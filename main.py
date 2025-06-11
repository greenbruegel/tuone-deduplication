import pandas as pd
from src.step_1 import TextCleaner
from src.step_2 import standardise_country_city
import logging
import sys
from datetime import datetime
import time
from tqdm import tqdm
import ast
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/pipeline_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Start timing the entire pipeline
t0_pipeline = time.time()

# Input file and columns to validate
file_path = "./storage/input/reconciliation_outputs_factory.xlsx"
subset_cols = [
    "factory_country",
    "factory_city",
    "owner_company_name",
    "product_name"
]

# 1) Load and validate data
logging.info(f"Reading input file: {file_path}")
df = pd.read_excel(file_path, sheet_name="summary_view_factory")
initial_len = len(df)

df = df.dropna(subset=subset_cols)
logging.info(f"Dropped {initial_len - len(df)} rows; {len(df)} remain after validation")

if df.empty:
    logging.error("No valid rows remaining after validation. Exiting.")
    sys.exit(1)

# 2) Explode list columns
list_cols = ["owner_company_name", "product_name"]

def safe_eval_list(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            s = x.strip()
            if s.startswith('[') and s.endswith(']'):
                return ast.literal_eval(s)
            return [s]
        except Exception:
            return [x]
    return [x]

for col in list_cols:
    if col in df.columns:
        df[col] = df[col].apply(safe_eval_list)
        df = df.explode(col)
        logging.info(f"Exploded column {col}")

df.reset_index(drop=True, inplace=True)
logging.info(f"DataFrame has {len(df)} rows after exploding lists")

# 3) Cleaning pipeline on raw data
cleaner = TextCleaner()
pipeline = [
    (cleaner.to_lower, "lower"),
    (cleaner.remove_diacritics, "diacritics"),
    (cleaner.normalize_nfkd, "nfkd"),
    (cleaner.strip, "strip"),
    (cleaner.remove_punctuation, "punctuation"),
    (cleaner.replace_hyphen_with_space, "hyphen"),
    (cleaner.expand_symbols, "symbol")
]
change_log = []
for fn, label in pipeline:
    logging.info(f"Starting pipeline step: {label}")
    df, counts = fn(df, subset_cols)
    change_log.append(counts)
    changes = ", ".join(f"{c}:{cnt}" for c,cnt in counts.items() if cnt)
    logging.info(f"Step {label} done. Changes: {changes or 'none'}")

# 4) Deduplicate cleaned and standardize locations
unique_loc = df[['factory_city', 'factory_country']].drop_duplicates().reset_index(drop=True)
logging.info(f"Found {len(unique_loc)} unique city-country combinations to standardize")

# Estimate and log
ess_sec = len(unique_loc)
logging.info(f"Estimated time for geonames calls: {ess_sec//60}m {ess_sec%60}s")

# Standardize locations
start_loc = time.time()
unique_loc = standardise_country_city(
    unique_loc,
    city_col="factory_city",
    country_col="factory_country",
    verbose=True,     # <- prints each lookup as it happens
    details=False 
)
elapsed_loc = time.time() - start_loc
logging.info(f"Location standardization took {elapsed_loc:.2f}s ({elapsed_loc/len(unique_loc):.2f}s per lookup)")

# 5) Build mapping keyed by (city, country) and log mappings by (city, country) and log mappings
unique_loc['_key'] = list(zip(unique_loc.factory_city, unique_loc.factory_country))

maps = {}
def build_and_log_map(col_name, series):
    mp = dict(zip(unique_loc['_key'], series))
    for k, v in mp.items():
        logging.info(f"Mapping prepared for {col_name}: {k} -> {v}")
    return mp

maps['country_standardized'] = build_and_log_map('country_standardized', unique_loc['country_standardized'])
maps['country_iso2']        = build_and_log_map('country_iso2',        unique_loc['country_iso2'])
maps['country_not_found']   = build_and_log_map('country_not_found',   unique_loc['country_not_found'])
maps['factory_city_adm_name']  = build_and_log_map('factory_city_adm_name',  unique_loc['factory_city_adm_name'])
maps['factory_city_adm_code']  = build_and_log_map('factory_city_adm_code',  unique_loc['factory_city_adm_code'])
maps['factory_city_adm_level'] = build_and_log_map('factory_city_adm_level', unique_loc['factory_city_adm_level'])
maps['factory_city_latitude']  = build_and_log_map('factory_city_latitude',  unique_loc['factory_city_latitude'])
maps['factory_city_longitude'] = build_and_log_map('factory_city_longitude', unique_loc['factory_city_longitude'])
maps['factory_city_not_found'] = build_and_log_map('factory_city_not_found', unique_loc['factory_city_not_found'])

# 6) Apply mappings back to main df
logging.info("Mapping standardized location data back to original DataFrame")
df['_key'] = list(zip(df.factory_city, df.factory_country))
for col, mp in maps.items():
    df[col] = df['_key'].map(mp)
df.drop(columns=['_key'], inplace=True)

# 7) Compute clusters
group_cols = ['country_standardized', 'factory_city_adm_name', 'owner_company_name', 'product_name']
complete_mask = df[group_cols].notna().all(axis=1)
df['cluster_num'] = np.nan 
df['cluster_id']  = '000000'
tmp = df.loc[complete_mask].copy()
tmp['cluster_num'] = tmp.groupby(group_cols).ngroup() + 1
sizes = tmp.groupby('cluster_num')['factory_city_adm_name'].transform('size')

tmp['cluster_id'] = (
    sizes.gt(1)                       # keep only clusters with ≥2 rows
    .mul(tmp['cluster_num'])          # multiply to keep the number or 0
    .astype(int)
    .astype(str)
    .str.zfill(6)                     # pad to six digits
)
df.loc[complete_mask, ['cluster_num', 'cluster_id']] = tmp[['cluster_num', 'cluster_id']]

# 8) Save output
output_cols = [col for col in [
    'factory_unique_id','factory_name','factory_city','factory_city_adm_name',
    'factory_country','country_standardized','country_iso2','country_not_found',
    'owner_company_unique_id','owner_company_name','product_name','factory_city_adm_code',
    'factory_city_adm_level','factory_city_latitude','factory_city_longitude','cluster_id'
] if col in df.columns]
output_file = "./storage/output/clean_output.xlsx"
logging.info(f"Saving output to {output_file}")
df[output_cols].to_excel(output_file, index=False)

# Final timing
t1_pipeline = time.time()
logging.info(f"Total pipeline time: {(t1_pipeline - t0_pipeline)/60:.2f} minutes")