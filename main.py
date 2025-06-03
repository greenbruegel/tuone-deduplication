import os
import re
import sys
from datetime import datetime
from itertools import combinations
from collections import defaultdict
from copy import copy

import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font
from rapidfuzz import fuzz
from neo4j import GraphDatabase
from pymongo import MongoClient

# Ensure access to project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import cleaning functions
from src.functions import (
    run_factory_centric_enrichment,
    df_to_lower,
    df_normalize_nfkd,
    df_remove_diacritics,
    df_to_strip,
    df_remove_punctuation,
    df_replace_hyphen_with_space,
    df_expand_symbols,
    df_group_by_scenario,
    df_clean_company,
    df_expand_abbreviations
)
from src.mongo import fetch_meta, fetch_nodes_and_rels

# Load Excel
file_path = "reconciliation_outputs_factory.xlsx"
df1 = pd.read_excel(file_path, sheet_name="summary_view_factory")

subset_cols = ["factory_name", "factory_country", "factory_city", "owner_company_name", "product_name"]

change_log = []

# -----------------------------
# Preprocessing Count Functions
# -----------------------------
def apply_and_log(df, func, func_name, subset_cols):
    df, counts = func(df.copy(), subset_cols)
    counts['function'] = func_name
    change_log.append(counts)
    return df

# Apply normalization steps with logging
df1 = apply_and_log(df1, df_to_lower, "lower", subset_cols)
df1 = apply_and_log(df1, df_remove_diacritics, "diacritics", subset_cols)
df1 = apply_and_log(df1, df_normalize_nfkd, "nfkd", subset_cols)
df1 = apply_and_log(df1, df_to_strip, "strip", subset_cols)

# Show preview
print(df1.head())

# Display change summary
change_df = pd.DataFrame(change_log).set_index("function").fillna(0).astype(int)
print("\nChange Count Table:")
print(change_df)

import pandas as pd
import numpy as np
import recordlinkage
import networkx as nx



# ----------------------------
# 2. Normalize for Flat Comparison
# ----------------------------
def normalize(val):
    if isinstance(val, list) and val:
        return sorted(set(str(x).strip().lower() for x in val if str(x).strip()))
    elif pd.notna(val):
        return [str(val).strip().lower()]
    else:
        return []

for col in ['factory_name', 'factory_country', 'factory_city', 'owner_company_name', 'product_name']:
    df1[col] = df1[col].apply(normalize)

# Flatten first element for blocking & string matching
for col in ['factory_name', 'factory_country', 'factory_city', 'owner_company_name', 'product_name']:
    df1[f'{col}_flat'] = df1[col].apply(lambda x: x[0] if x else '')

# ----------------------------
# 3. Blocking (on country + city)
# ----------------------------
indexer = recordlinkage.Index()
indexer.block('factory_country_flat')

candidate_links = indexer.index(df1)

# ----------------------------
# 4. Compare
# ----------------------------
compare = recordlinkage.Compare()
compare.exact('factory_country_flat', 'factory_country_flat', label='country')
compare.exact('factory_city_flat', 'factory_city_flat', label='city')
compare.string('factory_name_flat', 'factory_name_flat', method='jarowinkler', threshold=0.90, label='name')
compare.exact('owner_company_name_flat', 'owner_company_name_flat', label='owner')
compare.string('product_name_flat', 'product_name_flat', method='jarowinkler', threshold=0.90, label='product')

features = compare.compute(candidate_links, df1)

# ----------------------------
# 5. Matching Logic
# ----------------------------
matches = features[
    (features['country'] == 1) &
    (features['city'] == 1) &
    (
        (features['name'] == 1) |
        (features['owner'] == 1) |
        (features['product'] == 1)
    )
]

match_pairs = matches.index.tolist()
print("Matched Pairs:")
print(match_pairs)

# ----------------------------
# 6. Grouping Matched Records
# ----------------------------
G = nx.Graph()
G.add_edges_from(match_pairs)

record_to_group = {}
for group_id, component in enumerate(nx.connected_components(G)):
    for record in component:
        record_to_group[record] = group_id

print("\nRecord to Group Mapping:")
print(record_to_group)

# ----------------------------
# 7. Annotate Output
# ----------------------------
df1['group_id'] = -1
df1['is_duplicate'] = False

for record_idx, group_id in record_to_group.items():
    df1.at[record_idx, 'group_id'] = group_id
    df1.at[record_idx, 'is_duplicate'] = True

# ----------------------------
# 8. Display Results
# ----------------------------
print("\nFinal Annotated Output:")
print(df1[['factory_name_flat', 'product_name_flat', 'group_id', 'is_duplicate']])

# Save to file if needed
df1.to_excel("full_step1.xlsx", index=False)
df1 = df1.rename(columns={"is_duplicate": "is_duplicate_step1", "group_id": "group_id_step1"})
df2 = df1[df1['is_duplicate_step1'] == True].copy()
df3 = df1[df1['is_duplicate_step1'] == False].copy()


df2.to_excel("is_duplicate_step1.xlsx", index=False)
df3.to_excel("is_not_duplicate_step1.xlsx", index=False)



# Step 2 -------------------------------------------------------

pattern = r"[.,;!?]"

# Apply normalization steps with logging
df1 = apply_and_log(df1, df_remove_punctuation , "ponctuation", subset_cols)
df1 = apply_and_log(df1, df_replace_hyphen_with_space, "hyphen", subset_cols)
df1 = apply_and_log(df1, df_expand_symbols, "symbol", subset_cols)

# Show preview
print(df1.head())

# Display change summary
change_df = pd.DataFrame(change_log).set_index("function").fillna(0).astype(int)
print("\nChange Count Table:")
print(change_df)



# ----------------------------
# 2. Normalize for Flat Comparison
# ----------------------------
def normalize(val):
    if isinstance(val, list) and val:
        return sorted(set(str(x).strip().lower() for x in val if str(x).strip()))
    elif pd.notna(val):
        return [str(val).strip().lower()]
    else:
        return []

for col in ['factory_name', 'factory_country', 'factory_city', 'owner_company_name', 'product_name']:
    df1[col] = df1[col].apply(normalize)

# Flatten first element for blocking & string matching
for col in ['factory_name', 'factory_country', 'factory_city', 'owner_company_name', 'product_name']:
    df1[f'{col}_flat'] = df1[col].apply(lambda x: x[0] if x else '')

# ----------------------------
# 3. Blocking (on country + city)
# ----------------------------
indexer = recordlinkage.Index()
indexer.block('factory_country_flat')

candidate_links = indexer.index(df1)

# ----------------------------
# 4. Compare
# ----------------------------
compare = recordlinkage.Compare()
compare.exact('factory_country_flat', 'factory_country_flat', label='country')
compare.exact('factory_city_flat', 'factory_city_flat', label='city')
compare.string('factory_name_flat', 'factory_name_flat', method='jarowinkler', threshold=0.90, label='name')
compare.exact('owner_company_name_flat', 'owner_company_name_flat', label='owner')
compare.string('product_name_flat', 'product_name_flat', method='jarowinkler', threshold=0.90, label='product')

features = compare.compute(candidate_links, df1)

# ----------------------------
# 5. Matching Logic
# ----------------------------
matches = features[
    (features['country'] == 1) &
    (features['city'] == 1) &
    (
        (features['name'] == 1) |
        (features['owner'] == 1) |
        (features['product'] == 1)
    )
]

match_pairs = matches.index.tolist()
print("Matched Pairs:")
print(match_pairs)

# ----------------------------
# 6. Grouping Matched Records
# ----------------------------
G = nx.Graph()
G.add_edges_from(match_pairs)

record_to_group = {}
for group_id, component in enumerate(nx.connected_components(G)):
    for record in component:
        record_to_group[record] = group_id

print("\nRecord to Group Mapping:")
print(record_to_group)

# ----------------------------
# 7. Annotate Output
# ----------------------------
df1['group_id'] = -1
df1['is_duplicate'] = False

for record_idx, group_id in record_to_group.items():
    df1.at[record_idx, 'group_id'] = group_id
    df1.at[record_idx, 'is_duplicate'] = True

# ----------------------------
# 8. Display Results
# ----------------------------
print("\nFinal Annotated Output:")
print(df1[['factory_name_flat', 'product_name_flat', 'group_id', 'is_duplicate']])

# Save to file if needed
df1.to_excel("full_step1.xlsx", index=False)
df1 = df1.rename(columns={"is_duplicate": "is_duplicate_step2", "group_id": "group_id_step2"})
df4 = df1[df1['is_duplicate_step2'] == True].copy()
df5 = df1[df1['is_duplicate_step2'] == False].copy()


df4.to_excel("is_duplicate_step2.xlsx", index=False)
df5.to_excel("is_not_duplicate_step2.xlsx", index=False)


# STEP 3 ------------------------------------------------

# # Company and product-specific cleaning
subset_cols= ["owner_company_name"]
# Apply normalization steps with logging
df1 = apply_and_log(df1, df_clean_company , "cleanCompany", subset_cols)
subset_cols= ["product_name"]
df1 = apply_and_log(df1, df_expand_abbreviations, "productAbbrevation",["product_name"])


# Show preview
print(df1.head())

# Display change summary
change_df = pd.DataFrame(change_log).set_index("function").fillna(0).astype(int)
print("\nChange Count Table:")
print(change_df)



# ----------------------------
# 2. Normalize for Flat Comparison
# ----------------------------
subset_cols = ["factory_name", "factory_country", "factory_city", "owner_company_name", "product_name"]
def normalize(val):
    if isinstance(val, list) and val:
        return sorted(set(str(x).strip().lower() for x in val if str(x).strip()))
    elif pd.notna(val):
        return [str(val).strip().lower()]
    else:
        return []

for col in ['factory_name', 'factory_country', 'factory_city', 'owner_company_name', 'product_name']:
    df1[col] = df1[col].apply(normalize)

# Flatten first element for blocking & string matching
for col in ['factory_name', 'factory_country', 'factory_city', 'owner_company_name', 'product_name']:
    df1[f'{col}_flat'] = df1[col].apply(lambda x: x[0] if x else '')

# ----------------------------
# 3. Blocking (on country + city)
# ----------------------------
indexer = recordlinkage.Index()
indexer.block('factory_country_flat')

candidate_links = indexer.index(df1)

# ----------------------------
# 4. Compare
# ----------------------------
compare = recordlinkage.Compare()
compare.exact('factory_country_flat', 'factory_country_flat', label='country')
compare.exact('factory_city_flat', 'factory_city_flat', label='city')
compare.string('factory_name_flat', 'factory_name_flat', method='jarowinkler', threshold=0.90, label='name')
compare.exact('owner_company_name_flat', 'owner_company_name_flat', label='owner')
compare.string('product_name_flat', 'product_name_flat', method='jarowinkler', threshold=0.90, label='product')

features = compare.compute(candidate_links, df1)

# ----------------------------
# 5. Matching Logic
# ----------------------------
matches = features[
    (features['country'] == 1) &
    (features['city'] == 1) &
    (
        (features['name'] == 1) |
        (features['owner'] == 1) |
        (features['product'] == 1)
    )
]

match_pairs = matches.index.tolist()
print("Matched Pairs:")
print(match_pairs)

# ----------------------------
# 6. Grouping Matched Records
# ----------------------------
G = nx.Graph()
G.add_edges_from(match_pairs)

record_to_group = {}
for group_id, component in enumerate(nx.connected_components(G)):
    for record in component:
        record_to_group[record] = group_id

print("\nRecord to Group Mapping:")
print(record_to_group)

# ----------------------------
# 7. Annotate Output
# ----------------------------
df1['group_id'] = -1
df1['is_duplicate'] = False

for record_idx, group_id in record_to_group.items():
    df1.at[record_idx, 'group_id'] = group_id
    df1.at[record_idx, 'is_duplicate'] = True

# ----------------------------
# 8. Display Results
# ----------------------------
print("\nFinal Annotated Output:")
print(df1[['factory_name_flat', 'product_name_flat', 'group_id', 'is_duplicate']])

# Save to file if needed
df1.to_excel("full_step1.xlsx", index=False)
df1 = df1.rename(columns={"is_duplicate": "is_duplicate_step3", "group_id": "group_id_step3"})

df6 = df1[df1['is_duplicate_step3'] == True].copy()
df7 = df1[df1['is_duplicate_step3'] == False].copy()


df6.to_excel("is_duplicate_step3.xlsx", index=False)
df7.to_excel("is_not_duplicate_step3.xlsx", index=False)

df1 = df1.drop(columns=["is_duplicate", "group_id"])


# Step 4 (geo)------------------------