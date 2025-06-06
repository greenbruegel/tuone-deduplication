import os  
import sys
from collections import defaultdict
import ast

import pandas as pd
import numpy as np
import networkx as nx
import recordlinkage
import textdistance
from recordlinkage.base import BaseCompareFeature
import requests
import time
import pycountry
from rapidfuzz import fuzz
import geopandas
import geopy
import neo4j
import cleanco

# Load NLP model
import spacy
nlp = spacy.load("en_core_web_sm")

# Ensure access to project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Custom cleaning functions
from src.functions import (
    df_to_lower,
    df_normalize_nfkd,
    df_remove_diacritics,
    df_to_strip,
    df_remove_punctuation,
    df_replace_hyphen_with_space,
    df_expand_symbols,
    df_clean_company,
    df_expand_abbreviations,
    df_lemmatize_columns_spacy
)

def apply_and_log(df, func, func_name, subset_cols):
    '''Collects meta information about how many changes are caused by each cleaning function'''
    df, counts = func(df.copy(), subset_cols)
    counts['function'] = func_name
    change_log.append(counts)
    return df

def wrap_in_list(val):
    # Case 1: already a list → return as-is
    if isinstance(val, list):
        return val
    
    # Case 2: string that looks like a list → safely convert it
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass  # not a list-looking string

        return [val.strip()]  # fallback: wrap single string

    # Case 3: NaN or None
    return []

GEONAMES_USERNAME = "mjuge"

def get_iso2_country_code(name):
    try:
        if not name or str(name).strip().lower() == "nan":
            return None
        return pycountry.countries.lookup(name.strip()).alpha_2
    except:
        return None

def get_adm2_from_geonames(city, country=None):
    base_url = "http://api.geonames.org/searchJSON"
    params = {
        "q": city,
        "maxRows": 1,  # get only the first match
        "style": "FULL",
        "username": GEONAMES_USERNAME,
    }

    if country:
        country_code = get_iso2_country_code(country)
        if country_code:
            params["country"] = country_code

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        valid_codes = {"PPL", "PPLA", "PPLA2", "PPLC", "PPLG"}

        if "geonames" in data and len(data["geonames"]) > 0:
            result = data["geonames"][0]

            feature_class = result.get("fcl")
            feature_code = result.get("fcode")

            # ✅ DEBUG PRINT
            print(f"GeoNames → '{city}' → featureClass: '{feature_class}', featureCode: '{feature_code}'")
            print(f"→ Querying GeoNames with city='{city}', country='{country}' (ISO2={country_code})")

            # ✅ VALIDATION CHECK
            if feature_class == "P" and feature_code in valid_codes:
                adm_name = result.get("adminName2")
                adm_code = result.get("adminCode2")
                return adm_name, adm_code, False  # False = not flagged
            else:
                return None, None, True  # Invalid: not a populated place

        return None, None, True  # No results found

    except Exception as e:
        print(f"API error for city: {city}, country: {country} → {e}")
        return None, None, True

def df_assign_adm2_to_city_column(df, city_col, country_col=None):
    print(f"\nProcessing ADM2 from '{city_col}'" + (f" with fallback country column '{country_col}'" if country_col else ""))
    df = df.copy()

    # Create column names
    adm2name_col = f"{city_col}_adm2name"
    adm2code_col = f"{city_col}_adm2code"
    factory_not_in_adm2_col = f"{city_col}_not_in_adm2"

    # Flatten city column
    df[city_col] = df[city_col].apply(lambda x: x[0] if isinstance(x, list) and x else x)

    # Flatten country column if needed
    if country_col and country_col in df.columns:
        df[country_col] = df[country_col].apply(lambda x: x[0] if isinstance(x, list) and x else x)

    # Initialize new columns
    df[adm2name_col] = None
    df[adm2code_col] = None
    df[factory_not_in_adm2_col] = False

    # Enrich with ADM2
    for idx, row in df.iterrows():
        city = row[city_col]
        if not isinstance(city, str) or city.strip().lower() == "nan":
            continue

        country = None
        if country_col and country_col in df.columns:
            country_raw = row[country_col]
            if isinstance(country_raw, str) and country_raw.strip().lower() != "nan":
                country = country_raw.strip()

        adm2name, adm2code, factory_not_in_adm2 = get_adm2_from_geonames(city.strip(), country)

        df.at[idx, factory_not_in_adm2_col] = factory_not_in_adm2

        if factory_not_in_adm2:
            print(f"{idx}: '{city}' ({country}) → not a valid populated place → ADM2 skipped")
            continue

        print(f"{idx}: '{city}' ({country}) → name: '{adm2name}', code: '{adm2code}'")
        

        df.at[idx, adm2name_col] = adm2name
        df.at[idx, adm2code_col] = adm2code

        time.sleep(1)  # Respect GeoNames rate limits

    return df

# ----------------------------
# Custom list-string matcher -  this compares two columns of lists of strings, and returns 1 if any string in one list sufficiently matches any string in the other list
# ----------------------------
class CompareAnyStringListMatch(BaseCompareFeature):
    def __init__(self, left_on, right_on=None, threshold=0.80, method='jarowinkler', label=None):
        super().__init__(left_on, right_on, label)
        self.threshold = threshold
        self.method = method
        self.label = label

        if self.method == 'jarowinkler':
            self.scorer = textdistance.jaro_winkler.normalized_similarity
        elif self.method == 'levenshtein':
            self.scorer = textdistance.levenshtein.normalized_similarity
        else:
            raise ValueError(f"Unsupported method '{self.method}'.")

    def _compute_vectorized(self, s1, s2):
        results = []
        for idx, (list1, list2) in enumerate(zip(s1, s2)):
            def is_missing(val):
                if val is None:
                    return True
                if isinstance(val, float) and pd.isna(val):
                    return True
                if isinstance(val, str) and val.strip().lower() == 'nan':
                    return True
                if isinstance(val, list):
                    val = [str(x).strip().lower() for x in val if str(x).strip()]
                    return not val or val == ['nan']
                return False

            if is_missing(list1) or is_missing(list2):
                results.append(0)
                continue

            clean1 = [str(x).strip() for x in list1 if str(x).strip().lower() != 'nan']
            clean2 = [str(x).strip() for x in list2 if str(x).strip().lower() != 'nan']

            matched_pairs = []
            for item1 in clean1:
                for item2 in clean2:
                    score = self.scorer(item1, item2)
                    if score >= self.threshold:
                        matched_pairs.append((item1, item2, score))

            if matched_pairs:
                print(f"\n✅ Match found:")
                print(f"  List 1: {clean1}")
                print(f"  List 2: {clean2}")
                print(f"  Matching elements:")
                for m1, m2, score in matched_pairs:
                    print(f"    - '{m1}' ⇄ '{m2}' | Score: {score:.4f}")
                results.append(1)
            else:
                results.append(0)

        return pd.Series(results, index=s1.index, name=self.label)

# ----------------------------
# Load Data & Preprocess
# ----------------------------

file_path = "reconciliation_outputs_factory.xlsx"
df1 = pd.read_excel(file_path, sheet_name="factory")
df1 = df1.head(10)
subset_cols = ["factory_name", "factory_country", "factory_city", "owner_company_name", "product_name"]
change_log = []

# Apply cleaning
for func, name in [
    (df_to_lower, "lower"),
    (df_remove_diacritics, "diacritics"),
    (df_normalize_nfkd, "nfkd"),
    (df_to_strip, "strip")
]:
    df1 = apply_and_log(df1, func, name, subset_cols)

print(df1.head())
print("\nChange Count Table:")
print(pd.DataFrame(change_log).set_index("function").fillna(0).astype(int))

# ----------------------------
# Wrap cleaned strings into lists for matching (only for specific fields that may contain multiple values)
# ----------------------------

cols_to_wrap = ['owner_company_name', 'product_name']
for col in cols_to_wrap:
    df1[col] = df1[col].apply(wrap_in_list)

# ✅ Ensure proper dtype for compatibility with recordlinkage.BaseCompareFeature
df1['product_name'] = df1['product_name'].astype(object)
df1['owner_company_name'] = df1['owner_company_name'].astype(object)

# Step 2 -------------------------------------------------------

pattern = r"[.,;!?]"

# Company and product-specific cleaning
subset_cols= ["owner_company_name"]
df1 = apply_and_log(df1, df_clean_company , "cleanCompany", subset_cols)
# Apply normalization steps with logging
subset_cols = ["factory_name", "factory_country", "factory_city", "owner_company_name", "product_name"]
df1 = apply_and_log(df1, df_remove_punctuation , "ponctuation", subset_cols)
df1 = apply_and_log(df1, df_replace_hyphen_with_space, "hyphen", subset_cols)
df1 = apply_and_log(df1, df_expand_symbols, "symbol", subset_cols)

# STEP 3 ------------------------------------------------
subset_cols= ["product_name"]
df1 = apply_and_log(df1, df_expand_abbreviations, "productAbbrevation",["product_name"])

subset_cols= ["product_name"]
df1 = apply_and_log(df1, df_lemmatize_columns_spacy , "lemma", subset_cols)
# subset_cols= ["product_name"]
# df1 = apply_and_log(df1, df_to_strip, "strip", subset_cols)

# Display change summary
change_df = pd.DataFrame(change_log).set_index("function").fillna(0).astype(int)
print("\nChange Count Table:")
print(change_df)

# Step 4 (geo)------------------------ apply admin2
df1 = df_assign_adm2_to_city_column(df1, city_col="factory_city", country_col="factory_country")

# ----------------------------
# Blocking - we only allow comparison for entries that are in the same country ### we could add TECH to the blocking (would need to be an option from list)
# ----------------------------
indexer = recordlinkage.Index()
indexer.block('factory_country')

candidate_links = indexer.index(df1)

# ----------------------------
# Compare
# ----------------------------
compare = recordlinkage.Compare() #initialises a comparison engine
compare.exact('factory_country', 'factory_country', label='country') 
compare.exact('factory_city', 'factory_city', label='city')
compare.exact('factory_city_adm2code', 'factory_city_adm2code', label='city_adm2code')
compare.string('factory_name', 'factory_name', label='name')

compare.add(CompareAnyStringListMatch('product_name', 'product_name', threshold=0.80, method='jarowinkler', label='product'))
compare.add(CompareAnyStringListMatch('owner_company_name', 'owner_company_name', threshold=0.80, method='jarowinkler', label='owner'))

features = compare.compute(candidate_links, df1)

# ----------------------------
# Match Logic - country must be identical, city name or code must be identical, owner must be identical, consider how to include product. 
# ----------------------------

matches = features[
    (features['country'] == 1) &
    (
        (features['city'] == 1) |
        (features['city_adm2code'] == 1)
    ) &
    (features['owner'] == 1)
]

match_pairs = matches.index.tolist()
print("Matched Pairs:", match_pairs)

# ----------------------------
# Grouping
# ----------------------------
G = nx.Graph()
G.add_edges_from(match_pairs)
record_to_group = {record: group_id for group_id, component in enumerate(nx.connected_components(G)) for record in component}

# ----------------------------
# Annotate Output
# ----------------------------
df1['group_id'] = -1
df1['is_duplicate'] = False

for record_idx, group_id in record_to_group.items():
    df1.at[record_idx, 'group_id'] = group_id
    df1.at[record_idx, 'is_duplicate'] = True

# ----------------------------
# Save Output
# ----------------------------
df1[['factory_unique_id','factory_name','factory_city','factory_city_adm2name','factory_country',
     'owner_company_unique_id', 'owner_company_name', 'owner_jv_unique_id', 'owner_jv_name', 
     'product_unique_id','product_name','product_name_lemma',
     'investment_unique_id','investment_status','investment_amount','investment_phase',
     'capacity_unique_id','capacity_status','capacity_amount','capacity_phase',
     'factory_city_adm2code','factory_city_not_in_adm2','group_id','is_duplicate'
     ]].to_excel("clean_output_ben.xlsx", index=False)

#df1 = df1.rename(columns={"is_duplicate": "is_duplicate_step1", "group_id": "group_id_step1"})
#df1[df1['is_duplicate_step1']].to_excel("is_duplicate_step1.xlsx", index=False)
#df1[~df1['is_duplicate_step1']].to_excel("is_not_duplicate_step1.xlsx", index=False)

# #Change print ---------------------------
# change_df.to_excel("changes.xlsx",index=True)
