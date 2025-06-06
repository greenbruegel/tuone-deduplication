from rapidfuzz import fuzz
import pandas as pd
from itertools import combinations
from collections import defaultdict
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import networkx as nx
import os
import re
import pandas as pd
from collections import defaultdict
from itertools import combinations
from openpyxl import load_workbook
from neo4j import GraphDatabase
from pymongo import MongoClient
import pandas as pd
import os
import pandas as pd
from collections import defaultdict
from itertools import combinations
from datetime import datetime
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font
from copy import copy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import recordlinkage


import pandas as pd
import unicodedata



# -----------------------------
# Preprocessing Functions
# -----------------------------

# Updated df_to_lower with proper logging and count
def df_to_lower(df, subset_cols):
    """
    Lowercase transformation with support for strings and lists of strings.
    Logs all changes and returns column-wise change counts.
    """
    change_counts = {col: 0 for col in subset_cols}

    def lower_case(x):
        if isinstance(x, str):
            lowered = x.lower()
            if lowered != x:
                print(f"Lowered string: '{x}' → '{lowered}'")
                return lowered, 1
            return lowered, 0
        elif isinstance(x, list):
            lowered_list = []
            count = 0
            for i in x:
                if isinstance(i, str):
                    lowered_i = i.lower()
                    if lowered_i != i:
                        print(f"Lowered item in list: '{i}' → '{lowered_i}'")
                        count += 1
                    lowered_list.append(lowered_i)
                else:
                    lowered_list.append(i)
            return lowered_list, count
        return x, 0

    for col in subset_cols:
        print(f"\nProcessing column: '{col}'")
        lowered_results = df[col].apply(lower_case)
        df[col] = lowered_results.apply(lambda x: x[0])
        change_counts[col] = lowered_results.apply(lambda x: x[1]).sum()

    return df, change_counts


import unicodedata

def df_normalize_nfkd(df, subset_cols):
    change_counts = {col: 0 for col in subset_cols}

    def normalize_text(x):
        if isinstance(x, str):
            normalized = unicodedata.normalize('NFKD', x)
            if normalized != x:
                print(f"NFKD normalized string: '{x}' → '{normalized}'")
            return normalized, normalized != x
        elif isinstance(x, list):
            result = []
            count = 0
            for i in x:
                if isinstance(i, str):
                    normalized = unicodedata.normalize('NFKD', i)
                    if normalized != i:
                        print(f"NFKD normalized item in list: '{i}' → '{normalized}'")
                        count += 1
                    result.append(normalized)
                else:
                    result.append(i)
            return result, count
        return x, 0

    for col in subset_cols:
        print(f"\nProcessing column: '{col}'")
        results = df[col].apply(normalize_text)
        df[col] = results.apply(lambda x: x[0])
        change_counts[col] = results.apply(lambda x: x[1]).sum()

    return df, change_counts

def df_to_strip(df, subset_cols):
    change_counts = {col: 0 for col in subset_cols}

    def strip_text(x):
        if isinstance(x, str):
            stripped = x.strip()
            if stripped != x:
                print(f"Stripped string: '{x}' → '{stripped}'")
            return stripped, stripped != x
        elif isinstance(x, list):
            result = []
            count = 0
            for i in x:
                if isinstance(i, str):
                    stripped = i.strip()
                    if stripped != i:
                        print(f"Stripped list item: '{i}' → '{stripped}'")
                        count += 1
                    result.append(stripped)
                else:
                    result.append(i)
            return result, count
        return x, 0

    for col in subset_cols:
        print(f"\nProcessing column: '{col}'")
        results = df[col].apply(strip_text)
        df[col] = results.apply(lambda x: x[0])
        change_counts[col] = results.apply(lambda x: x[1]).sum()

    return df, change_counts

def df_remove_diacritics(df, subset_cols):
    change_counts = {col: 0 for col in subset_cols}

    def remove_accents(x):
        if isinstance(x, str):
            cleaned = ''.join(c for c in unicodedata.normalize('NFD', x) if unicodedata.category(c) != 'Mn')
            if cleaned != x:
                print(f"Removed diacritics: '{x}' → '{cleaned}'")
            return cleaned, cleaned != x
        elif isinstance(x, list):
            result = []
            count = 0
            for i in x:
                if isinstance(i, str):
                    cleaned = ''.join(c for c in unicodedata.normalize('NFD', i) if unicodedata.category(c) != 'Mn')
                    if cleaned != i:
                        print(f"Removed diacritics in list item: '{i}' → '{cleaned}'")
                        count += 1
                    result.append(cleaned)
                else:
                    result.append(i)
            return result, count
        return x, 0

    for col in subset_cols:
        print(f"\nProcessing column: '{col}'")
        results = df[col].apply(remove_accents)
        df[col] = results.apply(lambda x: x[0])
        change_counts[col] = results.apply(lambda x: x[1]).sum()

    return df, change_counts

## STEP2: more complex normalization 

import re

def df_remove_punctuation(df, subset_cols):
    pattern = r"[.,;!?]"
    change_counts = {col: 0 for col in subset_cols}

    for col in subset_cols:
        print(f"\nProcessing column: '{col}'")

        def clean_text(x):
            if isinstance(x, str):
                cleaned = re.sub(pattern, '', x)
                return cleaned, int(cleaned != x)

            elif isinstance(x, list):
                cleaned_items = []
                count = 0
                for item in x:
                    if isinstance(item, str):
                        cleaned_item = re.sub(pattern, '', item)
                        if cleaned_item != item:
                            count += 1
                        cleaned_items.append(cleaned_item)
                    else:
                        cleaned_items.append(item)
                return cleaned_items, count  # ✅ return list, not string

            return x, 0  # non-str, non-list: unchanged

        results = df[col].apply(clean_text)
        df[col] = results.apply(lambda x: x[0])
        change_counts[col] = results.apply(lambda x: x[1]).sum()

    return df, change_counts


def df_replace_hyphen_with_space(df, subset_cols):
    change_counts = {col: 0 for col in subset_cols}

    for col in subset_cols:
        print(f"\nProcessing column: '{col}'")

        def replace_hyphen(x):
            if isinstance(x, str):
                replaced = x.replace('-', ' ')
                return replaced, replaced != x

            elif isinstance(x, list):
                new_list = []
                count = 0
                for item in x:
                    if isinstance(item, str):
                        replaced_item = item.replace('-', ' ')
                        if replaced_item != item:
                            count += 1
                        new_list.append(replaced_item)
                    else:
                        new_list.append(item)
                return new_list, count

            return x, 0

        results = df[col].apply(replace_hyphen)
        df[col] = results.apply(lambda x: x[0])
        change_counts[col] = results.apply(lambda x: x[1]).sum()

    return df, change_counts
import re

# Symbol mapping including full-width variants
symbol_mapping = {
    "&": "and",
    "＆": "and",      # Full-width ampersand
    "%": "percent",
    "％": "percent",  # Full-width percent
}

def expand_symbols(text):
    symbols_pattern = re.compile('({})'.format('|'.join(map(re.escape, symbol_mapping.keys()))), re.UNICODE)
    def replace(match):
        return symbol_mapping[match.group(0)]
    return symbols_pattern.sub(replace, text)

def df_expand_symbols(df, subset_cols):
    change_counts = {col: 0 for col in subset_cols}

    for col in subset_cols:
        print(f"\nProcessing column: '{col}'")

        def replace_symbols(x):
            if isinstance(x, str):
                replaced = expand_symbols(x)
                return replaced, replaced != x

            elif isinstance(x, list):
                new_list = []
                count = 0
                for item in x:
                    if isinstance(item, str):
                        replaced_item = expand_symbols(item)
                        if replaced_item != item:
                            count += 1
                        new_list.append(replaced_item)
                    else:
                        new_list.append(item)
                return new_list, count

            return x, 0

        results = df[col].apply(replace_symbols)
        df[col] = results.apply(lambda x: x[0])
        change_counts[col] = results.apply(lambda x: x[1]).sum()

    return df, change_counts


def df_group_by_scenario(df, scenario):
    high_precision_cols = ["factory_country", "factory_city"]
    subset_cols = ["factory_name", "owner_company_name", "product_name"]

    # Normalize: force string and lowercase, including NaN → 'nan'
    for col in high_precision_cols + subset_cols:
        df[col] = df[col].apply(lambda x: str(x).lower().strip())

    DB1 = pd.DataFrame(columns=df.columns)
    DB2 = pd.DataFrame(columns=df.columns.tolist() + ["grouped_with"])
    used_indices = set()

    def is_valid_match(val1, val2):
        # Ignore 'nan' == 'nan' — treat it as no match
        return val1 == val2 and val1 != 'nan'

    def matching_fields(row1, row2):
        high_precision = (
            is_valid_match(row1["factory_country"], row2["factory_country"]) and
            is_valid_match(row1["factory_city"], row2["factory_city"])
        )
        match_count = sum(is_valid_match(row1[col], row2[col]) for col in subset_cols)
        return match_count, high_precision

    for i, row_i in df.iterrows():
        if i in used_indices:
            continue
        group = [i]
        grouped_with = []

        for j, row_j in df.iterrows():
            if j == i or j in used_indices:
                continue
            match_count, high_precision = matching_fields(row_i, row_j)
            if high_precision:
                if (scenario == 1 and match_count >= 2) or (scenario == 2 and match_count >= 1):
                    group.append(j)
                    grouped_with.append(j)

        if len(group) > 1:
            used_indices.update(group)
            for idx in group:
                temp_row = df.loc[idx].copy()
                temp_row["grouped_with"] = [g for g in group if g != idx]
                DB2 = pd.concat([DB2, pd.DataFrame([temp_row])])
        else:
            DB1 = pd.concat([DB1, pd.DataFrame([row_i])])
            used_indices.add(i)

    return DB1.reset_index(drop=False), DB2.reset_index(drop=False)

#Step 3-----------------------------
import pandas as pd
import ast
from cleanco import basename

def df_clean_company(df, subset_cols):
    change_counts = {}

    def clean_item(original):
        if not isinstance(original, str):
            return original, False
        original_stripped = original.strip()
        cleaned = basename(original_stripped)
        changed = cleaned != original_stripped
        return cleaned, changed

    def cleanco_func(x):
        changes = 0
        if isinstance(x, str):
            # Try parsing as list-string
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    cleaned_list = []
                    for item in parsed:
                        cleaned, changed = clean_item(item)
                        changes += int(changed)
                        cleaned_list.append(cleaned)
                    return cleaned_list, changes
            except (ValueError, SyntaxError):
                # Not a list-string, treat as simple string
                cleaned, changed = clean_item(x)
                return cleaned, int(changed)
        elif isinstance(x, list):
            cleaned_list = []
            for item in x:
                cleaned, changed = clean_item(item)
                changes += int(changed)
                cleaned_list.append(cleaned)
            return cleaned_list, changes
        elif isinstance(x, (float, int)) or pd.api.types.is_scalar(x):
            if pd.isna(x):
                return x, 0
            return x, 0
        else:
            return x, 0

    for col in subset_cols:
        total_changes = 0
        cleaned_values = []
        for val in df[col]:
            cleaned_val, changes = cleanco_func(val)
            total_changes += changes
            cleaned_values.append(cleaned_val)
        df[col] = cleaned_values
        change_counts[col] = total_changes

    return df, change_counts


import re

# Define abbreviation mapping
abbreviation_mapping = {
    "PV": "photovoltaic",
    "solar PV": "solar photovoltaic",
    "CSP": "concentrated solar power",
    "Li-ion": "lithium-ion",
    "LiFePO4": "lithium iron phosphate",
    "BESS": "battery energy storage system",
    "ESS": "energy storage system",
    "EV": "electric vehicle",
    "BEV": "battery electric vehicle",
    "PHEV": "plug-in hybrid electric vehicle",
    "PHV": "plug-in hybrid vehicle",
    "ICEV": "internal combustion engine vehicle",
    "HEV": "hybrid electric vehicle",
    "GHG": "greenhouse gas",
    "CCUS": "carbon capture, utilisation and storage",
    "DAC": "direct air capture"
}

# Define expand_abbreviations BEFORE it is used
def expand_abbreviations(text):
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, abbreviation_mapping.keys())) + r')\b')
    def replace(match):
        return abbreviation_mapping[match.group(0)]
    return pattern.sub(replace, text)

# Now define df_expand_abbreviations AFTER expand_abbreviations exists
def df_expand_abbreviations(df, subset_cols):
    change_counts = {col: 0 for col in subset_cols}

    for col in subset_cols:
        print(f"\nProcessing column: '{col}'")

        def replace_abbrev(x):
            count = 0
            if isinstance(x, str):
                original = x
                replaced = expand_abbreviations(x)
                if original != replaced:
                    count = 1
                    print(f"Changed: '{original}' -> '{replaced}'")
                return replaced, count

            elif isinstance(x, list):
                new_list = []
                for item in x:
                    if isinstance(item, str):
                        replaced_item = expand_abbreviations(item)
                        if item != replaced_item:
                            count += 1
                            print(f"Changed in list: '{item}' -> '{replaced_item}'")
                        new_list.append(replaced_item)
                    else:
                        new_list.append(item)
                return new_list, count

            return x, 0

        results = df[col].apply(replace_abbrev)
        df[col] = results.apply(lambda x: x[0])
        change_counts[col] = results.apply(lambda x: x[1]).sum()

    return df, change_counts


import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

def df_lemmatize_columns_spacy(df, subset_cols):
    """
    Applies spaCy lemmatization to each string in a list of phrases.
    Converts each phrase into its lemmatized form (still as a phrase).
    Returns a new column with the same list structure but lemmatized strings.
    """
    change_counts = {col: 0 for col in subset_cols}

    def lemmatize_phrases(text):
        # Case 1: list of phrases
        if isinstance(text, list):
            lemmatized_list = []
            changed = 0
            for phrase in text:
                if isinstance(phrase, str):
                    doc = nlp(phrase)
                    lemmatized_phrase = " ".join([token.lemma_ for token in doc])
                    if lemmatized_phrase != phrase:
                        print(f"Lemmatized phrase: '{phrase}' → '{lemmatized_phrase}'")
                        changed += 1
                    lemmatized_list.append(lemmatized_phrase)
                else:
                    lemmatized_list.append(phrase)
            return lemmatized_list, changed

        # Case 2: single string
        elif isinstance(text, str):
            doc = nlp(text)
            lemmatized_phrase = " ".join([token.lemma_ for token in doc])
            if lemmatized_phrase != text:
                print(f"Lemmatized string: '{text}' → '{lemmatized_phrase}'")
                return [lemmatized_phrase], 1
            return [lemmatized_phrase], 0

        # Case 3: other types (leave unchanged)
        return text, 0

    for col in subset_cols:
        print(f"\nProcessing column: '{col}'")
        lemmatized_results = df[col].apply(lemmatize_phrases)
        df[f"{col}_lemma"] = lemmatized_results.apply(lambda x: x[0])
        change_counts[col] = lemmatized_results.apply(lambda x: x[1]).sum()

    return df, change_counts

    return df, change_counts


#Step where we create xlsx ------------


import pandas as pd
from functools import reduce
from collections import defaultdict

# === Helper Functions ===
def extract_ids(x):
    if isinstance(x, list):
        return [i.split("_")[1] if "_" in i else i for i in x]
    return []

def safe_lookup_list(ids, lookup, key):
    if not isinstance(ids, list):
        return []
    result = [lookup.get(i, {}).get(key, None) for i in ids]
    while len(result) < len(ids):
        result.append(None)
    return result[:len(ids)]

def build_reconciliation_lookup(log):
    df_log = pd.DataFrame(log)
    df_log["article_id"] = df_log["original_unique_id"].str.extract(r"(^[^_]+)")
    product_ids = df_log[df_log["entity_type"] == "product"].groupby("new_unique_id")["article_id"].apply(set).to_dict()
    company_ids = df_log[df_log["entity_type"] == "company"].groupby("new_unique_id")["article_id"].apply(set).to_dict()
    jv_ids = df_log[df_log["entity_type"] == "joint_venture"].groupby("new_unique_id")["article_id"].apply(set).to_dict()
    investment_ids = df_log[df_log["entity_type"] == "investment"].groupby("new_unique_id")["article_id"].apply(set).to_dict()
    capacity_ids = df_log[df_log["entity_type"] == "capacity"].groupby("new_unique_id")["article_id"].apply(set).to_dict()

    for uid in df_log["new_unique_id"]:
        fallback = {uid.split("_")[0]}
        if uid.startswith("product_"):
            product_ids.setdefault(uid, fallback)
        elif uid.startswith("company_"):
            company_ids.setdefault(uid, fallback)
        elif uid.startswith("investment_"):
            investment_ids.setdefault(uid, fallback)
        elif uid.startswith("capacity_"):
            capacity_ids.setdefault(uid, fallback)
        elif uid.startswith("joint_venture_"):
            jv_ids.setdefault(uid, fallback)

    return df_log, capacity_ids, product_ids, company_ids, jv_ids, investment_ids

def make_article_id_to_url(df_meta):
    df_meta["ID"] = df_meta["ID"].astype(str)
    return df_meta.set_index("ID")["url"].to_dict()

def resolve_urls(article_ids, article_id_to_url):
    if not isinstance(article_ids, (list, set)):
        return []
    return [article_id_to_url[aid] for aid in article_ids if aid in article_id_to_url]

def resolve_urls_from_uid(uids, article_id_to_url):
    if not isinstance(uids, list):
        return []
    return [article_id_to_url.get(uid[:7]) for uid in uids if isinstance(uid, str) and uid[:7] in article_id_to_url]


def deduplicate_nodes_and_rels(df_nodes, df_rels):
    return (
        df_nodes.drop_duplicates(subset="unique_id"),
        df_rels.drop_duplicates(subset=["source", "target", "type"])
    )

def extract_node_subsets(df_nodes):
    return {
        "joint_ventures": df_nodes[df_nodes["label"].str.lower() == "joint_venture"],
        "factories": df_nodes[df_nodes["label"].str.lower().str.contains("factory", na=False)].copy(),
        "capacities": df_nodes[df_nodes["label"].str.lower() == "capacity"],
        "products": df_nodes[df_nodes["label"].str.lower() == "product"],
        "companies": df_nodes[df_nodes["label"].str.lower() == "company"],
        "investment": df_nodes[df_nodes["label"].str.lower() == "investment"]
    }

def extract_relationship_subsets(df_rels):
    return {
        "owns": df_rels[df_rels["type"].str.lower() == "owns"],
        "at": df_rels[df_rels["type"].str.lower() == "at"],
        "produced_at": df_rels[df_rels["type"].str.lower() == "produced_at"],
        "funds": df_rels[df_rels["type"].str.lower() == "funds"]
    }

def group_linked_nodes(rel_df, source_nodes, source_col, target_col, entity_label_prefix):
    rel_df = rel_df.copy()
    df = rel_df.rename(columns={"source": source_col, "target": target_col})
    df = df.merge(
        source_nodes[["unique_id", "name"]].rename(columns={
            "unique_id": f"{entity_label_prefix}_unique_id",
            "name": f"{entity_label_prefix}_name"
        }),
        left_on=source_col, right_on=f"{entity_label_prefix}_unique_id", how="left"
    )
    grouped = df.groupby(target_col).agg({
        f"{entity_label_prefix}_unique_id": lambda x: list(x.dropna().unique()),
        f"{entity_label_prefix}_name": lambda x: list(x.dropna().unique())
    }).reset_index().rename(columns={target_col: "factory_unique_id"})
    return grouped



def run_factory_centric_enrichment(df_all_nodes, df_all_rels, df_meta):
    # capacity_article_ids, product_article_ids, company_article_ids, joint_venture_article_ids, investment_article_ids = build_reconciliation_lookup(main_reconciliation_log)
    article_id_to_url = make_article_id_to_url(df_meta)
    df_all_nodes, df_all_rels = deduplicate_nodes_and_rels(df_all_nodes, df_all_rels)
    nodes = extract_node_subsets(df_all_nodes)
    rels = extract_relationship_subsets(df_all_rels)

    # Extract city and country from already flattened factory nodes
    df_factory_locations = nodes["factories"][["name", "unique_id", "location_city", "location_country"]].rename(
        columns={
            "unique_id": "factory_unique_id",
            "location_city": "factory_city",
            "location_country": "factory_country",
            "name":"factory_name"
        }
    )


    df_owns_comp = group_linked_nodes(rels["owns"], nodes["companies"], "source", "target", "owner_company")
    df_owns_jv = group_linked_nodes(rels["owns"], nodes["joint_ventures"], "source", "target", "owner_jv")
    df_funds = group_linked_nodes(rels["funds"], nodes["investment"], "source", "target", "investment")
    df_products = group_linked_nodes(rels["produced_at"], nodes["products"], "source", "target", "product")
    df_capacities = group_linked_nodes(rels["at"], nodes["capacities"], "source", "target", "capacity")

    df_master = reduce(lambda left, right: pd.merge(left, right, on="factory_unique_id", how="outer"), [
        df_owns_comp, df_owns_jv, df_funds, df_products, df_capacities
    ])
    # Merge city and country info
    df_master = df_master.merge(df_factory_locations, on="factory_unique_id", how="left")


    inv_lookup = nodes["investment"].set_index("unique_id")[["name", "status", "amount", "phase"]].to_dict("index")
    cap_lookup = nodes["capacities"].set_index("unique_id")[["name", "status", "amount", "phase"]].to_dict("index")

    df_master["investment_name"] = df_master["investment_unique_id"].apply(lambda uids: safe_lookup_list(uids, inv_lookup, "name"))
    df_master["investment_status"] = df_master["investment_unique_id"].apply(lambda uids: safe_lookup_list(uids, inv_lookup, "status"))
    df_master["investment_amount"] = df_master["investment_unique_id"].apply(lambda uids: safe_lookup_list(uids, inv_lookup, "amount"))
    df_master["investment_phase"] = df_master["investment_unique_id"].apply(lambda uids: safe_lookup_list(uids, inv_lookup, "phase"))

    df_master["capacity_name"] = df_master["capacity_unique_id"].apply(lambda ids: safe_lookup_list(ids, cap_lookup, "name"))
    df_master["capacity_status"] = df_master["capacity_unique_id"].apply(lambda ids: safe_lookup_list(ids, cap_lookup, "status"))
    df_master["capacity_amount"] = df_master["capacity_unique_id"].apply(lambda ids: safe_lookup_list(ids, cap_lookup, "amount"))
    df_master["capacity_phase"] = df_master["capacity_unique_id"].apply(lambda ids: safe_lookup_list(ids, cap_lookup, "phase"))


    df_master["factory_urls"] = df_master["factory_unique_id"].apply(lambda uids: resolve_urls_from_uid(uids, article_id_to_url))
    df_master["product_urls"] = df_master["product_unique_id"].apply(lambda uids: resolve_urls_from_uid(uids, article_id_to_url))
    df_master["capacity_urls"] = df_master["capacity_unique_id"].apply(lambda uids: resolve_urls_from_uid(uids, article_id_to_url))
    df_master["owner_company_urls"] = df_master["owner_company_unique_id"].apply(lambda uids: resolve_urls_from_uid(uids, article_id_to_url))
    df_master["owner_jv_urls"] = df_master["owner_jv_unique_id"].apply(lambda uids: resolve_urls_from_uid(uids, article_id_to_url))
    df_master["investment_urls"] = df_master["investment_unique_id"].apply(lambda uids: resolve_urls_from_uid(uids, article_id_to_url))

    # df_master["factory_article_ids"] = df_master["factory_unique_id"].apply(extract_ids)
    # df_master["factory_urls"] = df_master["factory_article_ids"].apply(lambda ids: resolve_urls(ids, article_id_to_url))

    # df_master["product_article_ids"] = df_master["product_unique_id"].apply(extract_ids)
    # df_master["product_urls"] = df_master["product_article_ids"].apply(lambda ids: resolve_urls(ids, article_id_to_url))

    # df_master["capacity_article_ids"] = df_master["capacity_unique_id"].apply(extract_ids)
    # df_master["capacity_urls"] = df_master["capacity_article_ids"].apply(lambda ids: resolve_urls(ids, article_id_to_url))

    # df_master["company_article_ids"] = df_master["owner_company_unique_id"].apply(
    #     lambda uids: [aid for uid in uids for aid in company_article_ids.get(uid, [])] if isinstance(uids, list) else [])
    # df_master["company_urls"] = df_master["company_article_ids"].apply(lambda ids: resolve_urls(ids, article_id_to_url))

    # df_master["joint_venture_article_ids"] = df_master["owner_jv_unique_id"].apply(
    #     lambda uids: [aid for uid in uids for aid in joint_venture_article_ids.get(uid, [])] if isinstance(uids, list) else [])
    # df_master["joint_venture_urls"] = df_master["joint_venture_article_ids"].apply(lambda ids: resolve_urls(ids, article_id_to_url))

    df_master_final = df_master[[
        "factory_name", "factory_country", "factory_city", "factory_urls",
        "owner_company_name", 
        "owner_jv_name", 
        "product_name", 
        "capacity_name", "capacity_status",  "capacity_phase", "capacity_amount",
        "investment_name", "investment_status", "investment_phase", "investment_amount", 
    ]]


    # df_canonical = pd.DataFrame([{"entity_type": k, "unique_id": v} for k, vs in canonical_entities.items() for v in vs])

    df_factories_pivot = df_master.explode(["factory_unique_id"])
    df_owner_companies_pivot = df_master.explode("owner_company_name")
    df_owner_jvs_pivot = df_master.explode("owner_jv_name")
    df_products_pivot = df_master.explode(["product_name"])
    df_capacities_pivot = df_master.explode(["capacity_name", "capacity_status", "capacity_amount", "capacity_phase"])
    df_investments_pivot = df_master.explode(["investment_name", "investment_status", "investment_amount", "investment_phase"])

    with pd.ExcelWriter("reconciliation_outputs_factory.xlsx", engine="openpyxl") as writer:
        df_master.to_excel(writer, sheet_name="factory", index=False)
        df_master_final.to_excel(writer, sheet_name="summary_view_factory", index=False)
        df_factories_pivot.to_excel(writer, sheet_name="pivot_factories", index=False)
        df_owner_companies_pivot.to_excel(writer, sheet_name="pivot_owner_companies", index=False)
        df_owner_jvs_pivot.to_excel(writer, sheet_name="pivot_owner_jvs", index=False)
        df_products_pivot.to_excel(writer, sheet_name="pivot_products", index=False)
        df_capacities_pivot.to_excel(writer, sheet_name="pivot_capacities", index=False)
        df_investments_pivot.to_excel(writer, sheet_name="pivot_investments", index=False)
        # df_reconcile_log.to_excel(writer, sheet_name="reconciliation_log_factory", index=False)
        # df_canonical.to_excel(writer, sheet_name="canonical_entities_factory", index=False)

    print("\u2705 Saved factory-centric outputs to reconciliation_outputs_factory.xlsx")
    return df_master_final




