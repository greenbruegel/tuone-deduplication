# Deduplication

## Overview

**`tuone-normalisation`** performs **multi-stage grouping of factory records** by combining progressive string normalization, list-based fuzzy matching, and geographic enrichment through administrative regions.

Each stage applies **increasingly refined normalisation** to a core set of entity fields:

- `owner_company_name`
- `factory_name`
- `factory_city`
- `factory_country`
- `product_name`

After normalisation, each step performs **pairwise comparison of candidate record pairs** using **blocking on `factory_country`**, i.e. meaning that only records from the same country are considered as potential matches, which drastically reduces the number of comparisons.

The **matching logic evolves across stages**:

- Early stages rely on exact and fuzzy string similarity (e.g., name, city).
- Later stages introduce:
    - **lemmatization** for semantic alignment
    - **geographic disambiguation** using ADM2-level administrative boundaries via the GeoNames API

Once matches are identified, records are grouped using a **graph-based approach**: each matched pair is treated as an edge, and **connected components** in the resulting graph define deduplicated groups.

The pipeline outputs:

- Cleaned and annotated data at each stage
- Group IDs and duplication flags
- A full change log tracking the effects of each transformation step

## Node normalisation

### Normalisation functions

| Step | Normalisation function | Input | Output | Normalisation |
| --- | --- | --- | --- | --- |
| 1 | lower | • `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` | •`owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` |   → Lowercase transformation with support for strings and lists of strings.
 |
| 1 | diacritics | • `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` |  `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` | → Removes diacritic marks (accents like `é`, `ü`, `ç`) from strings using Unicode decomposition (NFD).

 |
| 1 | nfkd | • `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` | • `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` | → Separates characters and diacritics with support for strings and lists of strings.
 |
| 1 | strip | • `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` | • `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` | → Removes leading and trailing whitespace from strings or string elements inside lists. |
| 2 | cleanCompany | • `owner_company_name`
 | •`owner_company_name` | → Cleans company names by removing legal suffices like “Ltd”, “Inc”, “S.A.”, etc., using the cleanco package’s basename() function. |
| 2 | ponctuation | • `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` | `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` | → Removes selected punctuation characters (.,:?) from strings or lists of strings. |
| 2 | hyphen | • `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` | • `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` | → Replaces hyphens with spaces in strings or list elements. |
| 2 | symbol | • `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` | • `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` | → Expands symbols to words using a custom dictionary (& → and, % → percent), including full-width Unicode variants. |
| 3 | productAbbrevation | • `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` |  `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` | → Expands technical or industry-specific abbreviations (e.g., EV → electric vehicle) into their full descriptive forms to improve semantic consistency across records.  |
| 3 | lemma | • `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name` | • `owner_company_name`
• `factory_name`
• `factory_city`
• `factory_country`
• `product_name_lemma` | → Applies lemmatization to phrases using SpaCy, i.e. reduces words to their base from for more consistent matching and comparison.  |
| 4 | geoname | 
• `factory_city`
• `factory_country`
 | 
• `factory_city_adm2name`
• `factory_city_adm2code`
 | → Queries GeoNames API with a city name, optionally filtered by country. → Coverts country names to ISO2 codes → Retrieves ADM2 name and code → Populates places only (e.g. towns, capitals, municipalities) → Flags cities that are not recognized or are not valid populated places → Sleeps 1 second between requests to comply with GeoNames’usage policy.  |
|  |  |  |  |  |

### Log changes i.e. track the number of modified elements for each variable.

| Normalisation | factory_name | factory_country | factory_city | owner_company_name | product_name |
| --- | --- | --- | --- | --- | --- |
| lower | 11 058 | 10 894 | 9 416 | 10 816 | 4 003 |
| diacritics | 551 | 1 | 722 | 135 | 45 |
| nfkd | 0 | 0 | 0 | 0 | 0 |
| strip | 1 | 0 | 3 | 0 | 0 |
| cleanCompany |  |  |  | 400 |  |
| ponctuation | 57 | 8 | 46 | 50 | 516 |
| hyphen | 374 | 0 | 204 | 396 | 1 745 |
| symbol | 66 | 0 | 0 | 38 | 0 |
| productAbbrevation |  |  |  |  | 0 |
| lemma |  |  |  |  | 4 394 |

## Record matches

Step  1

| Variable | Match Type | Threshold / Condition | Notes |
| --- | --- | --- | --- |
| factory_country | Exact | == 1 | Used for blocking and filtering |
| factory_city | Exact | == 1 | Required for match |
| factory_name | String similarity | N/A | Basic string comparison |
| product_name | List similarity | Jaro-Winkler ≥ 0.80 | Uses CompareAnyStringListMatch |
| owner_company_name | List similarity | Jaro-Winkler ≥ 0.80 | Uses CompareAnyStringListMatch |
| Match logic | Combined | country AND city == 1, and (name OR product OR owner) == 1 | Used to form match pairs and group via connected components |

Step 2

| Variable | Match Type | Threshold / Condition | Notes |
| --- | --- | --- | --- |
| factory_country | Exact | == 1 | Same as Step 1 |
| factory_city | Exact | == 1 | Same |
| factory_name | String similarity | N/A | After punctuation, hyphen, and symbol normalization |
| product_name | List similarity | Jaro-Winkler ≥ 0.80 | Cleaned lists |
| owner_company_name | List similarity | Jaro-Winkler ≥ 0.80 | Post-cleaning (e.g., Ltd., GmbH removal) |
| Match logic | Combined | country AND city == 1, and (name OR product OR owner) == 1 | Same logic as Step 1 |

Step 3

| Variable | Match Type | Threshold / Condition | Notes |
| --- | --- | --- | --- |
| factory_country | Exact | == 1 | Same as previous |
| factory_city | Exact | == 1 | Required match |
| factory_name | String similarity | N/A | Same |
| product_name_lemma | List similarity | Jaro-Winkler ≥ 0.80 | After expanding abbreviations and applying lemmatization |
| owner_company_name | List similarity | Jaro-Winkler ≥ 0.80 | Same as Step 2 |
| Match logic | Combined | country AND city == 1, and (name OR product OR owner) == 1 | Improved semantic alignment |

Step 4

| Variable | Match Type | Threshold / Condition | Notes |
| --- | --- | --- | --- |
| factory_country | Exact | == 1 | Blocking key |
| factory_city | Exact | == 1 (OR with ADM2) | One of the two (city or ADM2) must match |
| factory_city_adm2name | Exact | == 1 (OR with city) | Populated via GeoNames ADM2 lookup |
| factory_name | String similarity | N/A | Same as previous |
| product_name_lemma | List similarity | Jaro-Winkler ≥ 0.80 | Same as Step 3 |
| owner_company_name | List similarity | Jaro-Winkler ≥ 0.80 | Same |
| Match logic | Combined | country == 1 AND (city OR adm2 == 1) AND (name OR product OR owner) == 1 | Adds geographic context to improve match robustness |

# Clustering

- Construct an **undirected graph** `G`.
- Each **node** = one record (identified by its DataFrame index).
- Each **edge** = a match between two records based on your matching logic
- `nx.connected_components(G)` finds **groups of connected nodes**.
- These are interpreted as **clusters of duplicates**.
- Each connected component gets a unique `group_id`.
    - If A matches B, and B matches C, then A, B, and C will all share a `group_id`.

Issue of **transitive chaining**: Naturally models transitive matches: A ≈ B and B ≈ C → A ≈ C .

Idea: 

- Assign a **unique group ID per match pair**.
- Ensure each record can belong to **multiple groups**, or track which pairs belong together without merging them transitively.
