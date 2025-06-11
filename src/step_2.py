"""
step_2.py  – Country / city standardisation and GeoNames look-ups
-----------------------------------------------------------------
* Removes duplicate definitions that were overriding each other
* Passes featureClass as a LIST (["A", "P"]) so GeoNames receives
  “…&featureClass=A&featureClass=P” instead of a single tuple value
* Adds a small request timeout and a retry delay constant
"""

import logging
import time
from functools import lru_cache

import pycountry
import requests


# ──────────────────────────────── Configuration ────────────────────────────────
GEONAMES_USERNAME   = "chiarastrama"
VALID_FEATURE_CODES = {"PPL", "PPLA", "PPLA2", "PPLC", "PPLG"}      # populated-place codes
FEATURE_CLASSES     = ["A", "P"]                                    # admin + populated
RATE_LIMIT_SECONDS  = 4                                             # polite delay
REQUEST_TIMEOUT     = 10                                            # seconds

# Manual aliases for non-sovereign or colloquial country names
COUNTRY_ALIASES = {
    "UK":               ("United Kingdom", "GB"),
    "ENGLAND":          ("United Kingdom", "GB"),
    "SCOTLAND":         ("United Kingdom", "GB"),
    "WALES":            ("United Kingdom", "GB"),
    "NORTHERN IRELAND": ("United Kingdom", "GB"),
    "TURKEY":           ("Turkey",          "TR"),
}

# ────────────────────────────── Helper functions ───────────────────────────────
@lru_cache(maxsize=None)
def standardize_country(country_name: str):
    """
    Return (official_name, ISO-2, failed_flag) for *country_name*.

    * Tries manual aliases first
    * Then pycountry.lookup (exact)
    * Then pycountry.search_fuzzy (approximate)
    """
    if not country_name or str(country_name).strip().lower() in {"nan", ""}:
        return None, None, True

    key = country_name.strip().upper()
    if key in COUNTRY_ALIASES:
        std_name, iso2 = COUNTRY_ALIASES[key]
        return std_name, iso2, False

    try:
        country = pycountry.countries.lookup(country_name.strip())
        return country.name, country.alpha_2, False
    except LookupError:
        try:
            country = pycountry.countries.search_fuzzy(country_name.strip())[0]
            return country.name, country.alpha_2, False
        except Exception:
            logging.warning(f"[standardize_country] country lookup failed for “{country_name}”")
            return None, None, True


@lru_cache(maxsize=None)
def get_adm_level(city: str, iso2: str | None, level: int,
                  max_retries: int = 5, backoff: int = 60):
    """
    Query GeoNames for *city* (ADM{level} first, then PPL).

    Automatically retries (with exponential back-off) when the server says
    “hourly limit” (value 10) or “daily limit” (value 11).

    Returns (adm_name, adm_code, lat, lon, failed_flag).
    """
    attempt = 0
    admin_code = f"ADM{level}"

    while attempt <= max_retries:
        attempt += 1
        params = {
            "q": city,
            "maxRows": 10,
            "style": "FULL",
            "featureClass": ["A", "P"],
            "username": GEONAMES_USERNAME,
        }
        if iso2:
            params["country"] = iso2

        try:
            resp = requests.get(
                "http://api.geonames.org/searchJSON",
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            js = resp.json()
        except Exception as exc:
            logging.error(f"[get_adm_level] request error “{city}” → {exc}")
            return None, None, None, None, True

        # ── handle GeoNames error payloads ────────────────────────────────
        if "status" in js:                               # request was *rejected*
            code = int(js["status"]["value"])
            msg  = js["status"]["message"]
            logging.warning(f"[GeoNames {code}] {msg}")

            # 10 = hourly limit, 11 = daily limit → back-off then retry
            if code in {10, 11} and attempt <= max_retries:
                wait = backoff * attempt                # 60, 120, 180, …
                logging.info(f"   retrying in {wait} s (attempt {attempt}/{max_retries})")
                time.sleep(wait)
                continue

            # any other status code is fatal for this lookup
            return None, None, None, None, True

        data = js.get("geonames", [])
        if not data:                                     # no hits
            return None, None, None, None, True

        # ── choose best candidate ─────────────────────────────────────────
        target_admin = target_ppl = None
        for rec in data:
            fcl, fcode = rec.get("fcl"), rec.get("fcode")
            if fcl == "A" and fcode == admin_code:
                target_admin = rec
                break
            if not target_ppl and fcl == "P" and fcode in VALID_FEATURE_CODES:
                target_ppl = rec

        chosen = target_admin or target_ppl
        if not chosen:
            return None, None, None, None, True

        name = chosen.get(f"adminName{level}") or chosen.get("name")
        code = chosen.get(f"adminCode{level}")
        lat  = chosen.get("lat")
        lon  = chosen.get("lng")
        return name, code, lat, lon, False

    # exhausted retries
    logging.error(f"[get_adm_level] giving up on “{city}” after {max_retries} retries")
    return None, None, None, None, True


# ────────────────────────────── Main API function ──────────────────────────────
def standardise_country_city(
    df,
    city_col: str,
    country_col: str | None = None,
    *,
    details: bool = False,
    verbose: bool = False,
):
    """
    Given a DataFrame containing *city_col* and *country_col*, return the same
    DataFrame augmented with standardised country data and GeoNames admin info.

    If *details* is True, a list of per-row dicts is returned instead of the df.
    If *verbose* is True, each lookup is logged as it happens.
    """
    df = df.copy().reset_index(drop=True)

    # pre-create output columns
    out_cols = {
        "country_standardized":    None,
        "country_iso2":            None,
        "country_not_found":       False,
        "factory_city_adm_name":   None,
        "factory_city_adm_code":   None,
        "factory_city_adm_level":  None,
        "factory_city_latitude":   None,
        "factory_city_longitude":  None,
        "factory_city_not_found":  False,
    }
    for col, default in out_cols.items():
        df[col] = default

    detail_records = []
    total = len(df)

    for idx, row in df.iterrows():
        raw_country = row[country_col] if country_col else None
        city_raw    = row[city_col]

        # handle list-type city cells that may have been exploded
        if isinstance(city_raw, list) and city_raw:
            city_raw = city_raw[0]
        city = str(city_raw).strip()

        # ── Country standardisation ──────────────────────────────────────────
        std_name, iso2, c_failed = standardize_country(str(raw_country))
        df.at[idx, "country_standardized"] = std_name
        df.at[idx, "country_iso2"]         = iso2
        df.at[idx, "country_not_found"]    = c_failed

        # record for optional verbose logging / details
        record = {
            "country_raw":          raw_country,
            "country_standardized": std_name,
            "country_iso2":         iso2,
            "city":                 city,
            "adm_found":            False,
            "adm_level":            None,
            "adm_name":             None,
            "latitude":             None,
            "longitude":            None,
        }

        # ── City look-up ────────────────────────────────────────────────────
        if not city or city.lower() == "nan" or c_failed:
            df.at[idx, "factory_city_not_found"] = True
        else:
            for level in range(1, 6):
                name, code, lat, lon, failed = get_adm_level(city, iso2, level)
                if not failed and name:
                    df.at[idx, "factory_city_adm_name"]   = name
                    df.at[idx, "factory_city_adm_code"]   = code
                    df.at[idx, "factory_city_adm_level"]  = level
                    df.at[idx, "factory_city_latitude"]   = lat
                    df.at[idx, "factory_city_longitude"]  = lon
                    record.update(
                        adm_found=True,
                        adm_level=level,
                        adm_name=name,
                        latitude=lat,
                        longitude=lon,
                    )
                    break
            else:
                df.at[idx, "factory_city_not_found"] = True

        if verbose:
            logging.info(f"[{idx + 1}/{total}] {record}")

        detail_records.append(record)
        time.sleep(RATE_LIMIT_SECONDS)     # honour GeoNames’ fair-use policy

    return detail_records if details else df
