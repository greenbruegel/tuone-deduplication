# Tuone Deduplication

A Python-based tool for standardizing and geocoding location data, with a focus on deduplication and GeoNames integration.

## Overview

This tool helps standardize and validate location data by:
- Standardizing country names to their official names and ISO-2 codes
- Geocoding cities using GeoNames API
- Handling administrative divisions and populated places
- Implementing rate limiting for API calls
- Providing detailed logging and error handling

## Features

- Country name standardization using `pycountry`
- City geocoding with GeoNames API
- Support for administrative divisions (ADM1-5)
- Rate limiting to respect GeoNames API quotas
- Detailed logging of the standardization process
- Support for manual country aliases (e.g., UK, England, Scotland)
- Caching of results using `lru_cache`


## Configuration

The tool uses the following configuration parameters:
- `GEONAMES_USERNAME`: Your GeoNames API username
- `TOKENS_PER_HOUR`: Rate limit for GeoNames API (default: 1000)
- `RATE_LIMIT_SECONDS`: Delay between requests (default: 4)
- `REQUEST_TIMEOUT`: API request timeout in seconds (default: 10)
