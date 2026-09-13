## Purpose

Ingests, aligns, and normalizes observations from PeakWeather (302 Swiss stations) and Swiss IMIS (180 high-alpine stations) into a unified training dataset.

## ADDED Requirements

### Requirement: PeakWeather ingestion from HuggingFace

The system SHALL download and parse the MeteoSwiss/PeakWeather dataset from HuggingFace. The dataset contains 302 Swiss weather stations with 10-minute temporal resolution. The system SHALL extract: 2m temperature, wind speed, wind direction, precipitation, surface pressure, relative humidity, and global radiation.

#### Scenario: Successful PeakWeather download

- **WHEN** the pipeline is run with a HuggingFace token and date range
- **THEN** the system downloads and parses PeakWeather data for all 302 stations within the date range

#### Scenario: Incremental update

- **WHEN** the pipeline is run and local PeakWeather data already exists for part of the date range
- **THEN** the system downloads only the missing date range and appends to the existing dataset

### Requirement: Swiss IMIS ingestion

The system SHALL ingest Swiss IMIS data from the SLF open API and EnviDat archive. IMIS provides 180 high-alpine stations at 30-minute temporal resolution. Variables: wind speed, wind direction, air temperature, snow surface temperature, snow depth, relative humidity.

#### Scenario: Successful IMIS API fetch

- **WHEN** the pipeline is run with a date range
- **THEN** the system fetches IMIS observations for all 180 stations via the SLF API, falling back to EnviDat bulk files for historical data outside the API's temporal window

### Requirement: Station metadata schema

Each station record SHALL include the following metadata fields:
- Station ID (string, unique within network)
- Network identifier (`peakweather` or `imis`)
- Latitude, Longitude (WGS84 decimal degrees)
- Elevation (meters above sea level)
- Terrain features (from the terrain feature encoder, linked by station coordinates)

The metadata schema SHALL be consistent across both networks.

#### Scenario: Station metadata alignment

- **WHEN** data from both networks is loaded
- **THEN** every station record contains all metadata fields, with terrain features computed from the DEM at the station's coordinates

### Requirement: Temporal alignment

PeakWeather data (10-min) and IMIS data (30-min) SHALL be aligned to a common temporal resolution. The system SHALL support configurable target resolution (default: 30 minutes). PeakWeather 10-min data SHALL be aggregated to the target resolution using: mean for temperature/pressure/humidity, vector-mean for wind, sum for precipitation.

#### Scenario: Align 10-min to 30-min

- **WHEN** PeakWeather 10-min data and IMIS 30-min data are loaded with target resolution 30 minutes
- **THEN** PeakWeather records are aggregated to 30-min intervals and timestamps align exactly with IMIS records

#### Scenario: Incomplete intervals

- **WHEN** a 30-min aggregation window has fewer than 2 of 3 expected 10-min records
- **THEN** that interval is marked as missing (NaN) rather than computed from insufficient data

### Requirement: Normalization for interoperability

The system SHALL normalize all variables to common units across both networks. Temperature in Kelvin, wind speed in m/s, wind direction in degrees (meteorological convention, 0=N), precipitation in mm/interval, pressure in hPa, humidity in fraction (0-1). The system SHALL apply quality control flags: reject physically impossible values (e.g., negative wind speed, temperature below 180K or above 340K, wind direction outside 0-360).

#### Scenario: Cross-network variable consistency

- **WHEN** a training batch is constructed mixing PeakWeather and IMIS stations
- **THEN** all variables use identical units, naming conventions, and quality control thresholds

### Requirement: Output format

The aligned dataset SHALL be stored as a collection of per-station time series in a format supporting efficient random access by station and time range. Each record contains: timestamp, all weather variables, station metadata reference, and a quality flag per variable.

#### Scenario: Random access by station and time

- **WHEN** a training dataloader requests data for station X between times T1 and T2
- **THEN** the system returns the time series slice in O(log N) time without scanning the full dataset

**Deferred data sources note:** SAFRAN S2M (mentioned in proposal as a potential Alpine data source) is deferred — priority 6, not in Tier 1-2 scope. The PeakWeather and IMIS networks provide sufficient Alpine station coverage for pre-training. SAFRAN S2M can be evaluated for inclusion in a future iteration if additional gridded Alpine training data proves necessary.
