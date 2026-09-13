## Purpose

ERA5 reanalysis input conditioning for Alpine and global pre-training, enabling the cross-domain transfer path that HRRR's US-only coverage cannot support.

## ADDED Requirements

### Requirement: ERA5 data access

The system SHALL download ERA5 reanalysis data from the Copernicus Climate Data Store (CDS) API for a configurable spatial domain and time range. The system SHALL support both the legacy CDS API and the new CDS-Beta API.

#### Scenario: Download Alpine domain data

- **WHEN** a request specifies the Alpine domain (lat 45-48, lon 5-11) and a date range
- **THEN** the system downloads the 6 required surface variables from ERA5 for that domain and period

#### Scenario: CDS API authentication failure

- **WHEN** the CDS API key is missing or invalid
- **THEN** the system raises a clear error with instructions for obtaining and configuring a CDS API key

#### Scenario: Resumable download

- **WHEN** a download is interrupted
- **THEN** the system resumes from where it left off on the next invocation, skipping already-downloaded timesteps

### Requirement: Variable extraction and mapping

The system SHALL extract the 6 surface variables from ERA5 data and map them to the project's canonical variable schema, matching the NWP passthrough adapter's output format.

ERA5 variables:
- `2t` → T2M (2m temperature, K)
- `10u` → U10 (10m u-wind, m/s)
- `10v` → V10 (10m v-wind, m/s)
- `tp` → PRATE (total precipitation, accumulated → rate conversion required)
- `sp` → SP (surface pressure, Pa)
- `blh` → BLH (boundary layer height, m)

#### Scenario: Variable mapping matches HRRR schema

- **WHEN** ERA5 data is processed through the conditioner
- **THEN** the output variable ordering and units match what the NWP passthrough produces from HRRR data, ensuring the downscaling head receives identically structured input regardless of source

### Requirement: Grid extraction for target domains

The system SHALL extract spatial subsets of ERA5 data for the Alpine domain (Phase 2) and Colorado domain (Phase 3 validation). Grid coordinates SHALL use WGS84 latitude/longitude.

#### Scenario: Alpine domain extraction

- **WHEN** Phase 2 training requests ERA5 data for a Swiss station location
- **THEN** the conditioner extracts the surrounding ERA5 grid cells and returns a tensor covering the local domain

#### Scenario: Colorado domain extraction

- **WHEN** Phase 3 requests ERA5 data for a SNOTEL station
- **THEN** the conditioner extracts the equivalent Colorado subdomain from ERA5

### Requirement: Temporal alignment with station observations

The system SHALL align ERA5 timestamps (hourly) with station observation timestamps from PeakWeather (10-min) and IMIS (30-min). Alignment SHALL select the nearest ERA5 timestep within a configurable tolerance (default: 30 minutes).

#### Scenario: Station-ERA5 temporal pairing

- **WHEN** a station observation at time T is paired with ERA5 data
- **THEN** the system selects the ERA5 timestep nearest to T, within the configured tolerance, and raises an error if no ERA5 timestep falls within tolerance
