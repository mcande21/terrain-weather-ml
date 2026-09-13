## Purpose

Specifies the subsetting and ingestion strategy for the Reynolds Creek 21TB hourly gridded product, scoping the download to what the training and evaluation pipelines actually need.

## ADDED Requirements

### Requirement: Variable subsetting

The system SHALL download only the following variables from the Reynolds Creek gridded product:

- Air temperature (2m)
- Wind speed (10m)
- Wind direction (10m)
- Precipitation (liquid equivalent)
- Relative humidity
- Incoming shortwave radiation

All other variables (snow depth, soil temperature, etc.) SHALL be excluded from the initial download to minimize storage.

#### Scenario: Variable selection at download

- **WHEN** the Reynolds Creek downloader is configured
- **THEN** it requests only the 6 specified variables, rejecting files or records that contain only excluded variables

### Requirement: Spatial extent

The system SHALL download data covering the full Reynolds Creek Experimental Watershed spatial domain (approximately 239 km^2 in southwestern Idaho). The system SHALL NOT subset spatially — the full gridded domain at 10m resolution is needed for evaluating the downscaling head against a dense gridded truth.

#### Scenario: Full spatial coverage

- **WHEN** data download completes
- **THEN** the downloaded grid covers the entire Reynolds Creek watershed boundary with no spatial gaps

### Requirement: Temporal window

The system SHALL support configurable temporal subsetting. The default temporal window SHALL be the most recent 5 water years (October 1 through September 30). The full 40-year record SHALL be downloadable by setting the temporal window to the full available range.

#### Scenario: Default 5-year download

- **WHEN** the downloader runs with default settings
- **THEN** it downloads data for the 5 most recent complete water years only

#### Scenario: Custom temporal window

- **WHEN** a start and end date are specified
- **THEN** the downloader retrieves data only within that range

### Requirement: Staged download with checkpointing

Given the dataset size (21TB full, estimated 2-3TB for 5-year subset), the system SHALL implement staged downloading with checkpoint files. The download SHALL be resumable — interruption and restart SHALL not require re-downloading completed files.

#### Scenario: Interrupted download

- **WHEN** a download is interrupted and restarted
- **THEN** the system resumes from the last completed file, verified by file hash comparison

### Requirement: Output format

Downloaded Reynolds Creek data SHALL be converted to the same per-variable NetCDF format used by the training pipeline. Each file SHALL contain one variable for one day, with hourly time steps and the full spatial grid.

#### Scenario: File organization

- **WHEN** download and conversion complete
- **THEN** the output directory contains NetCDF files organized as `{variable}/{year}/{variable}_{YYYYMMDD}.nc`, each with hourly time steps and spatial dimensions matching the 10m grid
