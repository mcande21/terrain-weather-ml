# inference/micro-weather-api Specification

## Purpose
Serves sub-km weather predictions at arbitrary terrain coordinates via a REST API, providing the input layer for colorado-avalanche-ml Phase 2/3 avalanche instability modeling.

## Requirements

### Requirement: Prediction endpoint

The system SHALL expose a FastAPI endpoint that accepts a query with (latitude, longitude, datetime) and returns a sub-km weather prediction at that location. The endpoint path SHALL be `POST /predict`.

**Request body:**
- `latitude` (float, WGS84 decimal degrees)
- `longitude` (float, WGS84 decimal degrees)
- `datetime` (ISO 8601 string, UTC)

**Response body:**
- `wind_speed` (float, m/s)
- `wind_direction` (float, degrees, meteorological convention)
- `temperature` (float, Kelvin)
- `precipitation_rate` (float, mm/hr)
- `elevation` (float, meters — from DEM at query point)
- `aspect` (float, degrees — from DEM at query point)
- `model_version` (string — checkpoint identifier)
- `source_hrrr_cycle` (string — HRRR analysis/forecast cycle used)

#### Scenario: Single point prediction

- **WHEN** a valid (lat, lon, datetime) query is submitted
- **THEN** the system returns all weather variables at that location with sub-km terrain resolution

#### Scenario: Location outside model domain

- **WHEN** the query coordinates fall outside the loaded model's spatial domain
- **THEN** the system returns HTTP 422 with a message specifying the valid domain bounds

#### Scenario: Future datetime beyond HRRR forecast horizon

- **WHEN** the query datetime is more than 18 hours in the future
- **THEN** the system returns HTTP 422 indicating the maximum forecast lead time

### Requirement: Batch prediction endpoint

The system SHALL expose a batch endpoint at `POST /predict/batch` that accepts up to 1000 (latitude, longitude, datetime) tuples in a single request. The response SHALL contain one prediction per input tuple in the same order.

#### Scenario: Batch of 100 points

- **WHEN** a batch request with 100 coordinate-time tuples is submitted
- **THEN** the system returns 100 predictions in order, processing them efficiently (shared StormCast forward pass for same-time queries)

#### Scenario: Batch size exceeded

- **WHEN** a batch request with more than 1000 tuples is submitted
- **THEN** the system returns HTTP 422 indicating the maximum batch size

### Requirement: Interface contract with colorado-avalanche-ml

The prediction output SHALL conform to the contract expected by colorado-avalanche-ml Phase 2/3. The downstream system consumes hourly weather at (latitude, longitude, elevation, aspect) as input features for the avalanche instability model. The API response SHALL include elevation and aspect (from the terrain encoder) alongside weather variables so the downstream consumer receives a complete feature vector without a separate DEM query.

#### Scenario: Avalanche model integration

- **WHEN** colorado-avalanche-ml queries the API for a set of terrain points at hourly intervals over a 24-hour forecast window
- **THEN** each response contains wind speed, wind direction, temperature, precipitation rate, elevation, and aspect — the complete input feature set for the avalanche instability model

### Requirement: Health and metadata endpoints

The system SHALL expose:
- `GET /health` — returns 200 when the model is loaded and ready to serve
- `GET /metadata` — returns model version, checkpoint hash, spatial domain bounds, available forecast horizon, and the list of output variables with units

#### Scenario: Health check

- **WHEN** the model is loaded and inference is functional
- **THEN** `GET /health` returns HTTP 200 with `{"status": "healthy"}`

#### Scenario: Model not loaded

- **WHEN** the model has not finished loading or failed to load
- **THEN** `GET /health` returns HTTP 503 with a message indicating the loading state

### Requirement: HRRR input refresh

The system SHALL automatically fetch the latest available HRRR analysis cycle from AWS S3. The refresh interval SHALL be configurable (default: hourly). The system SHALL log which HRRR cycle is currently active and include it in prediction responses.

#### Scenario: HRRR cycle update

- **WHEN** a new HRRR analysis cycle becomes available on S3
- **THEN** the system fetches and processes it within the configured refresh interval, and subsequent predictions use the updated input

#### Scenario: HRRR fetch failure

- **WHEN** the HRRR fetch fails (network error, missing data on S3)
- **THEN** the system continues serving predictions from the most recent successful HRRR cycle and logs a warning with the failed cycle identifier

### Requirement: Default backbone selection

The inference service SHALL use the NWP passthrough adapter as its default backbone, reading raw HRRR data for coarse weather input. StormCast SHALL be available as an optional experimental backbone selectable via configuration.

#### Scenario: Default inference uses HRRR passthrough

- **WHEN** the inference service starts with default configuration
- **THEN** it initializes the NWP passthrough adapter and serves predictions from raw HRRR data without running neural backbone inference

#### Scenario: Experimental StormCast backbone

- **WHEN** the configuration specifies `backbone: stormcast`
- **THEN** the inference service loads the StormCast adapter with LoRA weights and uses neural inference for the 4 refined variables, supplementing with HRRR passthrough for the remaining 2

#### Scenario: Single data source at inference

- **WHEN** the service is running with the default NWP passthrough backbone
- **THEN** HRRR is the only external data dependency — no ERA5 access is required, and the service's update latency matches HRRR's cycle frequency (~hourly)
