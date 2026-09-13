## Purpose

Extracts a static terrain feature tensor from DEM raster data, providing the terrain conditioning input branch for the downscaling head.

## ADDED Requirements

### Requirement: DEM ingestion at arbitrary resolution

The system SHALL accept GeoTIFF DEM rasters at any input resolution and reproject/resample to a target grid defined by the caller. The system SHALL support SRTM 30m, Copernicus DEM 30m, and USGS 1/3 arc-second as input sources.

#### Scenario: Ingest SRTM 30m tile

- **WHEN** a SRTM 30m GeoTIFF covering a bounding box is provided
- **THEN** the system produces an elevation grid at the requested target resolution with CRS metadata preserved

#### Scenario: Mismatched CRS

- **WHEN** the input DEM CRS differs from the target projection
- **THEN** the system reprojects the DEM to the target CRS before feature extraction

### Requirement: Feature set matching PeakWeather topographic schema

The system SHALL compute the following terrain features from the DEM, matching the MeteoSwiss/PeakWeather topographic variable schema:

1. **Elevation** (meters above sea level)
2. **Slope** (degrees, computed via Horn's method)
3. **Aspect** (degrees clockwise from north, 0-360, flat pixels assigned -1)
4. **Plan curvature** (perpendicular to slope direction, units 1/m)
5. **Profile curvature** (parallel to slope direction, units 1/m)
6. **Sky-view factor (SVF)** (dimensionless 0-1, computed with 36 azimuth sectors and configurable maximum search radius, default 5km)
7. **Sx upwind exposure** (8 cardinal directions: N, NE, E, SE, S, SW, W, NW; degrees, positive = sheltered, negative = exposed; search radius configurable, default 3km)
8. **Topographic Position Index (TPI)** at 500m radius (elevation minus mean elevation within radius, meters)
9. **Surface roughness class** (NLCD land cover mapped to aerodynamic roughness length z0 in meters via a lookup table)

#### Scenario: Compute full feature set for a DEM patch

- **WHEN** a DEM patch and corresponding NLCD raster are provided
- **THEN** the system returns a tensor of shape `(C, H, W)` where C=14 (elevation, slope, aspect, plan curvature, profile curvature, SVF, 8 Sx directions, TPI) plus a roughness channel, and H/W match the DEM grid dimensions

#### Scenario: Missing NLCD coverage

- **WHEN** no NLCD raster is available for the region (e.g., Swiss Alps)
- **THEN** the system falls back to a configurable default roughness class and logs a warning

### Requirement: Static tensor output

The terrain feature tensor SHALL be computed once per DEM region and cached. It is a static input branch — it does not change per timestep. The output tensor SHALL be a PyTorch tensor on the configured device (CPU, CUDA, or MPS).

#### Scenario: Repeated inference calls for same region

- **WHEN** the downscaling head requests terrain features for a region that has already been computed
- **THEN** the system returns the cached tensor without recomputation

### Requirement: Normalization

Each terrain feature channel SHALL be normalized using statistics computed from the training dataset. The system SHALL store per-channel mean and standard deviation and apply z-score normalization. The normalization parameters SHALL be saved with the model checkpoint.

#### Scenario: Load normalization from checkpoint

- **WHEN** a trained model checkpoint is loaded
- **THEN** the terrain encoder applies the same normalization statistics that were used during training
