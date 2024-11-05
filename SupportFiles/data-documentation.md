# PeMS Traffic Data Processing Documentation

## Data Structure Overview
This document describes the processed traffic data structure resulting from the preprocessing pipeline. The data has been transformed and enhanced to facilitate machine learning applications while preserving the original information.

### File Organization
- Original data files are preserved in their original locations
- Processed files are prefixed with "processed_" and maintain the same directory structure
- Each processed file contains enhanced and normalized versions of the original metrics

### Column Descriptions

#### Temporal Features
| Column | Description | Type |
|--------|-------------|------|
| Timestamp | Original timestamp | datetime |
| Hour | Hour of day (0-23) | int |
| Day_of_Week | Day of week (0=Monday, 6=Sunday) | int |
| Is_Weekend | Binary indicator for weekend days | int (0/1) |
| Month | Month of year (1-12) | int |
| Year | Year | int |
| Is_Peak_Hour | Binary indicator for peak hours (7-9AM, 4-6PM) | int (0/1) |

#### Station Information
| Column | Description | Type |
|--------|-------------|------|
| Station | Station identifier | int |
| District | District number | int |
| Freeway | Freeway number | int |

#### Directional Features (One-Hot Encoded)
Original 'Direction' column is encoded into binary columns:
| Column | Description | Type |
|--------|-------------|------|
| Direction_S | South direction indicator | float (0/1) |
| Direction_E | East direction indicator | float (0/1) |
| Direction_W | West direction indicator | float (0/1) |
Note: North (N) is the reference category and is indicated when all direction columns are 0

#### Lane Type Features (One-Hot Encoded)
Original 'Lane_Type' column is encoded into binary columns:
| Column | Description | Type |
|--------|-------------|------|
| Lane_Type_CH | Conventional Highway indicator | float (0/1) |
| Lane_Type_FF | Fwy-Fwy connector indicator | float (0/1) |
| Lane_Type_FR | Off Ramp indicator | float (0/1) |
| Lane_Type_HV | HOV indicator | float (0/1) |
| Lane_Type_ML | Mainline indicator | float (0/1) |
| Lane_Type_OR | On Ramp indicator | float (0/1) |
Note: Collector/Distributor (CD) is the reference category and is indicated when all lane type columns are 0

#### Traffic Metrics
| Column | Description | Type |
|--------|-------------|------|
| Station_Length | Segment length covered by station | float |
| Samples | Total number of samples received | int |
| Pct_Observed | Percentage of observed (non-imputed) data points | float |
| Total_Flow | Total flow across all lanes (vehicles/5-min) | float |
| Avg_Occupancy | Average occupancy across all lanes | float |
| Avg_Speed | Flow-weighted average speed across all lanes | float |

#### Lane-Specific Metrics
For each lane (1-8), the following metrics are provided:
| Column Pattern | Description | Type |
|--------|-------------|------|
| Lane_N_Samples | Number of samples for lane N | int |
| Lane_N_Flow | Flow for lane N (vehicles/5-min) | float |
| Lane_N_Avg_Occ | Average occupancy for lane N | float |
| Lane_N_Avg_Speed | Average speed for lane N | float |
| Lane_N_Observed | Observation indicator for lane N | int (0/1) |
| Lane_N_Efficiency | Calculated efficiency (Flow × Speed) | float |

#### Derived Metrics
| Column | Description | Type |
|--------|-------------|------|
| Total_Lane_Efficiency | Sum of all lane efficiencies | float |
| Active_Lanes | Count of lanes with positive flow | int |

#### Normalized Features
For each numeric feature (except categorical IDs and temporal features), a normalized version is provided:
| Column Pattern | Description | Type |
|--------|-------------|------|
| {feature}_Normalized | Zero-mean, unit-variance version of the feature | float |

### Data Quality Notes
- Missing values in numeric columns are filled with column means
- Categorical variables are one-hot encoded for machine learning compatibility
- All numeric features (except IDs and temporal) are normalized
- Original categorical values are preserved for reference
- Timestamps are standardized to datetime format

### Usage Considerations
1. For machine learning:
   - Use normalized versions of features when applicable
   - Categorical variables are already one-hot encoded
   - Consider using Is_Weekend and Is_Peak_Hour as feature flags

2. For analysis:
   - Original metrics are preserved for interpretability
   - Efficiency metrics provide derived insights
   - Active_Lanes helps identify operational capacity

3. For time series analysis:
   - Timestamp is preserved in standard format
   - Multiple temporal features are available
   - Data is organized in 5-minute intervals

### File Format
- Output files are in CSV format
- First row contains headers
- Missing values are represented as NaN
- Decimal precision is preserved from source data

### Processing Notes
- All numeric calculations use 64-bit floating-point precision
- Temporal features are extracted from local timezone
- Lane efficiency is calculated as (flow × speed) for each lane
- Normalized features follow standard scaling (μ=0, σ=1)
