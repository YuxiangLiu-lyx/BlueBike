# BlueBike

## How to Build and Run the Code

### Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Download Preprocessed Data

The data processing pipeline takes 15-20 hours to run from scratch. Preprocessed data is available on Hugging Face Hub:

```bash
python download_data.py
```

This downloads the following files (~25MB total):
- `data/processed/daily/daily_departures.parquet` (for Baseline model)
- `data/processed/daily/daily_with_xgb_features.parquet` (for XGBoost models)
- `data/processed/daily/daily_with_poi_features.parquet` (for POI models)
- `data/processed/grid/grid_with_poi_features.parquet` (for grid prediction)

**Optional**: To run trip-level EDA visualizations, uncomment the `trips_cleaned.parquet` line in `download_data.py` (adds 1GB download).

**Note**: If you prefer to generate data from scratch, see the "Data Preparation from Scratch" section below.

### Running Models

After downloading the data, run the models in order:

```bash
# Baseline Model - predicts 2024 demand using historical patterns from 2015-2023
python src/models/baseline.py
# Output: results/models/baseline_metrics.csv, baseline_predictions_2024.csv

# XGBoost Model - uses temporal, weather, and location features (lat/lon, nearby popularity)
python src/models/xgboost_model.py
# Output: results/xgboost/xgboost_metrics.csv, feature_importance.csv, xgboost_model.pkl

# XGBoost without location/history - removes coordinates and historical popularity features
python src/models/xgboost_no_popularity.py
# Output: results/xgboost_no_popularity/xgboost_metrics_no_popularity.csv, feature_importance_no_popularity.csv, xgboost_model_no_popularity.pkl

# XGBoost with POI features - replaces location with Points of Interest data
python src/models/xgboost_poi_only.py
# Output: results/xgboost_poi_only/xgboost_metrics_poi_only.csv, feature_importance_poi_only.csv, xgboost_model_poi_only.pkl

# Grid-based demand prediction - requires trained xgboost_poi_only model (run above command first)
# Uses the trained model to predict demand for new locations across Boston metro area
python src/analysis/predict_grid_demand.py
# Output: results/grid_analysis/grid_predictions.parquet, expansion_opportunities.csv, grid_demand_comparison.csv
```

### Generating Visualizations

```bash
# Model comparison charts - compare baseline vs XGBoost models
python src/visualization/model_comparison.py
# Output: results/comparison/model_comparison.png

python src/visualization/three_model_comparison.py
# Output: results/comparison/three_model_comparison.png

# Feature importance visualizations
python src/visualization/feature_importance_poi_only.py
# Output: results/xgboost_poi_only/feature_importance_poi_only.png

python src/visualization/feature_importance_no_location.py
# Output: results/xgboost_no_popularity/feature_importance_no_popularity.png

# Interactive maps - open the HTML files in a browser to view
python src/visualization/poi_station_map.py
# Output: results/poi_analysis/station_type_map.html, station_classifications.csv

python src/visualization/grid_poi_map.py
# Output: results/grid_analysis/grid_poi_map.html (POI distribution across grid)

python src/visualization/grid_demand_map.py
# Output: results/grid_analysis/grid_demand_map.html (predicted demand vs actual stations)

# Exploratory data analysis charts
python src/visualization/daily_analysis.py
# Output: results/figures/daily/daily_patterns.png

python src/visualization/station_analysis.py
# Output: results/figures/station/station_activity.png

# Optional: Trip-level EDA (requires trips_cleaned.parquet - see download_data.py)
# python src/visualization/eda_overview.py
# Output: results/figures/overview/yearly_departure_trend.png, results/statistics/yearly_statistics.csv
```

### Data Preparation from Scratch (Optional)

If you want to reproduce the entire data processing pipeline from raw data, run these commands in order. Note that this process is time-consuming and requires substantial disk space.

```bash
# Download raw Bluebikes trip data from 2015-2024 (takes 30+ minutes, ~5GB disk space)
python src/data_collection/bluebikes_download.py
# Output: data/raw/*.csv

# Clean and standardize the raw trip data (takes 10-20 minutes)
python src/preprocessing/data_cleaning.py
# Output: data/processed/bluebike_cleaned/trips_cleaned.parquet

# Add temporal features (month, day of week, season, holidays, etc)
python src/preprocessing/feature_engineering.py
# Output: data/processed/daily/daily_departures.parquet, daily_departures_sample.csv
#         data/processed/hourly/hourly_departures.parquet, hourly_departures_sample.csv

# Aggregate trips by station and date
python src/preprocessing/prepare_station_daily.py
# Output: data/processed/daily/station_daily_with_coords.parquet, station_daily_with_coords_sample.csv

# Fetch weather data for each station (takes 6-8 hours due to API rate limits)
python src/data_collection/weather_api.py
# Output: data/external/weather/station_daily_with_weather.parquet, station_daily_with_weather_sample.csv

# Add location-based features (nearby area popularity)
python src/preprocessing/add_xgb_features.py
# Output: data/processed/daily/daily_with_xgb_features.parquet, daily_with_xgb_features_sample.csv

# Extract POI features from OpenStreetMap (takes 1-4 hours)
python src/preprocessing/add_poi_features.py
# Output: data/processed/daily/daily_with_poi_features.parquet, daily_with_poi_features_sample.csv

# Create grid features for spatial prediction (takes 2-7 hours for POI extraction)
python src/analysis/create_grid_features.py
python src/analysis/add_grid_poi.py
# Output: data/processed/grid/grid_with_poi_features.parquet, grid_with_poi_features_sample.csv
#         data/processed/grid/grid_base_features.parquet, grid_base_features_sample.csv
```

---

## Interactive Visualizations

This project includes interactive HTML maps that can be directly opened in a browser:
- **`results/poi_analysis/station_type_map.html`** - Station classification by surrounding POI types
- **`results/grid_analysis/grid_poi_map.html`** - POI distribution across 500m×500m grid cells in Boston metro area
- **`results/grid_analysis/grid_demand_map.html`** - Predicted demand heatmap compared with actual station distribution

All other visualizations (model comparison charts, feature importance plots, EDA figures) are static PNG images referenced throughout this README.

---

## Project Structure

### Directory Organization

- **`src/`** - Source code organized by functionality
  - `data_collection/` - Scripts for downloading raw trip data and fetching weather information
  - `preprocessing/` - Data cleaning and feature engineering pipelines
  - `models/` - Machine learning model implementations (Baseline, XGBoost variants)
  - `analysis/` - Advanced analysis scripts (grid-based spatial prediction)
  - `visualization/` - Visualization generation scripts for charts and interactive maps
- **`data/`** - Data storage (excluded from repository due to size)
  - `raw/` - Original trip data CSV files
  - `processed/` - Cleaned and feature-enriched datasets
  - `external/` - Weather data and POI caches
- **`results/`** - Model outputs, metrics, and visualizations
  - `models/` - Baseline model results
  - `xgboost*/` - Various XGBoost model outputs
  - `comparison/` - Model comparison charts
  - `figures/` - Exploratory data analysis visualizations
  - `poi_analysis/` - POI-related visualizations
  - `grid_analysis/` - Spatial prediction results and maps

### Development Phases

#### Phase 1: Initial Exploration and Baseline Model

In the initial phase, I focused on understanding the Bluebikes ridership patterns. I collected historical trip data from 2015-2024, cleaned and standardized the dataset, and engineered temporal features (month, day of week, season, holidays). I also integrated weather data from Open-Meteo API to capture external conditions affecting ridership. For the XGBoost model, I added location-based features including coordinates (latitude/longitude) and a `nearby_avg_popularity` metric representing the historical activity level of surrounding stations. I created various visualizations to explore the data and built two prediction models: a simple baseline and an XGBoost model. Detailed analysis of this phase is documented in the **Mid-term Progress** section below.

**Key Scripts (Phase 1):**
- `src/data_collection/bluebikes_download.py` - Downloads raw trip data from Bluebikes official website
- `src/preprocessing/data_cleaning.py` - Cleans and standardizes raw trip records
- `src/preprocessing/feature_engineering.py` - Adds temporal features (month, day, season, holidays)
- `src/preprocessing/prepare_station_daily.py` - Aggregates trips by station and date
- `src/data_collection/weather_api.py` - Fetches historical weather data for each station
- `src/preprocessing/add_xgb_features.py` - Adds location-based features including `nearby_avg_popularity`
- `src/models/baseline.py` - Implements a simple historical-pattern-based baseline model
- `src/models/xgboost_model.py` - Trains XGBoost model with all features (location, weather, temporal)

#### Phase 2: Generalization for New Station Prediction

The Phase 1 model, while accurate for existing stations, relies on location-specific features (coordinates and historical popularity) that would not be available for potential new station locations. To build a model capable of predicting demand at arbitrary locations across Boston, I removed these "leakage" features and instead extracted Points of Interest (POI) data from OpenStreetMap. POI features (subway stations, bus stops, restaurants, shops, offices, universities, schools, hospitals, banks, parks) provide generalizable contextual information about an area without requiring historical ridership data. I retrained XGBoost models with different feature combinations to evaluate the impact of removing location features and adding POI features. I also implemented a grid-based spatial prediction system that divides Boston metro area into 500m×500m cells and predicts demand for each cell. Detailed analysis is in the **Phase 2 Results** section.

**Key Scripts (Phase 2):**
- `src/preprocessing/add_poi_features.py` - Extracts POI features from OpenStreetMap for each station
- `src/models/xgboost_no_popularity.py` - XGBoost without coordinates and historical popularity
- `src/models/xgboost_poi_only.py` - XGBoost using POI features instead of location-specific features
- `src/analysis/create_grid_features.py` - Creates 500m×500m grid covering Boston metro area with temporal and weather features
- `src/analysis/add_grid_poi.py` - Extracts POI features for each grid cell
- `src/analysis/predict_grid_demand.py` - Predicts demand for grid cells using the POI-based model

**Visualization Scripts:**

All visualization scripts are in `src/visualization/`:
- `eda_overview.py` - System-level yearly statistics and growth trends
- `daily_analysis.py` - Daily and weekly ridership patterns
- `station_analysis.py` - Station-level activity distribution
- `model_comparison.py` - Compares Baseline vs XGBoost performance
- `three_model_comparison.py` - Compares Baseline, No Location, and POI Only models
- `feature_importance_poi_only.py` - Visualizes POI feature importance
- `feature_importance_no_location.py` - Visualizes feature importance without location features
- `poi_station_map.py` - Interactive map showing station classifications by surrounding POI types
- `grid_poi_map.py` - Interactive map displaying POI distribution across grid cells
- `grid_demand_map.py` - Interactive map showing predicted demand vs actual station distribution

---

## Proposal

### 1. Project Description & Motivation

The Bluebikes program, Boston's official bike-share system, makes its complete ridership data publicly available. This rich dataset provides details for every trip, including:

- Trip Duration (in seconds)
- Start and Stop Times
- Start and End Station Names & IDs
- Bike ID
- User Type (Casual vs. Member)

This data allows for a deep exploration of user ridership patterns and behaviors. The primary objective of this project is to leverage this dataset to uncover actionable insights into how, when, and where the service is used. Understanding these habits is the first step toward building predictive models that can help optimize the system's operational efficiency.


### 2. Project Objectives

The ultimate goal of this project is to provide data-driven recommendations that could help Bluebikes optimize its operational efficiency and improve bike availability for its users. To achieve this, I have defined the following core objectives:

- **Develop a Predictive Demand Model**: My primary technical objective is to build and validate a machine learning model that forecasts hourly bike demand. The model will predict the number of bike departures from any given station based on a range of features, including:
  - Station characteristics (e.g., location)
  - Temporal factors (time of day, day of week, holidays)
  - External conditions (historical weather data)

- **Generate a City-Wide Demand Heatmap for Strategic Network Planning**: Beyond forecasting for existing stations, the second aim is to formulate a spatial prediction model to estimate potential bike demand across the Boston/Cambridge area as a whole. This would be then used to locate strategically appropriate sites for the expansion and/or consolidation of the network.
  - **Prediction Target**: The potential number of bike departures within a given hour from a particular geographical grid cell—for instance, a 100m x 100m cell.
  - **Feature Engineering**: For the purpose of predicting demand at locations without previously existing stations, a rich geospatial feature set will be constructed at the level of the grid cell, which might include(not fixed):
    - The proximity to public transit interchange points, e.g., subway and bus stops
    - Density of points of interest, e.g., restaurants, offices, and parks
    - Local demographic features, e.g., population density
    - Local bike network characteristic features, e.g., the presence of protected bike lanes
  - **Expected Outcome**: The model output will be a complete city demand heatmap, highlighting untapped "hotspots" (potentially ideal locations for new stations) and demand "coldspots" (where existing stations may be underutilized and could be considered for relocation or removal).

- **Propose Novel Operational Strategies** (if time permitted): As a secondary objective, I will explore how the model could inform new strategies, such as implementing dynamic pricing during off-peak hours or offering user incentives to help redistribute bikes from over-supplied to under-supplied stations.

### 3. Data Collection Strategy

This project will utilize three primary data sources: Bluebikes' official ridership data, historical weather data, and various geospatial datasets describing the Boston area.

### Bluebikes Ridership Data

- **Source**: The complete dataset of historical trip data will be sourced directly from the official Bluebikes website: https://bluebikes.com/system-data
- **Methodology**: I will download the monthly trip data files (in CSV format). This will form the core of my dataset for training and testing my predictive models

### Historical Weather Data

- **Source**: Hourly historical weather data will be obtained via the Open-Meteo Historical Weather API, a reliable and publicly accessible source: https://open-meteo.com/en/docs/historical-weather-api
- **Methodology**: I will implement a Python script to programmatically collect the weather data. The process will be as follows:
  - Extract the latitude and longitude coordinates for every Bluebikes station from the ridership data
  - For my specified date range, the script will make API calls to request the corresponding hourly weather variables (including temperature, precipitation, and wind speed) for each station's location
  - The API responses will be parsed, cleaned, and structured into a tabular format ready to be merged with the Bluebikes trip data

### Geospatial & Demographic Data for Spatial Modeling

#### Public Transit Network

- **Source**: Official GTFS data for the MBTA network. The most up-to-date files are now version-controlled and can be found on the MBTA's official GitHub-style archive: https://cdn.mbta.com/archive/archived_feeds.txt
- **Methodology**: I will parse these files to extract the precise latitude and longitude of all subway and bus stops, which are critical drivers of bike trips.

#### Urban Infrastructure & Points-of-Interest (POIs)

- **Source**: This kind of data can be accessed via the OSMnx library: https://osmnx.readthedocs.io/en/stable/
- **Methodology**: I will use the Python library OSMnx to programmatically get data on the locations of key POIs (e.g., restaurants, offices, universities, parks).

#### Demographic Data

- **Source**: The U.S. Census Bureau's public data portal: https://data.census.gov/
- **Methodology**: I will acquire population density and potentially other demographic data at the census tract level. This will allow me to correlate potential bike demand with the characteristics of the people living and working in each area.


### 4. Modeling Approach

Since precise prediction of bike departures is challenging, my approach is to conduct a comparative study. I will implement several types of models to establish performance benchmarks and identify the most suitable method for this problem.

- **Baseline Model**: I will first establish a simple, non-machine learning baseline. This model will use straightforward rules to make predictions, serving as a critical benchmark to measure the effectiveness of my more advanced models

- **Tree-Based Models**: The ridership data is highly structured, making tree-based models an excellent candidate. I will explore ensemble methods like Gradient Boosting (e.g., XGBoost), which are known to perform very well on this type of tabular data.

  To effectively use these models, I will reframe the problem from a time-series task to a standard regression/classification task. This involves extensive feature engineering to explicitly extract key information—such as the hour of day, holiday status, station location, and weather conditions—from the raw data.

- **Deep Learning Models**: To explore alternative patterns in the data, especially potential time-series dependencies, I will also implement a deep learning model. This will provide a valuable point of comparison against the performance of the tree-based models

- **Exploratory LLM Integration** (If time permitted): As an extension, I will explore the capabilities of a Large Language Model (LLM). I plan to test its utility in two ways: first, as a tool for advanced feature engineering by interpreting the context of a situation, and second, by evaluating its ability to make direct predictions through prompting

### 5. Data Visualization Plan

My visualization strategy will focus on two key areas: exploratory data analysis to understand the data's underlying patterns and results visualization to interpret my model's performance. I will primarily use Python libraries such as Matplotlib, Seaborn, and Folium.

- **Geospatial Visualization**: I will create interactive maps of the Boston area to visualize station locations and ridership flow. A key visualization will be a heatmap of trip start and end points to quickly identify the most popular hubs and travel corridors in the city

- **Temporal Pattern Visualization**: To understand how demand changes over time, I will create:
  - Time-series plots showing daily and weekly trip counts to identify long-term trends and seasonality
  - Bar charts aggregating trips by the hour of the day and day of the week to reveal daily commute patterns and weekend activity

- **Correlation Visualization**: I will use scatter plots to investigate the relationship between weather conditions (like temperature and precipitation) and the number of bike trips, helping me validate key features for my model

- **Model Performance Visualization**: To evaluate my final model, I will plot the predicted demand values against the actual values over time. This will allow me to visually inspect where my model performs well and where it struggles

### 6. Test Plan

Because I am working with time-series data, I will not use a random split. I will perform a temporal split, using historical data for training and a more recent, unseen period for testing. For example, I might train my model on data from 2023-2024 and test its performance on data from 2025.

---

## Mid-term Progress

**Project Presentation**: https://youtu.be/K75Yq7wpM4Y

### Data Visualization

I created several visualizations to understand the Bluebikes data better. Here's what I found:

#### System Growth Over Time
![Yearly Trend](results/figures/overview/yearly_departure_trend.png)

The system grew from about 1.1 million trips in 2015 to 4.7 million trips in 2024, with the number of stations increasing from around 100 to over 400. I excluded 2020 data from model training due to COVID-19 impacts.

#### Daily Patterns
![Daily Patterns](results/figures/daily/daily_patterns.png)

I noticed clear patterns in when people use the bikes:
- **Weekdays vs Weekends**: People use bikes differently on weekdays compared to weekends. Weekdays show typical commute patterns.
- **Monthly Changes**: Summer months (June-August) are way busier than winter months.

#### Station Activity
![Station Activity](results/figures/station/station_activity.png)

Not all stations are equal. I found:
- The top 20 stations handle a huge portion of all trips
- Most stations are "medium activity" - they're not super busy but they're not empty either
- The network keeps growing - new stations get added almost every year

### Data Processing

Here's what I did:

#### Data Collection
I downloaded all the monthly trip files from the Bluebikes website (2015-2024). That's about 28.6 million trips total. Each trip record tells us when and where someone picked up a bike and where they dropped it off.

#### Data Cleaning
The raw data had some issues:
- **Different column names**: Older files used different names than newer ones. For example, "subscriber" vs "member". I standardized everything.
- **Missing data**: Some early files didn't have trip duration, so I calculated it from start time and end time.
- **Station ID problems**: Station IDs changed from numbers to letters+numbers over the years. I decided to use station names instead since those are more consistent.
- **Outliers**: Some trips have unrealistic durations (under 10 seconds or over several days), but I kept all records in the dataset.

#### Feature Engineering
I added a bunch of useful information to each trip:
- **Time features**: What month? What day of week? Is it a weekend? What season?
- **Holiday flag**: Is this date a US federal holiday?
- **Location features**: For each station, I calculated how busy the surrounding area is based on historical data

#### Weather Data
I used the Open-Meteo API to get daily weather for each station. The challenge was:
- I needed weather for 800+ stations
- I needed 10 years of data (2015-2024)
- The API has rate limits (about 10 requests per minute)

So I built a system that caches the data as it downloads, which means if it gets interrupted, I can pick up where I left off instead of starting over. It took about 7 hours to get all the weather data.

The weather data includes: temperature (max/min/mean), precipitation, rain, snow, and wind speed.

#### Data Aggregation
Since I'm predicting daily demand, I grouped all the trips by station and date. The final dataset has:
- **1.14 million records** (one for each station-date combination)
- **18 features** total
- **Target variable**: number of departures per station per day

### Modeling Methods

I built two models to compare:

#### Baseline Model
This is a simple model that assumes patterns repeat. Here's how it works:
1. Look at all the historical data (2015-2023, skipping 2020)
2. For each station, calculate what percentage of total trips it usually gets on day X of the year
3. Estimate what 2024's total trips will be using average growth rate from previous years
4. Multiply: station's historical percentage × estimated 2024 total = prediction

This model is simple but gives us something to beat.

#### XGBoost Model
I use XGBoost as my first machine learning model for prediction. It uses gradient boosting, which is basically a bunch of decision trees working together.

**Features I used**:
- Time: month, day, day of week, season, weekend flag, holiday flag
- Location: latitude, longitude, plus how busy the surrounding area usually is
- Weather: 9 weather variables (temperature, precipitation, wind)

**Feature selection note**:
When I included "year" as a feature, the model predicted total 2024 volume far higher than the actual value. This is probably because the model learned a simple linear trend rather than understanding the actual patterns. So I removed the "year" feature and also excluded "previous month's total trips" to avoid mixing seasonal patterns with growth trends. I also used strong regularization (L1 and L2) to prevent overfitting.

**Training setup**:
- Training data: 2015-2023 (excluding 2020) = about 680K records
- Test data: 2024 = about 205K records
- I used 500 trees with a max depth of 5

### Preliminary Results

#### Feature Importance

![Feature Importance](results/xgboost/feature_importance.png)

The most important features for prediction are:
1. **Nearby area activity** (22.1%) - Historical activity level of surrounding stations
2. **Temperature max** (13.0%) - Daily maximum temperature
3. **Latitude** (10.4%) - Station location (north-south)
4. **Longitude** (9.1%) - Station location (east-west)
5. **Temperature mean** (8.6%) - Daily average temperature
6. **Season** (6.6%) - Time of year
7. **Precipitation hours** (6.3%) - Duration of rain/snow

Location and weather are the primary drivers of bike demand, with the surrounding area's historical popularity being the single most important factor.

#### Model Comparison

![Model Comparison](results/comparison/model_comparison.png)

| Metric | Baseline | XGBoost | Improvement |
|--------|----------|---------|-------------|
| R² Score | 0.101 | 0.258 | +154% |
| MAE | 19.27 | 17.40 | -9.7% |
| RMSE | 32.68 | 28.69 | -12.2% |
| Total Error | +4.00% | +14.17% | Worse |

The XGBoost model shows significant improvements over the baseline: R² Score increased by 154% (from 0.101 to 0.258), MAE decreased by 9.7% (from 19.27 to 17.40 bikes/day), and RMSE decreased by 12.2% (from 32.68 to 28.69). However, the model overestimates total 2024 volume by 14%, compared to the baseline's 4% overestimation. This overestimation led to an important discovery about the practical value of this model.

#### Overestimation Analysis: Identifying Capacity Constraints

![Overestimation Analysis](results/analysis/overestimation_analysis.png)

I conducted a detailed analysis of days where the model's predictions significantly exceeded actual ridership. The key finding was that **these high-prediction days also had actual ridership far above the station's average**. This suggests that the model correctly identified high-demand periods, but the actual usage was constrained by station capacity (limited number of bikes or docks).

**Key Insights:**

1. **Capacity Bottlenecks**: When the model predicts much higher demand than observed, it often indicates the station is operating at or near full capacity. The "missing" trips aren't prediction errors—they're lost revenue opportunities due to insufficient bikes or docks.

2. **Expansion Opportunities**: Stations with consistent overestimation are prime candidates for capacity expansion. Adding more bikes or docks at these locations would likely increase actual ridership and generate additional revenue.

3. **Practical Value of Phase 1 Model**: While the location-based features (coordinates, `nearby_avg_popularity`) make this model unsuitable for predicting demand at entirely new locations, they are **highly valuable for capacity planning at existing stations**. The model excels at identifying which stations need more resources based on historical usage patterns and local context.

This analysis demonstrates that prediction "errors" can reveal actionable business insights. The Phase 1 model serves a different but equally important purpose: optimizing the existing network by identifying underutilized capacity versus capacity constraints.

### Summary

I have completed data collection, cleaning, and visualization for the Bluebikes dataset. I built and compared two prediction models: a simple baseline model and an XGBoost machine learning model. The XGBoost model shows clear improvements with R² increasing from 0.101 to 0.258, demonstrating that machine learning can capture patterns that simple historical methods miss. Analysis of the model's overestimation patterns revealed that high predictions often coincide with days when actual usage is already far above average, indicating **capacity constraints** rather than prediction errors. This finding demonstrates the practical value of this model: while its location-based features make it unsuitable for new station placement, it is **highly effective for identifying existing stations that need capacity expansion**.

**Code Organization**: All code is available in the GitHub repo:
- `src/data_collection/` - Data download scripts
- `src/preprocessing/` - Data cleaning and feature engineering
- `src/models/` - Baseline and XGBoost models
- `src/visualization/` - Visualization scripts
- `results/` - Figures and metrics
- `data/` - Raw and processed data (not uploaded due to size, excluded via .gitignore)

---

## Phase 2 Results



---

## Summary

