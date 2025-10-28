# BlueBike

## 1. Project Description & Motivation

The Bluebikes program, Boston's official bike-share system, makes its complete ridership data publicly available. This rich dataset provides details for every trip, including:

- Trip Duration (in seconds)
- Start and Stop Times
- Start and End Station Names & IDs
- Bike ID
- User Type (Casual vs. Member)

This data allows for a deep exploration of user ridership patterns and behaviors. The primary objective of this project is to leverage this dataset to uncover actionable insights into how, when, and where the service is used. Understanding these habits is the first step toward building predictive models that can help optimize the system's operational efficiency.


## 2. Project Objectives

The ultimate goal of this project is to provide data-driven recommendations that could help Bluebikes optimize its operational efficiency and improve bike availability for its users. To achieve this, we have defined the following core objectives:

- **Develop a Predictive Demand Model**: Our primary technical objective is to build and validate a machine learning model that forecasts hourly bike demand. The model will predict the number of bike departures from any given station based on a range of features, including:
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

- **Propose Novel Operational Strategies** (if time permitted): As a secondary objective, we will explore how the model could inform new strategies, such as implementing dynamic pricing during off-peak hours or offering user incentives to help redistribute bikes from over-supplied to under-supplied stations.

## 3. Data Collection Strategy

This project will utilize three primary data sources: Bluebikes' official ridership data, historical weather data, and various geospatial datasets describing the Boston area.

### Bluebikes Ridership Data

- **Source**: The complete dataset of historical trip data will be sourced directly from the official Bluebikes website: https://bluebikes.com/system-data
- **Methodology**: We will download the monthly trip data files (in CSV format). This will form the core of our dataset for training and testing our predictive models

### Historical Weather Data

- **Source**: Hourly historical weather data will be obtained via the Open-Meteo Historical Weather API, a reliable and publicly accessible source: https://open-meteo.com/en/docs/historical-weather-api
- **Methodology**: We will implement a Python script to programmatically collect the weather data. The process will be as follows:
  - Extract the latitude and longitude coordinates for every Bluebikes station from the ridership data
  - For our specified date range, the script will make API calls to request the corresponding hourly weather variables (including temperature, precipitation, and wind speed) for each station's location
  - The API responses will be parsed, cleaned, and structured into a tabular format ready to be merged with the Bluebikes trip data

### Geospatial & Demographic Data for Spatial Modeling

#### Public Transit Network

- **Source**: Official GTFS data for the MBTA network. The most up-to-date files are now version-controlled and can be found on the MBTA's official GitHub-style archive: https://cdn.mbta.com/archive/archived_feeds.txt
- **Methodology**: We will parse these files to extract the precise latitude and longitude of all subway and bus stops, which are critical drivers of bike trips.

#### Urban Infrastructure & Points-of-Interest (POIs)

- **Source**: This kind of data can be accessed via the OSMnx library: https://osmnx.readthedocs.io/en/stable/
- **Methodology**: We will use the Python library OSMnx to programmatically get data on the locations of key POIs (e.g., restaurants, offices, universities, parks).

#### Demographic Data

- **Source**: The U.S. Census Bureau's public data portal: https://data.census.gov/
- **Methodology**: We will acquire population density and potentially other demographic data at the census tract level. This will allow us to correlate potential bike demand with the characteristics of the people living and working in each area.


## 4. Modeling Approach

Since precise prediction of bike departures is challenging, our approach is to conduct a comparative study. We will implement several types of models to establish performance benchmarks and identify the most suitable method for this problem.

- **Baseline Model**: We will first establish a simple, non-machine learning baseline. This model will use straightforward rules to make predictions, serving as a critical benchmark to measure the effectiveness of our more advanced models

- **Tree-Based Models**: The ridership data is highly structured, making tree-based models an excellent candidate. We will explore ensemble methods like Gradient Boosting (e.g., XGBoost), which are known to perform very well on this type of tabular data.

  To effectively use these models, we will reframe the problem from a time-series task to a standard regression/classification task. This involves extensive feature engineering to explicitly extract key information—such as the hour of day, holiday status, station location, and weather conditions—from the raw data.

- **Deep Learning Models**: To explore alternative patterns in the data, especially potential time-series dependencies, we will also implement a deep learning model. This will provide a valuable point of comparison against the performance of the tree-based models

- **Exploratory LLM Integration** (If time permitted): As an extension, we will explore the capabilities of a Large Language Model (LLM). We plan to test its utility in two ways: first, as a tool for advanced feature engineering by interpreting the context of a situation, and second, by evaluating its ability to make direct predictions through prompting

## 5. Data Visualization Plan

Our visualization strategy will focus on two key areas: exploratory data analysis to understand the data's underlying patterns and results visualization to interpret our model's performance. We will primarily use Python libraries such as Matplotlib, Seaborn, and Folium.

- **Geospatial Visualization**: We will create interactive maps of the Boston area to visualize station locations and ridership flow. A key visualization will be a heatmap of trip start and end points to quickly identify the most popular hubs and travel corridors in the city

- **Temporal Pattern Visualization**: To understand how demand changes over time, we will create:
  - Time-series plots showing daily and weekly trip counts to identify long-term trends and seasonality
  - Bar charts aggregating trips by the hour of the day and day of the week to reveal daily commute patterns and weekend activity

- **Correlation Visualization**: We will use scatter plots to investigate the relationship between weather conditions (like temperature and precipitation) and the number of bike trips, helping us validate key features for our model

- **Model Performance Visualization**: To evaluate our final model, we will plot the predicted demand values against the actual values over time. This will allow us to visually inspect where our model performs well and where it struggles

## 6. Test Plan

Because we are working with time-series data, we will not use a random split. We will perform a temporal split, using historical data for training and a more recent, unseen period for testing. For example, we might train our model on data from 2023-2024 and test its performance on data from 2025.

---

# Midterm Report

**Project Presentation Video**: [Link to be added]

## 1. Data Visualization

We created several visualizations to understand the Bluebikes data better. Here's what we found:

### System Growth Over Time
![Yearly Trend](results/figures/overview/yearly_departure_trend.png)

The system grew from about 1.1 million trips in 2015 to 4.7 million trips in 2024, with the number of stations increasing from around 100 to over 400. We excluded 2020 data from model training due to COVID-19 impacts.

### Daily Patterns
![Daily Patterns](results/figures/daily/daily_patterns.png)

We noticed clear patterns in when people use the bikes:
- **Weekdays vs Weekends**: People use bikes differently on weekdays compared to weekends. Weekdays show typical commute patterns.
- **Monthly Changes**: Summer months (June-August) are way busier than winter months.

### Station Activity
![Station Activity](results/figures/station/station_activity.png)

Not all stations are equal. We found:
- The top 20 stations handle a huge portion of all trips
- Most stations are "medium activity" - they're not super busy but they're not empty either
- The network keeps growing - new stations get added almost every year

## 2. Data Processing

Here's what we did:

### Data Collection
We downloaded all the monthly trip files from the Bluebikes website (2015-2024). That's about 28.6 million trips total. Each trip record tells us when and where someone picked up a bike and where they dropped it off.

### Data Cleaning
The raw data had some issues:
- **Different column names**: Older files used different names than newer ones. For example, "subscriber" vs "member". We standardized everything.
- **Missing data**: Some early files didn't have trip duration, so we calculated it ourselves from start time and end time.
- **Station ID problems**: Station IDs changed from numbers to letters+numbers over the years. We decided to use station names instead since those are more consistent.
- **Weird data**: Some trips were impossibly short (like 10 seconds) or super long (days). We kept them for now but noted them.

### Feature Engineering
We added a bunch of useful information to each trip:
- **Time features**: What month? What day of week? Is it a weekend? What season?
- **Holiday flag**: Is this date a US federal holiday?
- **Location features**: For each station, we calculated how busy the surrounding area is based on historical data

### Weather Data
We used the Open-Meteo API to get daily weather for each station. The challenge was:
- We needed weather for 800+ stations
- We needed 10 years of data (2015-2024)
- The API has rate limits (about 10 requests per minute)

So we built a system that caches the data as it downloads, which means if it gets interrupted, we can pick up where we left off instead of starting over. It took about 7 hours to get all the weather data.

The weather data includes: temperature (max/min/mean), precipitation, rain, snow, and wind speed.

### Data Aggregation
Since we're predicting daily demand, we grouped all the trips by station and date. The final dataset has:
- **1.14 million records** (one for each station-date combination)
- **18 features** total
- **Target variable**: number of departures per station per day

## 3. Modeling Methods

We built two models to compare:

### Baseline Model
This is a simple model that assumes patterns repeat. Here's how it works:
1. Look at all the historical data (2015-2023, skipping 2020)
2. For each station, calculate what percentage of total trips it usually gets on day X of the year
3. Estimate what 2024's total trips will be using average growth rate from previous years
4. Multiply: station's historical percentage × estimated 2024 total = prediction

This model is simple but gives us something to beat.

### XGBoost Model
This is our "real" machine learning model. It uses gradient boosting, which is basically a bunch of decision trees working together.

**Features we used**:
- Time: month, day, day of week, season, weekend flag, holiday flag
- Location: latitude, longitude, plus how busy the surrounding area usually is
- Weather: 9 weather variables (temperature, precipitation, wind)

**What we did to prevent overfitting**:
- We intentionally did NOT include "year" as a feature. Why? Because if we did, the model would just learn "2024 = big number" and predict huge values. We want it to learn actual patterns instead.
- We also skipped "previous month's total trips" because that mixes up seasonal patterns with growth trends
- We used strong regularization (L1 and L2) to keep the model from memorizing the training data

**Training setup**:
- Training data: 2015-2023 (excluding 2020) = about 900K records
- Test data: 2024 = about 200K records
- We used 500 trees with a max depth of 5

## 4. Results

Here's how the two models performed:

![Model Comparison](results/comparison/model_comparison.png)

### Key Numbers

| Metric | Baseline | XGBoost | Winner |
|--------|----------|---------|--------|
| R² Score | 0.101 | 0.258 | XGBoost ✓ |
| MAE | 19.27 | 17.40 | XGBoost ✓ |
| RMSE | 32.68 | 28.69 | XGBoost ✓ |
| Total Error | +4.00% | +14.17% | Baseline ✓ |

### What This Means

**R² Score** is the most important metric. It tells us how much of the variation in the data our model can explain:
- Baseline explains 10.1% of the variation
- XGBoost explains 25.8% of the variation
- That's a 154% improvement!

But here's something interesting: even though XGBoost is better at predicting individual station-days, it overestimates the total 2024 volume by 14%, while Baseline only overestimates by 4%.

**MAE (Mean Absolute Error)** tells us the average prediction error:
- Baseline: off by about 19 bikes per day per station
- XGBoost: off by about 17 bikes per day per station

**Why is R² still relatively low?** Predicting daily bike demand at individual stations is really hard because:
- Weather can change suddenly
- Random events happen (concerts, sports games, construction)
- Individual behavior is unpredictable
- Some stations are just naturally more variable

But 25.8% is actually decent for this type of problem.

### Feature Importance

What matters most for predictions? Based on XGBoost:
1. **Temperature** (17.6%) - The biggest factor by far
2. **Month** (14.2%) - Seasonal patterns matter
3. **Nearby area activity** (11.8%) - Location quality
4. **Day of week** (10.3%) - Weekend vs weekday patterns

Weather and time patterns are the main drivers of bike demand.

### What's Next

Our next steps:
1. Try to fix the total volume overestimation issue (maybe add a calibration step)
2. Test different model architectures (maybe a simpler model would generalize better)
3. Start working on the spatial prediction (predicting demand for locations without existing stations)
4. Add more features if we can find useful ones

## Current Status

**What's working**:
- Data pipeline is solid and automated
- We have clean, processed data ready to use
- Two working models with reasonable performance
- Good visualizations that explain the patterns

**Challenges we faced**:
- Getting weather data took way longer than expected
- Balancing model complexity vs overfitting is tricky
- The data is noisy at the daily station level

**Code**: All code is in the GitHub repo, organized into:
- `src/data_collection/` - Scripts to download data
- `src/preprocessing/` - Data cleaning and feature engineering
- `src/models/` - Baseline and XGBoost models
- `src/visualization/` - All plots and charts
- `results/` - Figures and metrics (large prediction files not included)

