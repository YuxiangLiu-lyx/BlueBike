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

- **Generate a City-Wide Demand Heatmap for Strategic Network Planning**: Moving beyond forecasting for existing stations, our second objective is to build a spatial prediction model to estimate potential bike demand across the entire Boston/Cambridge area. This will allow us to identify optimal locations for network expansion and consolidation.
  - **Prediction Target**: The potential number of bike departures per hour within a given geographical grid cell (e.g., 100m x 100m).
  - **Feature Engineering**: To predict demand in areas without existing stations, we will create a rich geospatial feature set for each grid cell, including:
    - Proximity to public transit hubs (subway, bus stops)
    - Density of points-of-interest (e.g., restaurants, offices, parks)
    - Local demographic data (e.g., population density)
    - Characteristics of the local bike network (e.g., availability of protected bike lanes)
  - **Expected Outcome**: The model's output will be a city-wide demand heatmap, visually highlighting untapped "hotspots" (ideal locations for new stations) and demand "coldspots" (where existing stations may be underutilized and could be considered for relocation or removal).

- **Propose Novel Operational Strategies** (if time permitted): As a secondary objective, we will explore how the model could inform new strategies, such as implementing dynamic pricing during off-peak hours or offering user incentives to help redistribute bikes from over-supplied to under-supplied stations.


## 3. Data Collection Strategy

This project will utilize two primary data sources: Bluebikes' official ridership data and historical weather data for the Boston area.

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
- **Methodology**: We will parse these files to extract the precise latitude and longitude of all subway and bus stops.

#### Urban Infrastructure & Points-of-Interest (POIs)

- **Source**: OpenStreetMap (OSM), the leading open-source global map database. Data will be accessed via the OSMnx library. Link: https://osmnx.readthedocs.io/en/stable/
- **Methodology**: Using the Python library OSMnx, we will programmatically download data on the locations of key POIs (e.g., restaurants, offices, universities, parks) and the city's bike lane network.

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

