# Crime Analysis and Prediction Project

## Overview

This project aims to analyze and predict crimes in Chicago using advanced data science techniques, including classification and clustering. The project tackles the problem of crime classification and identifies crime hotspots to optimize resource allocation for crime prevention. The solution was developed as part of the IML Hackathon.

## Objective

This project addresses the "Help the Chicago Police Prevent Crime" challenge. The aim is to assist the Chicago Police Department in predicting the type, time, and location of crimes, enabling better resource allocation and crime prevention.

The challenge involves two main tasks:

1. **Crime Type Classification**: Developing a machine learning system to predict the type of crime (from 5 predefined classes) based on various features such as location, time, and other incident details.

2. **Crime Prevention**: Identifying 30 optimal locations and times for deploying police cars each day to maximize crime prevention, considering spatial and temporal proximity to potential crimes.

## Features

1. **Crime Classification**:

   - Implemented a classification model using an ensemble random forest to predict crime types based on preprocessed and selected features.

2. **Crime Clustering**:

   - Utilized clustering to identify crime hotspots.
   - Crimes were grouped by weekdays, and 100 clusters were created for each day.
   - Centroids were calculated and normalized based on time and spatial dimensions to ensure balanced distribution.

3. **Data Preprocessing**:

   - Performed feature selection to retain only the most relevant features (e.g., location and blocks).
   - Handled missing data by imputing values based on similar records.
   - Added dummy variables and normalized features for improved model performance.

4. **Visualization**:

   - Visualized crime distributions and clusters using x-y coordinates and time dimensions.

## File Structure

- `classifier.py`: Contains the implementation of the classification model for crime predictions.
- `Clustering.py`: Handles the clustering of crimes into hotspots and determines centroids.
- `Preprocessing.py`: Preprocesses the raw dataset, including feature selection, imputation, and normalization.
- `model.sav`: Serialized model file for saving the trained classification model.
- `Dataset_crimes.csv`: Contains the raw dataset of crimes in Chicago.
- `j_cluster.json`: Stores the generated crime clusters.
- `project.pdf`: Project paper.
