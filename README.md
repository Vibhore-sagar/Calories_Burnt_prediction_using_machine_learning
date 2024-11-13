# Calories Burnt Prediction Model

This project demonstrates a machine learning approach to predict the number of calories burnt during exercise based on various physiological factors. It leverages data pre-processing, exploratory data analysis, and an XGBoost regressor model to build and evaluate a predictive model.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
  
## Overview
In this project, we analyze a dataset containing information about exercise activities and corresponding calorie burn counts. After loading and merging the data, we perform exploratory data analysis (EDA) and apply feature engineering to prepare it for training. An XGBoost Regressor model is then used to predict calories burnt based on features like age, gender, height, and weight.

## Dataset
The data used for this project consists of:
1. `exercise.csv` - Data related to exercise activities.
2. `calories.csv` - Corresponding calorie counts for each exercise.

These datasets are merged based on a common identifier and processed to ensure data quality.

## Installation
To run this project, you need the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

Install the required libraries using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost 
```
## Model Training and Evaluation
1. **Data Preprocessing**: Null values are checked, and non-numeric data (e.g., gender) is converted into numerical format.
2. **Exploratory Data Analysis**: Includes distribution plots for each feature and a correlation heatmap to understand relationships in the data.
3. **Feature Selection**: Relevant features are selected for training, excluding unnecessary columns such as `User_ID`.
4. **Model Training**: An XGBoost Regressor model is trained on the processed data, using `train_test_split` to create training and test datasets.
5. **Evaluation**: Model performance is evaluated on the test dataset using Mean Absolute Error (MAE), a metric that indicates the average absolute difference between predicted and actual calorie values.

## Results
The trained model achieves a Mean Absolute Error (MAE) on the test dataset, suggesting the average deviation between the predicted and actual calorie values. Lower MAE values indicate better model performance, with closer alignment to the actual calorie counts.
