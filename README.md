# Forest Cover Type Classification

This project focuses on classifying the forest cover type from cartographic variables. The analysis is performed on the Forest Cover Type dataset from the UCI Machine Learning Repository. Two primary models, Random Forest and XGBoost, are trained and evaluated to predict the cover type.

## Table of Contents
* [Dataset](#dataset)
* [Project Workflow](#project-workflow)
* [Model Performance](#model-performance)
* [Why Random Forest Outperformed XGBoost](#why-random-forest-outperformed-xgboost)
* [How to Run](#how-to-run)

## Dataset

The dataset used in this project is the **Forest Cover Type dataset**. It contains tree observations from four wilderness areas in the Roosevelt National Forest of northern Colorado. This dataset was created by Jock A. Blackard, Dr. Denis J. Dean, and Dr. Charles W. Anderson of the Remote Sensing and GIS Program at Colorado State University.

The dataset includes 581,012 observations of 30x30 meter cells, each with 54 attributes, including:
*   Elevation
*   Aspect
*   Slope
*   Hillshade
*   Soil Type
*   Distance to landmarks (e.g., roads, hydrology)

The goal is to predict the forest cover type, which is one of seven classes.

## Project Workflow

The project follows these steps:

1.  **Exploratory Data Analysis (EDA):** The notebook begins by loading the `covertype.csv` dataset and performing an initial exploration to understand its structure. This includes using `df.describe()`, `df.dtypes`, checking for missing values with `df.isna().sum()`, and looking for duplicates with `df.loc[df.duplicated()]`. The data was found to be pre-cleaned and labeled.

2.  **Data Splitting and Scaling:**
    *   The data is split into features (X) and the target variable (y, "Cover\_Type").
    *   The dataset is then divided into training and testing sets with an 80/20 split.
    *   StandardScaler is used to scale the features, which is a common practice for many machine learning algorithms.

3.  **Model Training and Evaluation:**
    *   **Random Forest:** A `RandomForestClassifier` is implemented. `RandomizedSearchCV` is used for hyperparameter tuning to find the best combination of parameters like `n_estimators`, `max_depth`, etc. The model with the best parameters is then evaluated on the test set.
    *   **XGBoost:** An `XGBClassifier` is also trained. Similar to the Random Forest, `RandomizedSearchCV` is employed to find the optimal hyperparameters. The target labels are adjusted to start from 0, as required by XGBoost for multi-class classification. The model is then evaluated on the test data.

4.  **Results and Comparison:** The performance of both models is assessed using accuracy, precision, recall, f1-score, and confusion matrices.

## Model Performance

The following table summarizes the performance of the two models on the test set:

| Model | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.9594** | **0.9498** | **0.9165** | **0.9320** |
| XGBoost | 0.8927 | 0.91 | 0.86 | 0.88 |

As shown, the Random Forest model outperformed the XGBoost model across all major evaluation metrics.

## Why Random Forest Outperformed XGBoost

While XGBoost is often a top-performing model, in this specific case, the Random Forest classifier achieved higher accuracy. Several factors could contribute to this outcome:

*   **Data Characteristics:** The dataset is large, clean, and well-structured, which are ideal conditions for Random Forest to perform well.
*   **Hyperparameter Tuning:** Random Forest is generally less sensitive to hyperparameter tuning compared to XGBoost. It's possible that with more extensive tuning, the XGBoost model's performance could be improved.
*   **Model Architecture:** Random Forest builds each decision tree independently, which can make it more robust to certain types of data and less prone to overfitting on this particular dataset. In contrast, XGBoost builds trees sequentially, with each tree correcting the errors of the previous ones, which can sometimes lead to overfitting if not carefully regularized.

## How to Run

To replicate this analysis, you can run the `Forest-Cover-Type-Classification.ipynb` notebook in an environment with the necessary Python libraries installed. The notebook is designed to be executed in a Kaggle environment with a GPU accelerator (NVIDIA Tesla T4) to speed up the training of the XGBoost model.

**Dependencies:**
*   pandas
*   numpy
*   matplotlib
*   seaborn
*   scikit-learn
*   xgboost
