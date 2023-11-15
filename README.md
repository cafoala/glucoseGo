# glucoseGo
A machine-learning derived heatmap for predicting hypoglycemia risk during exercise for people with type 1 diabetes

## Overview

This repository contains a series of Jupyter Notebooks organized into three main folders: **Preprocessing**, **Model Building**, and **Explainability**. The project focuses on predicting hypoglycemia risk during exercise in individuals with Type 1 Diabetes using machine learning techniques.
## Folder Structure
### FOLDER 1: Preprocessing

1. **Preprocessing of EXTOD 101 for ML:** Prepares the EXTOD 101 data, consisting of 35 individuals with 8 weeks of data, for machine learning analysis.

2. **Preprocessing EXTOD Education Dataset:** Processes data from the EXTOD education pilot study, involving 106 participants, for machine learning.

3. **Preprocessing JAEB T1-DEXI Dataset:** Similar to the EXTOD education dataset, this notebook prepares the JAEB T1-DEXI data for analysis.

4. **Preprocessing JAEB T1-DEXIP Dataset:** Preprocesses data from the JAEB T1-DEXIP study, aligning with the overall project's methodology.

5. **Target Creation:** Develops target variables for machine learning and statistical analysis, focusing on hypo- and hyper-glycemia during and after exercise.

6. **Final Preparation for Machine Learning:** Finalizes data for model training and validation, ensuring quality and proper formatting.

7. **Analysis of Participant Characteristics in Exercise Studies:** Provides a comprehensive analysis of participant characteristics from various exercise studies.

### FOLDER 2: Model Building

1. **Forward Feature Selection:** Implements a model-based approach to incrementally add features that improve model performance, specifically using ROC Area Under Curve.

2. **Run ML Models:** Executes Logistic Regression and XGBoost models on both full and reduced sets of features, comparing performance and visualizing results.

3. **Creating Figures:** Generates, visualizes, and analyzes results from various machine learning models through informative plots.

4. **Heatmap (Contour Plot) of Model:** Visualizes the model's predicted probability changes across starting glucose levels and exercise duration in a heatmap format.

5. **Performance Evaluation of XGBoost Models on a Hold-Out Dataset:** Assesses and compares the predictive performance of two XGBoost models on a 10% hold-out dataset.

### FOLDER 3: Explainability

1. **SHAP Analysis:** Focuses on explaining machine learning models using SHAP values to understand the importance and impact of different features.

2. **Subgroup Analysis of Two-Featured Model:** Conducts a detailed subgroup analysis of a two-featured XGBoost model, exploring its performance across various patient profiles.

3. **Calibration Curves:** Assesses and visualizes calibration curves for binary classification models to evaluate their reliability.

4. **Learning Curves:** Generates and visualizes learning curves to understand the relationship between training set size and model performance.

## General Information

* Each notebook contains a detailed explanation of its objectives, methodology, and the results obtained.
* The notebooks are designed to provide a comprehensive understanding of the process of developing and evaluating machine learning models for predicting hypoglycemia risk in Type 1 Diabetes patients.

## Data Privacy Note

Due to privacy concerns and the sensitive nature of the medical data used in this project, the datasets are not publicly available in this repository. However, we may be able to share some anonymized data upon request. Please contact the project maintainers for more information.


## Contributing

Feel free to contribute to this project by suggesting improvements, reporting bugs, or submitting pull requests. Please read CONTRIBUTING.md for guidelines on how to contribute.
License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Contact

For any queries or further information, please contact the project maintainers.