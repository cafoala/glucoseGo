import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)
import shap
from numpy import argmax
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, log_loss, brier_score_loss, roc_auc_score, roc_curve, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
import tune
import optuna
import hyperopt
import matplotlib.pyplot as plt
import pickle
from scipy.stats import sem


def standardise_data(X_train, X_test):
    """
    Converts all data to a similar scale.
    Standardisation subtracts mean and divides by standard deviation
    for each feature.
    Standardised data will have a mena of 0 and standard deviation of 1.
    The training data mean and standard deviation is used to standardise both
    training and test set data.
    """
    
    # Initialise a new scaling object for normalising input data
    sc = StandardScaler() 

    # Set up the scaler just on the training set
    sc.fit(X_train)

    # Apply the scaler to the training and test sets
    train_std=sc.transform(X_train)
    test_std=sc.transform(X_test)
    
    return train_std, test_std


def calculate_accuracy(observed, predicted):
    
    """
    Calculates a range of accuracy scores from observed and predicted classes.
    
    Takes two list or NumPy arrays (observed class values, and predicted class 
    values), and returns a dictionary of results.
    
     1) observed positive rate: proportion of observed cases that are +ve
     2) Predicted positive rate: proportion of predicted cases that are +ve
     3) observed negative rate: proportion of observed cases that are -ve
     4) Predicted negative rate: proportion of predicted cases that are -ve  
     5) accuracy: proportion of predicted results that are correct    
     6) precision: proportion of predicted +ve that are correct
     7) recall: proportion of true +ve correctly identified
     8) f1: harmonic mean of precision and recall
     9) sensitivity: Same as recall
    10) specificity: Proportion of true -ve identified:        
    11) positive likelihood: increased probability of true +ve if test +ve
    12) negative likelihood: reduced probability of true +ve if test -ve
    13) false positive rate: proportion of false +ves in true -ve patients
    14) false negative rate: proportion of false -ves in true +ve patients
    15) true positive rate: Same as recall
    16) true negative rate
    17) positive predictive value: chance of true +ve if test +ve
    18) negative predictive value: chance of true -ve if test -ve
    
    """
    
    # Converts list to NumPy arrays
    if type(observed) == list:
        observed = np.array(observed)
    if type(predicted) == list:
        predicted = np.array(predicted)
    
    # Calculate accuracy scores
    observed_positives = observed == 1
    observed_negatives = observed == 0
    predicted_positives = predicted == 1
    predicted_negatives = predicted == 0
    
    true_positives = (predicted_positives == 1) & (observed_positives == 1)
    
    false_positives = (predicted_positives == 1) & (observed_positives == 0)
    
    true_negatives = (predicted_negatives == 1) & (observed_negatives == 1)
    
    false_negatives = (predicted_negatives == 1) & (observed_negatives == 0)
    
    accuracy = np.mean(predicted == observed)
    
    precision = (np.sum(true_positives) /
                 (np.sum(true_positives) + np.sum(false_positives)))
        
    recall = np.sum(true_positives) / np.sum(observed_positives)
    
    sensitivity = recall
    
    f1 = 2 * ((precision * recall) / (precision + recall))
    
    specificity = np.sum(true_negatives) / np.sum(observed_negatives)
    
    positive_likelihood = sensitivity / (1 - specificity)
    
    negative_likelihood = (1 - sensitivity) / specificity
    
    false_positive_rate = 1 - specificity
    
    false_negative_rate = 1 - sensitivity
    
    true_positive_rate = sensitivity
    
    true_negative_rate = specificity
    
    positive_predictive_value = (np.sum(true_positives) / 
                                 np.sum(observed_positives))
    
    negative_predictive_value = (np.sum(true_negatives) / 
                                  np.sum(observed_negatives))
    
    # Create dictionary for results, and add results
    results = dict()
    
    results['observed_positive_rate'] = np.mean(observed_positives)
    results['observed_negative_rate'] = np.mean(observed_negatives)
    results['predicted_positive_rate'] = np.mean(predicted_positives)
    results['predicted_negative_rate'] = np.mean(predicted_negatives)
    results['accuracy'] = accuracy
    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1
    results['sensitivity'] = sensitivity
    results['specificity'] = specificity
    results['positive_likelihood'] = positive_likelihood
    results['negative_likelihood'] = negative_likelihood
    results['false_positive_rate'] = false_positive_rate
    results['false_negative_rate'] = false_negative_rate
    results['true_positive_rate'] = true_positive_rate
    results['true_negative_rate'] = true_negative_rate
    results['positive_predictive_value'] = positive_predictive_value
    results['negative_predictive_value'] = negative_predictive_value
    
    return results


def train_xgb(X_train, y_train, n_trials=50):
    """
    Train an XGBoost classifier using Optuna for Bayesian hyperparameter tuning.

    Parameters:
    - X_train: DataFrame, training feature set
    - y_train: Series or array-like, training target variable
    - n_trials: int, optional, number of trials for hyperparameter optimization. Default is 50.

    Returns:
    - model: Fitted XGBoost classifier
    """
    # Optimize using Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: tune.xgb_objective(trial, X_train, y_train), n_trials=n_trials)
    best_params = study.best_params
    
    # Fit XGBoost model with the best parameters
    model = XGBClassifier(**best_params, seed=42)
    model.fit(X_train, y_train)

    return model

def train_lr(X_train, y_train, max_evals=60):
    """
    Train a logistic regression model using Hyperopt for Bayesian hyperparameter tuning.

    Parameters:
    - X_train: DataFrame, training feature set
    - y_train: Series or array-like, training target variable
    - max_evals: int, optional, maximum number of evaluations for hyperparameter tuning. Default is 60.

    Returns:
    - model: Fitted Logistic Regression model
    """
    # Define hyperparameter space for logistic regression
    params = {
        'penalty': hyperopt.hp.choice('penalty', ['l2', 'l1']),
        'C': hyperopt.hp.loguniform('C', -4, 4),
        'solver': hyperopt.hp.choice('solver', ['lbfgs', 'liblinear']),
        'random_state': 42
    }

    # Optimize hyperparameters using Hyperopt
    best_params = tune.hyperopt_tune(LogisticRegression, params, None, X_train, y_train, 'roc_auc', max_evals)
    
    # Fit logistic regression with the best parameters
    model = LogisticRegression(**best_params)
    model.fit(X_train, y_train)

    return model


def calculate_shap(lr, model, X, columns=None):
    """
    Calculate SHAP values for the given model and data.

    Parameters:
    - lr: bool, True if model is logistic regression, False otherwise (for XGBoost)
    - model: trained model instance (either logistic regression or XGBoost)
    - X: DataFrame or numpy array, input data for which to calculate SHAP values
    - columns: list, optional, column names for the output. Used only for XGBoost
    
    Returns:
    - shap_values: SHAP values for the model predictions
    """
    if lr:
        background = shap.maskers.Independent(X, max_samples=100)
        explainer = shap.explainers.Linear(model, background)
    else:
        explainer = shap.TreeExplainer(model, output_names=columns)
    
    shap_values = explainer(X)
    return shap_values


def k_fold_accuracies(X, y, strat_col, lr=True, features='two'):
    '''
    Perform k-fold cross-validation and return performance metrics.
    
    Parameters:
    - X: pd.DataFrame - Feature set
    - y: pd.Series - Target variable
    - strat_col: str - Column name for stratification
    - lr: bool - If True, Logistic Regression is used, otherwise XGB.
    - features: str - Identifier for the feature set
    
    Returns:
    - pd.DataFrame with performance metrics for each fold
    '''
    columns = X.columns

    X_np = X.values
    y_np = y.values
    
    # Set up k-fold training/test splits
    number_of_splits = 10
    skf = StratifiedKFold(n_splits = number_of_splits, shuffle=True, random_state=42)
    splits = skf.split(X_np, strat_col)

    # Set up thresholds
    thresholds = np.arange(0, 1.01, 0.01)

    overall_results = []
    # Create arrays for overall results (rows=threshold, columns=k fold replicate)

    # Test and predicted for each fold
    test_sets = []
    predicted_probas = []
    observed = []
    
    # Explainability
    shap_values = []
    coeffs = []

    #Hyperparameters
    hyperparameters = []

    # Loop through the k-fold splits
    loop_index = 0
    for train_index, test_index in splits:

        # Create lists for k-fold results
        threshold_accuracy = []
        threshold_precision = []
        threshold_recall = []
        threshold_f1 = []
        threshold_predicted_positive_rate = []
        threshold_observed_positive_rate = []
        threshold_true_positive_rate = []
        threshold_false_positive_rate = []
        threshold_specificity = []
        threshold_balanced_accuracy = []

        # Get X and Y train/test
        X_train, X_test = X_np[train_index], X_np[test_index]
        y_train, y_test = y_np[train_index], y_np[test_index]

        # Set up and fit model (n_jobs=-1 uses all cores on a computer)
        if lr:
            model = train_lr(X_train, y_train)
            with open(f"../../results/models/lr_{features}_{loop_index}", "wb") as fp:   
                #Pickling 
                pickle.dump(model, fp)
            
        else:
            model = train_xgb(X_train, y_train)
            model.fit(X_train, y_train)
            with open(f"../../results/models/xgb/{features}_{loop_index}", "wb") as fp:   
                #Pickling 
                pickle.dump(model, fp)
        
        hyperparameters.append(model.get_params())

        # Get probability of hypo and non-hypo
        probabilities = model.predict_proba(X_test)
        # Take just the hypo probabilities (column 1)
        predicted_proba = probabilities[:,1]
        predicted_probas.append(predicted_proba)
        # Store the observed result
        observed.append(y_test)
        # Add probabilities to test set for future analysis
        test_sets.append(test_index)
        
        # Explainability
        interpretation = calculate_shap(lr, model, X, columns)
        shap_values.append(interpretation)

        # Coeffs
        if lr==True:
            coeffs.append(model.coef_[0])

        # Loop through increments in probability of survival
        for cutoff in thresholds: #  loop 0 --> 1 on steps of 0.1
            # Get whether passengers survive using cutoff
            predicted_survived = predicted_proba >= cutoff
            # Call accuracy measures function
            accuracy = calculate_accuracy(y_test, predicted_survived)
            # Add accuracy scores to lists
            threshold_accuracy.append(accuracy['accuracy'])
            threshold_precision.append(accuracy['precision'])
            threshold_recall.append(accuracy['recall'])
            threshold_f1.append(accuracy['f1'])
            threshold_predicted_positive_rate.append(
                    accuracy['predicted_positive_rate'])
            threshold_observed_positive_rate.append(
                    accuracy['observed_positive_rate'])
            threshold_true_positive_rate.append(accuracy['true_positive_rate'])
            threshold_false_positive_rate.append(accuracy['false_positive_rate'])
            threshold_specificity.append(accuracy['specificity'])
            threshold_balanced_accuracy.append((accuracy['specificity']+accuracy['recall'])/2)
        
        # Select threshold with the best balance between sensitivity and specificity
        ix = argmax(threshold_balanced_accuracy)
 
        # Calculate ROC AUC
        roc_auc = roc_auc_score(y_test, predicted_proba)
        mae = mean_absolute_error(y_test, predicted_proba)
        # Calculate brier and logloss
        brier = brier_score_loss(y_test, predicted_proba)
        logloss = log_loss(y_test, predicted_proba)

        overall_results.append([roc_auc, mae, logloss, brier, thresholds[ix], threshold_accuracy[ix], threshold_precision[ix],
                        threshold_recall[ix], threshold_f1[ix], threshold_predicted_positive_rate[ix],
                        threshold_observed_positive_rate[ix], threshold_true_positive_rate[ix], 
                        threshold_false_positive_rate[ix], threshold_specificity[ix], 
                        threshold_balanced_accuracy[ix]])
        # Increment loop index
        loop_index += 1
    # Add results at this threshold to k-fold results
    k_fold_results = pd.DataFrame(overall_results)
    k_fold_results.columns=['roc_auc', 'mae', 'logloss', 'brier', 'threshold','accuracy', 'precision', 'recall', 'f1','predicted_positive_rate', 'observed_positive_rate', 'tpr','fpr','specificity','balanced_accuracy']
    
    return k_fold_results, test_sets, predicted_probas, observed,  shap_values, coeffs, hyperparameters


def add_mean_to_df(mean_df, k_fold_results, model_name, feature_set):
    '''
    Append the mean of the k-fold cross-validation results to the provided dataframe.
    
    Parameters:
    - mean_df: pd.DataFrame - The dataframe to which the mean results will be appended.
    - k_fold_results: pd.DataFrame - Results of the k-fold cross-validation.
    - model_name: str - Name of the model for which k-fold cross-validation was performed.
    - feature_set: str - Identifier for the set of features used.
    
    Returns:
    - pd.DataFrame - Updated dataframe with the mean k-fold results appended.
    '''
    
    # Get the mean values from the k-fold results
    mean = k_fold_results.describe().loc['mean']
    
    # Add model name and feature set identifier to the mean values
    mean['model'] = model_name
    mean['features'] = feature_set
    
    # Append the mean values to the provided dataframe
    mean_df = mean_df.append(mean)
    
    return mean_df


def add_proba_col(df, test_sets_index, predicted_probas, colname):
    '''
    Append a column with predicted probabilities and their associated fold to the dataframe.
    
    Parameters:
    - df: pd.DataFrame - Dataframe to which the predicted probabilities and fold number will be appended.
    - test_sets_index: list of list[int]] - Indices of test sets for each fold.
    - predicted_probas: list[np.array] - Predicted probabilities for each test set.
    - colname: str - Name of the new column to add for probabilities.
    
    Returns:
    - pd.DataFrame - Updated dataframe with the new columns appended.
    '''
    
    # Initialize new columns for predicted probabilities and fold numbers
    df[[colname, colname+'_fold']] = -1
    
    # Loop through each fold
    for i in range(len(test_sets_index)):
        # Assign predicted probabilities to the appropriate indices in the dataframe
        df.loc[test_sets_index[i], colname] = predicted_probas[i]
        
        # Assign fold numbers to the appropriate indices in the dataframe
        df.loc[test_sets_index[i], colname+'_fold'] = i
        
    return df


def calculate_confidence_intervals(df, metrics):
    """
    Calculate the mean and 95% confidence interval for specified metrics.
    
    Parameters:
    - df: DataFrame, contains the data for metrics calculation
    - metrics: list, metrics for which to calculate mean and confidence interval
    
    Returns:
    - results: DataFrame, mean and 95% CI for each metric
    """
    results = []
    for met in metrics:
        avg = df[met].mean()
        # Calculate the standard error of the mean
        sem = df[met].std() / np.sqrt(len(df))
        # Calculate the 95% confidence interval
        ci_low = avg - 1.96 * sem
        ci_high = avg + 1.96 * sem
        results.append([met, f"{avg:.2f} ({ci_low:.2f}-{ci_high:.2f})"])
    
    return pd.DataFrame(results, columns=['metric', 'value'])


def k_fold_threshold_curves(ax, observed, predicted_proba, color, linestyle, label):
    """
    Plot the mean ROC curve over multiple k-fold cross-validations along with its 95% confidence interval.
    
    Parameters:
    - ax (matplotlib.axes._subplots.AxesSubplot): The axes object on which the curve will be plotted.
    - observed (list of arrays): List of arrays containing observed binary labels for each fold.
    - predicted_proba (list of arrays): List of arrays containing predicted probabilities for each fold.
    - color (str): Color for the ROC curve.
    - linestyle (str): Linestyle for the ROC curve (e.g., '-', '--').
    - label (str): Label for the ROC curve.
    
    Returns:
    - ax (matplotlib.axes._subplots.AxesSubplot): The axes object with the plotted ROC curve and shaded 95% CI.
    
    The function computes the ROC curve for each fold and then calculates the mean ROC curve. 
    It then plots this mean curve on the provided axes object. Additionally, the function computes 
    the 95% confidence interval for the mean ROC curve and shades this area on the plot.
    """
    # Set up lists for results
    k_fold_tpr = [] # true positive rate
    number_folds = len(observed)
    mean_fpr = np.linspace(0, 1, 100)  # common set of FPR values

    # Loop through k fold predictions and get ROC results 
    for i in range(number_folds):
        # Get fpr, tpr and thresholds for each k-fold from scikit-learn's ROC method
        fpr, tpr, thresholds = roc_curve(observed[i], predicted_proba[i])
        # Interpolate the tpr values for the common fpr values
        k_fold_tpr.append(np.interp(mean_fpr, fpr, tpr))
        k_fold_tpr[-1][0] = 0.0

    mean_tpr = np.mean(k_fold_tpr, axis=0)  # Compute the mean TPR values
    mean_tpr[-1] = 1.0

    # Compute the standard error and derive the upper and lower bounds for a 95% CI
    std_tpr = sem(k_fold_tpr, axis=0)
    tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)

    # Plotting
    ax.plot(mean_fpr, mean_tpr, color=color, linestyle=linestyle, label=label)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=0.2)  # Shading the CI , label='95%CI'

    return ax


def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)


def sens_spec_curve(ax, observed, predicted_proba, color, linestyle):
    """
    Plot the sensitivity-specificity curve over multiple k-fold cross-validations with a shaded region representing
    the 95% confidence interval for sensitivity.
    
    Parameters:
    - ax (matplotlib.axes._subplots.AxesSubplot): The axes object on which the curve will be plotted.
    - observed (list of arrays): List of arrays containing observed binary labels for each fold.
    - predicted_proba (list of arrays): List of arrays containing predicted probabilities for each fold.
    - color (str): Color for the sensitivity-specificity curve.
    - linestyle (str): Linestyle for the sensitivity-specificity curve (e.g., '-', '--').
    
    Returns:
    - ax (matplotlib.axes._subplots.AxesSubplot): The axes object with the plotted sensitivity-specificity curve and shaded 95% CI for sensitivity.
    
    The function computes the sensitivity and specificity for each threshold and each fold. It then calculates the 
    mean sensitivity and specificity across folds. The function plots this mean curve on the provided axes object.
    Additionally, the function computes the 95% confidence interval for the mean sensitivity and shades this area on 
    the plot.
    """
    
    # Lists to store sensitivity and specificity values for each k-fold
    k_fold_sensitivity = []
    k_fold_specificity = []
    number_folds = len(observed)
    
    # Define a range of thresholds from 0 to 1
    thresholds = np.arange(0.0, 1.01, 0.01)

    # Calculate sensitivity and specificity for each fold and each threshold
    for i in range(number_folds):
        obs = observed[i]
        proba = predicted_proba[i]

        sensitivity = []
        specificity = []

        for cutoff in thresholds:
            # Convert predicted probabilities to class labels based on the threshold
            predicted_class = proba >= cutoff
            
            # Calculate the confusion matrix
            matrix = confusion_matrix(obs, predicted_class, labels=[1,0])
            
            # Calculate specificity and sensitivity from the confusion matrix
            spec = matrix[1][1] / (matrix[1][0] + matrix[1][1])
            sens = matrix[0][0] / (matrix[0][0] + matrix[0][1])
            
            sensitivity.append(sens)
            specificity.append(spec)

        k_fold_sensitivity.append(sensitivity)
        k_fold_specificity.append(specificity)

    # Compute the mean sensitivity and specificity across folds
    mean_sensitivity = np.mean(k_fold_sensitivity, axis=0)
    mean_specificity = np.mean(k_fold_specificity, axis=0)
    
    # Calculate standard error for sensitivity
    std_sensitivity = sem(k_fold_sensitivity, axis=0)

    # Calculate the 95% confidence interval bounds for sensitivity
    sens_upper = np.minimum(mean_sensitivity + 1.96 * std_sensitivity, 1)
    sens_lower = np.maximum(mean_sensitivity - 1.96 * std_sensitivity, 0)

    # Plot the mean sensitivity-specificity curve
    ax.plot(mean_sensitivity, mean_specificity, color=color, linestyle=linestyle)

    # Shade the region representing the 95% CI for sensitivity
    ax.fill_betweenx(mean_specificity, sens_lower, sens_upper, color=color, alpha=0.2)

    return ax

