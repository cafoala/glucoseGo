import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import warnings
import scipy.stats
from sklearn.metrics import roc_auc_score, brier_score_loss, balanced_accuracy_score

warnings.filterwarnings('ignore')


def calculate_coefficient(coeffs, column_names):
    """
    Visualizes the top 25 features based on their weights in a logistic regression model.

    Parameters:
    - coeffs: Array-like, the coefficients of the model
    - column_names: List, the names of the features corresponding to the coefficients

    Returns:
    - log_weights: DataFrame, contains feature weights and their absolute values
    - fig: Matplotlib Figure object
    """
    log_weights = pd.DataFrame(coeffs, columns=column_names).mean().reset_index()
    log_weights.columns = ['feature', 'weights']
    # Create dataframe with feature names and weights
    #log_weights = pd.DataFrame(coeffs, columns=['weights'], index=column_names)
    
    # Calculate absolute weights and sort them
    log_weights['abs_weight'] = abs(log_weights['weights'])
    log_weights.sort_values('abs_weight', ascending=False, inplace=True)

    # Set up figure for visualization
    fig, ax = plt.subplots(figsize=(8,8))

    # Get top 25 labels and values
    labels = log_weights['feature'].values[0:25]
    pos = np.arange(len(labels))
    val = log_weights['weights'].values[0:25]

    # Create bar plot
    ax.bar(pos, val)
    ax.set_ylabel('Feature weight (standardised features)')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, ha="right", rotation_mode="anchor")
    plt.suptitle('Weights of features')
    plt.tight_layout()

    return log_weights, fig


def mean_shap(list_shap_values):
    """
    Calculate the mean SHAP values across a list of SHAP results.

    Parameters:
    - list_shap_values: List of SHAP value arrays/matrices

    Returns:
    - shap_values: Mean SHAP values
    """
    shap_values = list_shap_values[0]
    number_of_splits = len(list_shap_values)
    for i in range(1, number_of_splits):
        shap_values = shap_values + list_shap_values[i]
    shap_values = shap_values/number_of_splits
    return shap_values



### Subgroup analysis ###

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    #return np.round(m, 2), f'{np.round(m, 2)} ({np.round(m-h, 2)}, {np.round(m+h,2)})'
    return m, m-h, m+h


def return_frame_cis(df):
    print(df)
    return pd.DataFrame([[df.bouts.iloc[0], df.participants.iloc[0], 
                          #df.ratio_hypos.iloc[0], 
                          #df.ratio_predicted_hypos.iloc[0],
                          mean_confidence_interval(df.ratio_hypos)[0],
                          mean_confidence_interval(df.ratio_hypos)[1]*100,
                          mean_confidence_interval(df.ratio_hypos)[2]*100,
                          mean_confidence_interval(df.ratio_predicted_hypos)[0],
                          mean_confidence_interval(df.ratio_predicted_hypos)[1]*100,
                          mean_confidence_interval(df.ratio_predicted_hypos)[2]*100, 
                          mean_confidence_interval(df.roc)[0],
                          mean_confidence_interval(df.roc)[1],
                          mean_confidence_interval(df.roc)[2],
                          mean_confidence_interval(df.bac)[0],	
                          mean_confidence_interval(df.bac)[1],
                          mean_confidence_interval(df.bac)[2],
                          mean_confidence_interval(df.mae)[0],
                          mean_confidence_interval(df.mae)[1],
                          mean_confidence_interval(df.mae)[2],
                          
                          ]],  columns=['bouts', 'participants',# 'rate_hypos', 'ratio_predicted_hypos',
                                        'rate_hypos', 'rate_hypos-','rate_hypos+',
                                        'ratio_predicted_hypos','ratio_predicted_hypos-','ratio_predicted_hypos+',
                                        'roc','roc-','roc+', 
                                        'bac','bac-', 'bac+',
                                        'mae', 'mae-', 'mae+'])

def accuracy(df, y, probs_col):
    try:
        roc = roc_auc_score(df[y], df[probs_col])
    except:
        roc = np.nan
    threshold = 0.1
    try:
        df['y_pred'] = (df[probs_col] > threshold).astype(int)
        bac = brier_score_loss(df[y], df['y_pred'])
        ratio_hypos = df[y].mean()
        ratio_predicted_hypos = df[probs_col].mean()
        mae = (df[y]-df[probs_col]).mean()*100
    except:
        bac=np.nan
        ratio_hypos= np.nan
        ratio_predicted_hypos=np.nan
        mae =np.nan
    return pd.DataFrame([[roc,
                          bac,
                          mae, 
                          ratio_hypos,
                          ratio_predicted_hypos]],
                          columns=['roc', 'bac', 'mae', 'ratio_hypos', 'ratio_predicted_hypos']) #'bouts', 'participants', 'ratio hypos',


def group(df, foldname, y, probs_col):
    fold_results = df.groupby(foldname).apply(lambda x: accuracy(x, y, probs_col))
    fold_results['bouts'] = df.shape[0]
    fold_results['participants'] = len(df.ID.unique())

    return fold_results


def subgroup_analysis(df, subgroup_colname, y_colname='y', y_probas_colname='probas_xgb_two', foldname='probas_xgb_two_fold'):
    if subgroup_colname is None:
            fold_results = df.groupby([foldname]).apply(lambda x: accuracy(x, y_colname, y_probas_colname)).reset_index().drop(columns='level_1')
            fold_results['bouts'] = df.shape[0]

            fold_results['participants'] = len(df.ID.unique())
            #fold_results['ratio_hypos'] = df[y_colname].mean()
            #fold_results['ratio_predicted_hypos'] = df[y_probas_colname].mean()

            mean_results = return_frame_cis(fold_results)
            mean_results['subgroup'] = 'all'

            return mean_results
    else:
        fold_results = df.groupby(subgroup_colname).apply(lambda x: group(x, foldname, y_colname, y_probas_colname)).round(3).reset_index().drop(columns='level_2')
        #f, p = calculate_p(fold_results, subgroup_colname)    
        mean_results = fold_results.groupby(subgroup_colname).apply(lambda x: return_frame_cis(x)).reset_index().drop(columns='level_1')
        #mean_results['f-stat'] = f
        #mean_results['p-val'] = p
    mean_results.rename(columns={subgroup_colname: 'subgroup'}, inplace=True)
    return mean_results