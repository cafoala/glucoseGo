import hyperopt
from functools import partial
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import sklearn.metrics
import xgboost as xgb
import optuna
import numpy as np
np.random.seed(42)
import random
random.seed(42)


#SEED = 108
N_FOLDS = 5
CV_RESULT_DIR = "./xgboost_cv_results"


def objective(params, Model, fit_params, X, y, scorer): #  
    '''
    Objective function for hyperopt to minimise.
    
    Parameters:
        params (dict): A selection of the parameter space to fit
        Model (machine learning model): A machine learning model with a fit function and probability prediction
        fit_params (dict): fit params for the model e.g. early stopping, verbose
        X (np array or pandas df): training dataset to fit model to
        y (np array or pandas df): target for training data
        scorer (str): how to generate the loss for the model (e.g. 'roc_auc', 'accuracy')
        
    Returns:
        dict: Results including the loss, status and params fitted
    '''
    # Fit model with parameters
    model = Model(**params)

    cv_score = np.mean(cross_val_score(model, X, y, 
                                       cv = 10, 
                                       n_jobs = -1, 
                                       fit_params = fit_params, 
                                       scoring = scorer,
                                       ))
    
    # hyperopt minimizes the loss, hence the minus sign behind cv_score
    return {'loss': -cv_score, 'status': hyperopt.STATUS_OK, 'params': params}


def hyperopt_tune(Model, param_space, fit_params, X_train, y_train, scorer, n_iter):
    '''
    Tunes ML model using hyperopt
    
    Parameters:
        Model (machine learning model): A machine learning model with a fit function and probability prediction
        params (dict): A selection of the parameter space to fit
        fit_params (dict): fit params for the model e.g. early stopping, verbose
        X_train (np array or pandas df): training dataset to fit model to
        y_train (np array or pandas df): target for training data
        scorer (str): how to generatse the loss for the model (e.g. 'roc_auc', 'accuracy')
        n_iter (int): number of iterations to run the optimization function
        
    Returns:
        dict: Results including the loss, status and params fitted
    
    '''
    # create a partial function to bind the rest of the arguments we want to pass to objective
    obj = partial(objective, Model=Model, 
                  fit_params=fit_params, 
                  X = X_train, y = y_train, 
                  scorer=scorer)

    # A trials object that will store the results of all iterations
    trials = hyperopt.Trials()
    hyperopt.fmin(fn = obj, space = param_space, 
                  algo = hyperopt.tpe.suggest, 
                  max_evals = n_iter, 
                  trials = trials)
    # returns the values of parameters from the best trial
    return trials.best_trial['result']['params']


def xgb_objective(trial, X, y):
    '''
    Objective function for xgboost using optuna.
    
    Parameters:
    - trial: Optuna trial object.
    - X (DataFrame): Features.
    - y (Series): Target variable.

    Returns:
    - float: AUC score of the cross-validated xgboost model.
    '''
    dtrain = xgb.DMatrix(X, label=y)

    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_jobs": -1,
        "n_estimators" : trial.suggest_int('n_estimators', 1, 1000),
        #"booster": trial.suggest_categorical("booster", ["gbtree",  "dart", "gblinear",]), #
        "max_depth": trial.suggest_int("max_depth", 1, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        #"lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        #"alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "eta": trial.suggest_float('eta', 0.05, 0.3),
        "random_state": 42,
        # New ones
        'learning_rate':trial.suggest_loguniform('learning_rate',0.005,0.5),
        'reg_alpha':trial.suggest_int('reg_alpha', 0, 5),
        'reg_lambda':trial.suggest_int('reg_lambda', 0, 5),
        'gamma':trial.suggest_int('gamma', 0, 5),
    }
    # Initiate a pruner to feed to the study
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-auc")
    history = xgb.cv(param, dtrain, 
                        num_boost_round=100,
                        callbacks=[pruning_callback], 
                        nfold=10,
                        stratified=True,#False?
                        early_stopping_rounds=100,
                        verbose_eval=False,
                        seed=42)

    # Set n_estimators as a trial attribute; Accessible via study.trials_dataframe().
    trial.set_user_attr("n_estimators", len(history))

    # Extract the best score.
    mean_auc = history["test-auc-mean"].values[-1]

    return mean_auc

def objective_trial(trial,X,y):
    '''
    Objective function for xgboost model optimization using optuna.
    
    Parameters:
    - trial: Optuna trial object.
    - X (DataFrame): Features.
    - y (Series): Target variable.

    Returns:
    - float: Mean accuracy score of the cross-validated xgboost model.
    '''
    param = {
            "n_estimators" : trial.suggest_int('n_estimators', 1, 1000),
            'max_depth':trial.suggest_int('max_depth', 2, 25),
            'reg_alpha':trial.suggest_int('reg_alpha', 0, 5),
            'reg_lambda':trial.suggest_int('reg_lambda', 0, 5),
            'min_child_weight':trial.suggest_int('min_child_weight', 0, 5),
            'gamma':trial.suggest_int('gamma', 0, 5),
            'learning_rate':trial.suggest_loguniform('learning_rate',0.005,0.5),
            'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.1,1,0.01),
            'nthread' : -1, 
            'tree_method': 'gpu_hist'
            }
    
    model = xgb.XGBClassifier(**param)
    
    return cross_val_score(model, X, y, cv=3).mean()