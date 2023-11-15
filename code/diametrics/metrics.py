import copy
import pandas as pd
import numpy as np
from scipy import signal
import warnings
from datetime import timedelta
import statistics
from sklearn import metrics
# ASK MIKE/MICHAEL ABOUT THIS
import _glycemic_events_helper, preprocessing
#import src.diametrics._glycemic_events_helper as _glycemic_events_helper
#import src.diametrics._glycemic_events_dicts as _glycemic_events_dicts

#fift_mins = timedelta(minutes=15)
#thirt_mins = timedelta(minutes=30)

UNIT_THRESHOLDS = {
    'mmol/L': {
        'norm_tight': 7.8,
        'hypo_lv1': 3.9,
        'hypo_lv2': 3,
        'hyper_lv1': 10,
        'hyper_lv2': 13.9
    },
    'mg/dL': {
        'norm_tight': 140,
        'hypo_lv1': 70,
        'hypo_lv2': 54,
        'hyper_lv1': 180,
        'hyper_lv2': 250
    }
}

    
def all_standard_metrics(df, return_df=True, lv1_hypo=None, lv2_hypo=None, lv1_hyper=None, lv2_hyper=None, additional_tirs=None, event_mins=15, event_long_mins=120):
    """
    Calculate standard metrics of glycemic control for glucose data.

    Args:
        df (DataFrame): Input DataFrame containing glucose data.
        return_df (bool, optional): Flag indicating whether to return the results as a DataFrame. Defaults to True.
        lv1_hypo (float, optional): Level 1 hypoglycemia threshold. Defaults to None.
        lv2_hypo (float, optional): Level 2 hypoglycemia threshold. Defaults to None.
        lv1_hyper (float, optional): Level 1 hyperglycemia threshold. Defaults to None.
        lv2_hyper (float, optional): Level 2 hyperglycemia threshold. Defaults to None.
        additional_tirs (list, optional): Additional time in range thresholds. Defaults to None.
        event_mins (int, optional): Duration in minutes for identifying glycemic events. Defaults to 15.
        event_long_mins (int, optional): Duration in minutes for identifying long glycemic events. Defaults to 120.

    Returns:
        DataFrame or dict: Calculated standard metrics as a DataFrame if return_df is True, or as a dictionary if return_df is False.

    Raises:
        Exception: If the input DataFrame fails the data check.

    """
    def run(df, lv1_hypo, lv2_hypo, lv1_hyper, lv2_hyper, additional_tirs, event_mins, event_long_mins):
        if check_df(df):
            results = {}

            # Drop rows with missing time or glucose values
            df = df.dropna(subset=['time', 'glc']).reset_index(drop=True)
            
            # Amount of data available
            #data_suff = data_sufficiency(df)
            #data_suff['Days'] = str(pd.to_datetime(data_suff['End DateTime']) - pd.to_datetime(data_suff['Start DateTime']))
            #results.update(data_suff)
            
            # Average glucose
            avg_glc_result = average_glc(df)
            results.update(avg_glc_result)

            # eA1c
            ea1c_result = ea1c(df)
            results.update(ea1c_result)
            
            # Glycemic variability
            glyc_var = glycemic_variability(df)
            results.update(glyc_var)
            
            # AUC
            auc_result = auc(df)
            results.update(auc_result)
            
            # LBGI and HBGI
            bgi_results = bgi(df)
            results.update(bgi_results)
            
            # MAGE
            mage_results = mage(df)
            results.update(mage_results)

            # Time in ranges
            ranges = time_in_range(df)
            results.update(ranges)

            unique_ranges = unique_time_in_range(df, additional_tirs)
            results.update(unique_ranges)
            
            # New method
            hypos = glycemic_episodes(df, lv1_hypo, lv2_hypo, lv1_hyper, lv2_hyper, event_mins, event_long_mins)
            results.update(hypos)
            
            #if return_df: 
                # Convert to DataFrame
                #results = pd.DataFrame.from_dict([results])

            return results
        
        else:
            return {}
            raise Exception("Data check failed. Please ensure the input DataFrame is valid.")
    
    if 'ID' in df.columns:
        results = df.groupby('ID').apply(lambda group: pd.DataFrame.from_dict(run(group.drop(columns='ID'), lv1_hypo, lv2_hypo, lv1_hyper, lv2_hyper, additional_tirs, event_mins, event_long_mins), orient='index').T).reset_index().drop(columns='level_1')
        return results
    else:    
        results = run(df, lv1_hypo, lv2_hypo, lv1_hyper, lv2_hyper, additional_tirs, event_mins, event_long_mins)
        return results    
    

def check_df(df):
    '''
    Check if the file given is a valid dataframe
    '''
    if not isinstance(df, pd.DataFrame):
        # I want to return this info to user somehow??
        warnings.warn('Not a dataframe')
        return False
    else:
        # drop any null values in the glc column
        df = df.dropna(subset=['time', 'glc'])
        if df.empty:
            warnings.warn('Empty dataframe')
            return False
        else:
            return True


def average_glc(df):
    """
    Calculate the average glucose reading from the 'glc' column in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'glc' column with glucose readings.

    Returns:
        float: The average glucose reading.

    Note:
        - The function uses the 'mean' method from pandas.DataFrame to calculate the average.
        - It returns the average glucose reading as a float value.
    """
    def run(df):
        units = preprocessing.detect_units(df)
        # Calculate the mean of the 'glc' column in the DataFrame
        average = df['glc'].mean()
        return {f'Average glucose ({units})': average}
    
    if 'ID' in df.columns:
        results = df.groupby('ID').apply(lambda group: pd.DataFrame.from_dict(run(group), orient='index').T).reset_index().drop(columns='level_1')
        return results
    else:    
        results = run(df)
        return results


def percentiles(df):
    """
    Calculate various percentiles of glucose readings in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'glc' column with glucose readings.

    Returns:
        dict: A dictionary containing the calculated percentiles of glucose readings.

    Note:
        - The function uses the numpy function np.percentile to calculate the specified percentiles.
        - The percentiles calculated are: 0th, 10th, 25th, 50th (median), 75th, 90th, and 100th.
        - The values are returned as a dictionary with keys representing the percentile labels and values representing the corresponding percentile values.
    """
    def run(df):
        # Calculate the specified percentiles of the 'glc' column in the DataFrame
        percentile_0, percentile_10, percentile_25, percentile_50, percentile_75, percentile_90, percentile_100 = np.percentile(df['glc'], [0, 10, 25, 50, 75, 90, 100])

        # Create a dictionary with the calculated percentiles
        percentiles_dict = {
            'Min. glucose': percentile_0,
            '10th percentile': percentile_10,
            '25th percentile': percentile_25,
            '50th percentile': percentile_50,
            '75th percentile': percentile_75,
            '90th percentile': percentile_90,
            'Max. glucose': percentile_100
        }

        return percentiles_dict
    
    if 'ID' in df.columns:
        results = df.groupby('ID').apply(lambda group: pd.DataFrame.from_dict(run(group), orient='index').T).reset_index().drop(columns='level_1')
        return results
    else:    
        results = run(df)
        return results



def glycemic_variability(df):
    """
    Calculate the glycemic variability metrics for glucose readings in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'glc' column with glucose readings.

    Returns:
        dict: A dictionary containing the calculated glycemic variability metrics.

    Note:
        - The function uses the 'average_glc' function to calculate the average glucose reading.
        - It then calculates the standard deviation (SD) of glucose readings using the 'std' method from pandas.Series.
        - The coefficient of variation (CV) is calculated as (SD * 100 / average glucose).
        - The calculated SD and CV values are returned as a dictionary with corresponding labels.
    """
    def run(df):
        units = preprocessing.detect_units(df)
        # Calculate the average glucose reading using the 'average_glc' function
        avg_glc = df['glc'].mean()
        # Calculate the standard deviation (SD) of glucose readings
        sd = df['glc'].std()
        # Calculate the coefficient of variation (CV) as (SD * 100 / average glucose)
        cv = sd * 100 / avg_glc
        # Create a dictionary with the calculated glycemic variability metrics
        variability_metrics = {
            f'SD ({units})': sd,
            'CV (%)': cv
        }
        return variability_metrics
        
    if 'ID' in df.columns:
        results = df.groupby('ID').apply(lambda group: pd.DataFrame.from_dict(run(group), orient='index').T).reset_index().drop(columns='level_1')
        return results
    else:    
        results = run(df)
        return results

    
def ea1c(df):
    """
    Calculate estimated average HbA1c (eA1c) based on the average glucose readings in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'glc' column with glucose readings.

    Returns:
        float: The estimated average HbA1c (eA1c) value.

    Note:
        - The function calculates the average glucose reading from the 'glc' column in the DataFrame.
        - It determines the units of the glucose readings using the 'detect_units' function from the 'preprocessing' module.
        - If the units are 'mmol/l', the eA1c is calculated using the formula: (average glucose + 2.59) / 1.59.
        - If the units are not 'mmol/l', the eA1c is calculated using the formula: (average glucose + 46.7) / 28.7.
        - The calculated eA1c value is returned as a float.
    """
    def run(df):
        avg_glc = df['glc'].mean()

        # Check the units of glucose readings using the 'detect_units' function from the 'preprocessing' module
        if preprocessing.detect_units(df) == 'mmol/L':
            # Calculate eA1c using the formula: (average glucose + 2.59) / 1.59
            ea1c_result = (avg_glc + 2.59) / 1.59
        else:
            # Calculate eA1c using the formula: (average glucose + 46.7) / 28.7
            ea1c_result = (avg_glc + 46.7) / 28.7

        return {'eA1c (%)': ea1c_result}
    
    if 'ID' in df.columns:
        results = df.groupby('ID').apply(lambda group: pd.DataFrame.from_dict(run(group), orient='index').T).reset_index().drop(columns='level_1')
        return results
    else:    
        results = run(df)
        return results    

def calculate_auc(df):
    """
    Calculate the area under the curve (AUC) for a group of glucose readings.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'glc' column with glucose readings and a 'time' column with timestamps.

    Returns:
        float: The average AUC for the group.

    Note:
        - The function calculates the AUC using the 'glc' values and the corresponding time differences from the start time.
        - The time differences are converted to hours for AUC calculation.
        - If the DataFrame has only one row, it returns np.nan.
    """
    if df.shape[0] > 1:
        start_time = df.time.iloc[0]
        mins_from_start = df.time.apply(lambda x: x - start_time)
        df['hours_from_start'] = mins_from_start.apply(lambda x: (x.total_seconds() / 60) / 60)
        avg_auc = metrics.auc(df['hours_from_start'], df['glc'])
        return avg_auc
    else:
        return np.nan
        

def auc(df):
    """
    Calculate the area under the curve (AUC) for glucose readings in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'glc' column with glucose readings and a 'time' column with timestamps.

    Returns:
        dict: A dictionary containing the hourly average AUC, daily AUC breakdown, and hourly AUC breakdown.

    Note:
        - The function calculates the AUC by breaking down the DataFrame into hourly and daily intervals.
        - It uses the 'calculate_auc' function to calculate the AUC for each group.
        - The hourly AUC breakdown is a DataFrame with columns 'date', 'hour', and 'auc'.
        - The daily AUC breakdown is a Series with dates as the index and average AUC values as the values.
        - The hourly average AUC is the mean of the AUC values in the hourly breakdown.
    """
    def run(df):
        if preprocessing.detect_units(df) == 'mmol/L':
            units = 'mmol h/L'
        else:
            units = 'mg h/dL'
        # Add 'date' and 'hour' columns to the DataFrame based on the 'time' column
        df['date'] = df['time'].dt.date
        df['hour'] = df['time'].dt.hour

        try:
            # Calculate the AUC for each hourly group using the 'calculate_auc' function
            hourly_breakdown = df.groupby([df.date, df.hour]).apply(lambda group: calculate_auc(group)).reset_index()
            hourly_breakdown.columns = ['date', 'hour', 'auc']

            # Calculate the daily average AUC
            daily_breakdown = hourly_breakdown.groupby('date').auc.mean()

            # Calculate the hourly average AUC
            hourly_avg = hourly_breakdown['auc'].mean()
        except:
            hourly_avg = np.nan
        return {f'AUC ({units})': hourly_avg}# 'auc_daily_breakdown': daily_breakdown, 'auc_hourly_breakdown': hourly_breakdown}
    
    df = copy.copy(df)
    if 'ID' in df.columns:
        results = df.groupby('ID').apply(lambda group: pd.DataFrame.from_dict(run(group), orient='index').T).reset_index().drop(columns='level_1')
        return results
    else:    
        results = run(df)
        return results
    
def mage(df):
    """
    Calculate the mean amplitude of glycemic excursions (MAGE) using scipy's signal class.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'glc' column with glucose readings and a 'time' column with timestamps.

    Returns:
        dict: A dictionary containing the MAGE value.

    Note:
        - The function uses scipy's signal.find_peaks function to find peaks and troughs in the glucose readings.
        - It then calculates the positive and negative MAGE and returns their mean.
    """
    def run(df):
        units = preprocessing.detect_units(df)
        # Find peaks and troughs using scipy signal
        peaks, properties = signal.find_peaks(df['glc'], prominence=df['glc'].std())
        troughs, properties = signal.find_peaks(-df['glc'], prominence=df['glc'].std())

        # Create a dataframe with peaks and troughs in order
        single_indexes = df.iloc[np.concatenate((peaks, troughs, [0, -1]))]
        single_indexes.sort_values('time', inplace=True)

        # Make a difference column between the peaks and troughs
        single_indexes['diff'] = single_indexes['glc'].diff()

        # Calculate the positive and negative MAGE and their mean
        mage_positive = single_indexes[single_indexes['diff'] > 0]['diff'].mean()
        mage_negative = single_indexes[single_indexes['diff'] < 0]['diff'].mean()

        if pd.notnull(mage_positive) and pd.notnull(mage_negative):
            mage_mean = statistics.mean([mage_positive, abs(mage_negative)])
        elif pd.notnull(mage_positive):
            mage_mean = mage_positive
        elif pd.notnull(mage_negative):
            mage_mean = abs(mage_negative)
        else:
            mage_mean = np.nan #0

        return {f'MAGE ({units})': mage_mean}
    
    if 'ID' in df.columns:
        results = df.groupby('ID').apply(lambda group: pd.DataFrame.from_dict(run(group), orient='index').T).reset_index().drop(columns='level_1')
        return results
    else:    
        results = run(df)
        return results

def time_in_range(df):
    """
    Helper function for time in range calculation with normal thresholds. Calculates the percentage of readings within
    each threshold by dividing the number of readings within range by total length of the series.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'glc' column with glucose readings.

    Returns:
        dict: A dictionary containing the percentages of readings within each threshold range.

    Note:
        - The function calculates the percentage of readings within different threshold ranges for time in range (TIR) analysis.
        - TIR normal represents the percentage of readings within the range [3.9, 10].
        - TIR normal 1 represents the percentage of readings within the range [3.9, 7.8].
        - TIR normal 2 represents the percentage of readings within the range [7.8, 10].
        - TIR level 1 hypoglycemia represents the percentage of readings within the range [3, 3.9].
        - TIR level 2 hypoglycemia represents the percentage of readings below 3.
        - TIR level 1 hyperglycemia represents the percentage of readings within the range (10, 13.9].
        - TIR level 2 hyperglycemia represents the percentage of readings above 13.9.
    """
    def run(df):
        # Convert input series to NumPy array for vectorized calculations
        series = np.array(df['glc']) 
        # Get length of the series
        df_len = len(series)

        # Detect the units used in the df
        units = preprocessing.detect_units(df)
        
        # Use this to get the thresholds from the global dictionary
        thresholds = UNIT_THRESHOLDS.get(units, {})
        norm_tight = thresholds.get('norm_tight')
        hypo_lv1 = thresholds.get('hypo_lv1')
        hypo_lv2 = thresholds.get('hypo_lv2')
        hyper_lv1 = thresholds.get('hyper_lv1')
        hyper_lv2 = thresholds.get('hyper_lv2')

        # Calculate the percentage of readings within each threshold range
        tir_norm = np.around(np.sum((series >= hypo_lv1) & (series <= hyper_lv1)) / df_len * 100, decimals=2)
        tir_norm_1 = np.around(np.sum((series >= hypo_lv1) & (series < norm_tight)) / df_len * 100, decimals=2)
        tir_norm_2 = np.around(np.sum((series >= norm_tight) & (series <= hyper_lv1)) / df_len * 100, decimals=2)
        tir_lv1_hypo = np.around(np.sum((series < hypo_lv1) & (series >= hypo_lv2)) / df_len * 100, decimals=2)
        tir_lv2_hypo = np.around(np.sum(series < hypo_lv2) / df_len * 100, decimals=2)
        tir_lv1_hyper = np.around(np.sum((series <= hyper_lv2) & (series > hyper_lv1)) / df_len * 100, decimals=2)
        tir_lv2_hyper = np.around(np.sum(series > hyper_lv2) / df_len * 100, decimals=2)

        # Return the calculated values as a dictionary
        return {
            'TIR normal (%)': tir_norm,
            'TIR normal 1 (%)': tir_norm_1,
            'TIR normal 2 (%)': tir_norm_2,
            'TIR level 1 hypoglycemia (%)': tir_lv1_hypo,
            'TIR level 2 hypoglycemia (%)': tir_lv2_hypo,
            'TIR level 1 hyperglycemia (%)': tir_lv1_hyper,
            'TIR level 2 hyperglycemia (%)': tir_lv2_hyper
        }
    
    if 'ID' in df.columns:
        results = df.groupby('ID').apply(lambda group: pd.DataFrame.from_dict(run(group), orient='index').T).reset_index().drop(columns='level_1')
        return results
    else:    
        results = run(df)
        return results

def convert_to_rounded_percent(value, length):
    return round(value * 100 / length, 2)

def unique_tir_helper(glc_series, lower_thresh, upper_thresh):
    df_len = glc_series.size
    if lower_thresh==2.2:
        tir = convert_to_rounded_percent(glc_series.loc[glc_series <= upper_thresh].size, df_len)
    elif upper_thresh==22.2:
        tir = convert_to_rounded_percent(glc_series.loc[glc_series >= lower_thresh].size, df_len)
    else:
        tir = convert_to_rounded_percent(glc_series.loc[(glc_series <= upper_thresh) & (glc_series >= lower_thresh)].size, df_len)
    return tir


def unique_time_in_range(df, thresholds):
    units = preprocessing.detect_units(df)
    if thresholds is None:
        return {}
    results_dict = {}
    for i in thresholds:
        name = f'TIR {i[0]}-{i[1]}{units} (%)'
        tir = unique_tir_helper(df['glc'], i[0], i[1])
        results_dict[name] = tir
    return results_dict


def glycemic_episodes(df, hypo_lv1_thresh=None, hypo_lv2_thresh=None, hyper_lv1_thresh=None, hyper_lv2_thresh=None, mins=15, long_mins=120):
    """
    Calculate the statistics of glycemic episodes (hypoglycemic and hyperglycemic events) based on glucose readings.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'glc' column with glucose readings and a 'time' column with timestamps.
        hypo_lv1_thresh (float, optional): Level 1 hypoglycemic threshold. If not provided, it will be determined based on the units detected. Default is None.
        hypo_lv2_thresh (float, optional): Level 2 hypoglycemic threshold. If not provided, it will be determined based on the units detected. Default is None.
        hyper_lv1_thresh (float, optional): Level 1 hyperglycemic threshold. If not provided, it will be determined based on the units detected. Default is None.
        hyper_lv2_thresh (float, optional): Level 2 hyperglycemic threshold. If not provided, it will be determined based on the units detected. Default is None.
        mins (int, optional): Minimum duration in minutes for an episode to be considered. Default is 15.
        long_mins (int, optional): Minimum duration in minutes for a prolonged episode to be considered. Default is 120.

    Returns:
        dict: A dictionary containing the statistics of glycemic episodes.

    Note:
        - The function detects the units of glucose readings in the DataFrame.
        - The threshold values for hypoglycemic and hyperglycemic episodes are determined based on the detected units, unless explicitly provided.
        - The function calculates the statistics of glycemic episodes using the '_glycemic_events_helper.calculate_episodes' helper function.
        - The calculated statistics include the total number, LV1 (Level 1) events, LV2 (Level 2) events, prolonged events, average length, and total time spent in episodes for both hypoglycemic and hyperglycemic events.
    """
    def run(df, hypo_lv1_thresh, hypo_lv2_thresh, hyper_lv1_thresh, hyper_lv2_thresh, mins, long_mins):
        # Identify the units of the dataframe
        units = preprocessing.detect_units(df)

        # Determine threshold values if not provided
        thresholds = UNIT_THRESHOLDS.get(units, {})
        hypo_lv1_thresh = hypo_lv1_thresh or thresholds.get('hypo_lv1')
        hypo_lv2_thresh = hypo_lv2_thresh or thresholds.get('hypo_lv2')
        hyper_lv1_thresh = hyper_lv1_thresh or thresholds.get('hyper_lv1')
        hyper_lv2_thresh = hyper_lv2_thresh or thresholds.get('hyper_lv2')

        # Calculate statistics for hypoglycemic events
        total_hypos, lv1_hypos, lv2_hypos, prolonged_hypos, avg_length_hypos, total_time_hypos = _glycemic_events_helper.calculate_episodes(df, True, hypo_lv1_thresh, hypo_lv2_thresh, mins, long_mins)

        # Calculate statistics for hyperglycemic events
        total_hypers, lv1_hypers, lv2_hypers, prolonged_hypers, avg_length_hypers, total_time_hypers = _glycemic_events_helper.calculate_episodes(df, False, hyper_lv1_thresh, hyper_lv2_thresh, mins, long_mins)

        # Prepare results dictionary
        results = {'Total number hypoglycemic events': total_hypos, 
                    'Number LV1 hypoglycemic events': lv1_hypos, 
                    'Number LV2 hypoglycemic events':lv2_hypos, 
                    'Number prolonged hypoglycemic events':prolonged_hypos, 
                    'Avg. length of hypoglycemic events': avg_length_hypos, 
                    'Total time spent in hypoglycemic events':total_time_hypos,
                    'Total number hyperglycemic events':total_hypers, 
                    'Number LV1 hyperglycemic events':lv1_hypers,
                    'Number LV2 hyperglycemic events':lv2_hypers,
                    'Number prolonged hyperglycemic events':prolonged_hypers, 
                    'Avg. length of hyperglycemic events':avg_length_hypers,
                    'Total time spent in hyperglycemic events':total_time_hypers}
        return results
    
    if 'ID' in df.columns:
        results = df.groupby('ID').apply(lambda group: pd.DataFrame.from_dict(run(group, hypo_lv1_thresh, hypo_lv2_thresh, hyper_lv1_thresh, hyper_lv2_thresh, mins, long_mins), orient='index').T).reset_index().drop(columns='level_1')
        return results
    else:    
        results = run(df, hypo_lv1_thresh, hypo_lv2_thresh, hyper_lv1_thresh, hyper_lv2_thresh, mins, long_mins)
        return results

def data_sufficiency(df, interval=None, start_time=None, end_time=None):
    """
    Calculate the data sufficiency percentage based on the provided DataFrame, gap size, and time range.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'glc' column with glucose readings and a 'time' column with timestamps.
        gap_size (int): The size of the gap in minutes to check for data sufficiency.
        start_time (datetime.datetime, optional): The start time of the time range. If not provided, it will be determined from the DataFrame. Default is None.
        end_time (datetime.datetime, optional): The end time of the time range. If not provided, it will be determined from the DataFrame. Default is None.

    Returns:
        dict: A dictionary containing the start and end datetimes of the time range and the data sufficiency percentage.

    Raises:
        ValueError: If the gap size is not 5 or 15.

    Note:
        - The function calculates the data sufficiency based on the number of non-null values within the specified gap size intervals.
        - The start and end times of the time range are either provided or determined from the DataFrame.
        - The gap size must be either 5 or 15. Otherwise, a ValueError is raised.
        - The data sufficiency percentage is calculated as the ratio of non-null values to the total expected values.
    """
    def run(df, interval, start_time, end_time):
        # Determine start and end time from the DataFrame if not provided
        start_time = start_time or df['time'].iloc[0]
        end_time = end_time or df['time'].iloc[-1]

        # Subset the DataFrame based on the provided time range
        df = df.loc[(df['time'] >= start_time) & (df['time'] <= end_time)]

        # Calculate the interval size
        interval = interval or df['time'].diff().mode().iloc[0]
        # If it doesn't conform to 5 or 15 then don't count it
        if ((timedelta(minutes=4) < interval) & (interval < timedelta(minutes=6))):
            freq = '5min'
        elif ((timedelta(minutes=14) < interval) & (interval < timedelta(minutes=16))):
            freq = '15min'
        else:
            raise ValueError('Invalid gap size. Gap size must be 5 or 15.')

        # Calculate the number of non-null values
        number_readings = sum(df.set_index('time').groupby(pd.Grouper(freq=freq)).count()['glc'] > 0)
        # Calculate the total expected readings based on the start and end of the time range
        total_readings = ((end_time - start_time) + interval) / interval

        # Calculate the data sufficiency percentage
        if number_readings >= total_readings:
            data_sufficiency = 100
        else:
            data_sufficiency = number_readings * 100 / total_readings

        return {
            'Start DateTime': str(start_time.round('min')),
            'End DateTime': str(end_time.round('min')),
            'Data Sufficiency (%)': np.round(data_sufficiency, 1)
        }
    
    if 'ID' in df.columns:
        results = df.groupby('ID').apply(lambda group: pd.DataFrame.from_dict(run(group, interval, start_time, end_time), orient='index').T).reset_index().drop(columns='level_1')
        return results
    else:    
        results = run(df, interval, start_time, end_time)
        return results

def calc_bgi(glucose, units):
    """
    Calculate the Blood Glucose Index (BGI) based on glucose readings.

    Args:
        glucose (float): Glucose reading.
        units (str): Units of glucose measurement ('mmol/L' or 'mg/dL').

    Returns:
        float: Blood Glucose Index (BGI) value.

    Note:
        - The BGI calculation depends on the units of glucose.
        - The formula for BGI differs for 'mmol/L' and 'mg/dL'.
    """
    if units == 'mmol/L':
        num1 = 1.794
        num2 = 1.026
        num3 = 1.861
    else:
        num1 = 1.509
        num2 = 1.084
        num3 = 5.381
    bgi = num1 * (np.log(glucose) ** num2 - num3)
    return bgi

def lbgi(glucose, units):
    """
    Calculate the Low Blood Glucose Index (LBGI) based on glucose readings.

    Args:
        glucose (float): Glucose reading.
        units (str): Units of glucose measurement ('mmol/L' or 'mg/dL').

    Returns:
        float: Low Blood Glucose Index (LBGI) value.

    Note:
        - The LBGI is calculated using the BGI value.
        - The LBGI is a measure of the risk associated with low blood glucose levels.
    """
    bgi = calc_bgi(glucose, units)
    lbgi = 10 * (min(bgi, 0) ** 2)
    return lbgi

def hbgi(glucose, units):
    """
    Calculate the High Blood Glucose Index (HBGI) based on glucose readings.

    Args:
        glucose (float): Glucose reading.
        units (str): Units of glucose measurement ('mmol/L' or 'mg/dL').

    Returns:
        float: High Blood Glucose Index (HBGI) value.

    Note:
        - The HBGI is calculated using the BGI value.
        - The HBGI is a measure of the risk associated with high blood glucose levels.
    """
    bgi = calc_bgi(glucose, units)
    hbgi = 10 * (max(bgi, 0) ** 2)
    return hbgi

def bgi(df):
    """
    Calculate the Blood Glucose Index (BGI) metrics for a DataFrame of glucose readings.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'glc' column with glucose readings.

    Returns:
        dict: A dictionary containing the Low Blood Glucose Index (LBGI) and High Blood Glucose Index (HBGI) values.

    Note:
        - The function calculates the LBGI and HBGI based on the glucose readings and detects the units of measurement.
        - The LBGI and HBGI are average values calculated from individual readings using the 'lbgi' and 'hbgi' functions.
    """
    def run(df):
        units = preprocessing.detect_units(df)
        lbgi_result = df['glc'].apply(lambda x: lbgi(x, units)).mean()
        hbgi_result = df['glc'].apply(lambda x: hbgi(x, units)).mean()
        return {'LBGI': lbgi_result, 'HBGI': hbgi_result}
    
    if 'ID' in df.columns:
        results = df.groupby('ID').apply(lambda group: pd.DataFrame.from_dict(run(group), orient='index').T).reset_index().drop(columns='level_1')
        return results
    else:    
        results = run(df)
        return results
