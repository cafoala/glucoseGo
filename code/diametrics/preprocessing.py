import pandas as pd
import datetime
import numpy as np
import warnings
import copy

def check_df(df):
    """
    Check if the given object is a valid DataFrame.
    
    Args:
        df (object): The object to be checked.

    Returns:
        bool: True if the object is a valid DataFrame, False otherwise.

    Note:
        - If the object is not a DataFrame, a warning will be issued.
        - If the DataFrame has null values in the 'time' or 'glc' columns, it will be considered invalid.
        - An empty DataFrame will also be considered invalid.
    """
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
        
def replace_cutoffs(df, remove=False, cap=True, lo_cutoff=2.1, hi_cutoff=22.3):
    """
    Replace values in the 'glc' column of the given DataFrame based on specified cutoffs.

    Args:
        df (pandas.DataFrame): The DataFrame in which values will be replaced.
        remove (bool, optional): Indicates whether to remove rows with non-numeric 'glc' values. Defaults to False.
        cap (bool, optional): Indicates whether to cap values above the high cutoff and below the low cutoff. Defaults to True.
        lo_cutoff (float, optional): The low cutoff value for replacing 'glc' values. Defaults to 2.1.
        hi_cutoff (float, optional): The high cutoff value for replacing 'glc' values. Defaults to 22.3.

    Returns:
        pandas.DataFrame: The modified DataFrame with replaced 'glc' values and optionally removed rows.

    Note:
        - If remove is True, the function replaces non-numeric values in the 'glc' column with the respective cutoff values.
        - If cap is True, the function caps values above hi_cutoff and below lo_cutoff with the respective cutoff values.
        - The function also converts the 'glc' column to numeric values and converts the 'time' column to datetime.
    """
    df = copy.copy(df)
    if not remove:
        df['glc']= pd.to_numeric(df['glc'].replace({'High': hi_cutoff, 'Low': lo_cutoff, 'high': hi_cutoff, 'low': lo_cutoff, 
                             'HI':hi_cutoff, 'LO':lo_cutoff, 'hi':hi_cutoff, 'lo':lo_cutoff}))

        if cap:
            df.loc[df['glc']>hi_cutoff, 'glc'] = hi_cutoff
            df.loc[df['glc']<lo_cutoff, 'glc'] = lo_cutoff

    df = df[pd.to_numeric(df['glc'], errors='coerce').notnull()]
    df['glc'] = pd.to_numeric(df['glc'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.reset_index(drop=True)
    return df


def fill_missing_data(df, interval=5, method='pchip', limit=30, order=5):
    """
    Interpolate missing data in a time series using a specified interpolation method.

    Args:
        df (pandas.DataFrame): The time series dataset with a 'time' column and a numeric data column.
        interval (int): The resampling interval in minutes. Defaults to 5.
        method (str): The interpolation method to use. Defaults to 'pchip'.
        limit (int): The maximum number of consecutive missing data points to interpolate. Defaults to 30.
        order (int): The order of the interpolation method if 'method' is 'polynomial' or 'spline'. Defaults to 5.

    Returns:
        pandas.DataFrame: The DataFrame with missing data interpolated using the specified method.

    Note:
        - The 'time' column should be set as the DataFrame index before calling this function.
        - The 'time' column should be in datetime format.
        - Missing data points are identified as NaN values in the data column.
        - The function resamples the DataFrame at the specified interval using the mean aggregation.
        - The specified interpolation method is applied to fill in the missing data points.
        - The 'limit' parameter controls the maximum number of consecutive missing data points to interpolate.
        - For polynomial or spline interpolation, the 'order' parameter specifies the order of the interpolation.

    """
    df = copy.copy(df)

    # Create time-series index and take resample to whatever the interval is
    df = df.set_index('time')
    df_resampled = df.resample(f'{interval}min').mean()
    
    # Limit in minutes divided by interval to give number of readings
    limit = int(limit/interval)
    
    # Create a mask for the limit
    s = df_resampled['glc'].notnull()
    s = s.ne(s.shift()).cumsum()
    m = df_resampled.groupby([s, df_resampled['glc'].isnull()])['glc'].transform('size').where(df_resampled['glc'].isnull())
    
    # If the method is polynomial or spline, an order needs to be given
    if (method == 'polynomial') | (method == 'spline'):
        df_interp = df_resampled.interpolate(method=method, 
                                                     limit_area='inside',
                                                     limit_direction='forward',
                                                     limit=limit, order=order).mask(m>limit)
    # Else no order is needed
    else:
        df_interp = df_resampled.interpolate(method=method,
                                                     limit_area='inside',
                                                     limit_direction='forward',
                                                     limit=limit).mask(m>limit)
    df_interp = df_interp.round(1).reset_index()
    return df_interp


def set_time_frame(df, window):
    """
    Filter the given DataFrame based on the specified time period.

    Args:
        df (pandas.DataFrame): The DataFrame to be filtered.
        period (list or dict): The time period to filter the DataFrame. It can be specified as a list or a dictionary.

            - If period is a list, it should contain two elements representing the start and end times of the period.
            Only the rows with 'time' values greater than or equal to the start time and less than the end time will be returned.

            - If period is a dictionary, it should contain key-value pairs where the keys represent IDs and the values
            are two-element lists representing the start and end times for each ID. The function will return all rows
            where the 'ID' column matches the specified IDs and the 'time' values fall within the respective time windows.

    Returns:
        pandas.DataFrame: The filtered DataFrame containing only the rows that match the specified time period.
    
    Raises:
        ValueError: If the period argument is not of type list or dict.
    """
    if isinstance(window, list):
        return df.loc[(df['time']>=window[0])&(df['time']<window[1])]
    elif isinstance(window, dict):
        conditions = " | ".join(["((df['ID'] == '{0}')&(df['time']>='{1}')&(df['time']<='{2}'))".format(ID, window[0], window[1]) for ID, window in window.items()])
        cut_df = df[eval(conditions)].reset_index(drop=True)
        return cut_df
    else:
        raise ValueError("Invalid type for the 'period' argument. Expected a list or a dictionary.")
    
def detect_units(df):
    if df['glc'].min() > 50:
        return 'mg/dL'
    else:
        return 'mmol/L'
    

def change_units(df):
    """
    Convert glucose units in the DataFrame to a different unit based on a condition.

    Args:
        df (pandas.DataFrame): The DataFrame containing a 'glc' column with glucose readings.

    Returns:
        pandas.DataFrame: The DataFrame with glucose units converted based on the condition.

    Note:
        - The function checks the minimum glucose value in the DataFrame.
        - If the minimum value is greater than 50, the glucose units are converted to a different unit by multiplying with 0.0557 and rounding to one decimal place.
        - If the minimum value is less than or equal to 50, the glucose units are converted by multiplying with 0.0557 and rounding to the nearest integer.
    """
    df = copy.copy(df)
    if detect_units(df)=='mg/dL':
        # Convert glucose units by multiplying with 0.0557 and rounding to one decimal place
        df['glc'] = (df['glc'] * 0.0557).round(1)
    else:
        # Convert glucose units by multiplying with 0.0557 and rounding to the nearest integer
        df['glc'] = (df['glc'] / 0.0557).round(0)

    return df



    
