import pandas as pd
from datetime import timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def calc_diff(group):
    row1 = group.iloc[0]
    endtime = group.iloc[-1]['time_rep']
    starttime = row1['time_rep']
    diff = endtime-starttime
    row1['diff'] = diff
    return row1

def collapse_bool_array(df, bool_array):
    unique_num = bool_array.ne(bool_array.shift()).cumsum()
    number_consec = unique_num.map(unique_num.value_counts()).where(bool_array)
    df_unique = pd.DataFrame({'time_rep': df['time'], 'glc_rep':
                            df['glc'], 'unique_number': unique_num,
                            'consec_readings': number_consec})
    df_unique['time_rep'] = pd.to_datetime(df_unique['time_rep'])
    # Drop any null glucose readings and reset index
    df_unique.dropna(subset=['glc_rep'], inplace=True)
    df_unique.reset_index(inplace=True, drop=True)
    diff = df_unique.groupby('unique_number').apply(lambda group: calc_diff(group))
    diff = diff.drop(columns=['glc_rep'])#.reset_index()
    return diff

def calc_duration(unique_min, mins):
    # Only keep hypos that are 15 mins or longer (smaller than this doesn't count)
    results = unique_min[unique_min['diff'] >= timedelta(minutes=mins)]

    # Fill the consec readings with binary value to show whether they are hypos or
    # the periods between hypos
    results['consec_readings'] = results['consec_readings'].fillna(-1)
    results['event'] = results['consec_readings'] > 0

    # Merge any consecutive values left by removal of too-short episodes using
    # a new unique number
    results['unique'] = results['event'].ne(results['event'].shift()).cumsum()
    return results

def merge_events(results, mins):
    # Group by the unique number, select the min values and select relevant columns
    results_grouped = results.groupby('unique').min()[['time_rep',  'event', 'diff']] #'glc_rep',
    results_grouped['diff2'] = results_grouped['time_rep'].diff().shift(-1)
    # Drop the non-hypo periods and then drop the hypo column
    final_results = results_grouped.loc[results_grouped['event'] ==
                                        True].drop(columns=['event'])
                                        # Rename columns
    final_results.columns = ['start_time',  'initial_duration', 'duration'] # 'min_glc',

    # Fill final hypo with previous duration value in diff col then drop initial
    # duration
    final_results['duration'] = final_results['duration'].fillna(final_results['initial_duration'])
    final_results.drop(columns=['initial_duration'], inplace=True)
    # Drop the final column if it's less than 15 mins
    final_results = final_results.loc[final_results['duration']>=
                                        timedelta(minutes=mins)]
    final_results['end_time'] = final_results['start_time'] + final_results['duration']
    # Reset index
    final_results.reset_index(drop=True, inplace=True)
    return final_results

def overlap(row, lv2_events):
    for index, row_lv2 in lv2_events.iterrows():
        if row['start_time'] <= row_lv2['time_rep'] <= row['end_time']:
            return [True, row_lv2['prolonged']]
    else:
        return [False, False]

def calculate_episodes(df, hypo, thresh, thresh_lv2, mins, long_mins):
    if hypo:
        bool_array = df['glc'] < thresh
        bool_array_lv2 = df['glc'] < thresh_lv2
    else:
        bool_array = df['glc'] > thresh
        bool_array_lv2 = df['glc'] > thresh_lv2

    # All events
    unique_min = collapse_bool_array(df, bool_array)
    results = calc_duration(unique_min, mins)
    #results_lv2 = calc_duration(unique_min_lv2, mins)
    final_results = merge_events(results, mins)
    if final_results.empty:
        return 0, 0, 0, 0, 0, 0
    # Level 2 hypos
    unique_min_lv2 = collapse_bool_array(df, bool_array_lv2)
    lv2_events = unique_min_lv2.dropna(subset=['consec_readings'])
    lv2_events = lv2_events[lv2_events['diff']>=timedelta(minutes=mins)]
    lv2_events['prolonged'] = lv2_events['diff']>=timedelta(minutes=long_mins)
    final_results[['lv2', 'prolonged']] = final_results.apply(lambda row: overlap(row, lv2_events), axis=1, result_type ='expand')

    number_of_episodes = final_results.shape[0]
    number_of_lv2 = final_results.lv2.sum()
    number_of_lv1 = number_of_episodes - number_of_lv2
    prolonged = final_results['prolonged'].sum()

    avg_length = final_results.duration.mean().round('1s')
    total_time = final_results.duration.sum()

    # Return 0s if no hypos and nan if something weird happens
    if pd.notnull(avg_length):
        avg_length = avg_length
        total_time = total_time
    elif number_of_episodes == 0:
        avg_length = 0
        total_time = 0
    else:
        avg_length = np.nan
        total_time = np.nan
    return number_of_episodes, number_of_lv1, number_of_lv2, prolonged, str(avg_length), str(total_time)