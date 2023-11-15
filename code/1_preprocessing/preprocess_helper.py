from datetime import datetime as dt
import datetime
import pandas as pd
import numpy as np
from datetime import timedelta
from tsfresh import extract_features, extract_relevant_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
extraction_settings = ComprehensiveFCParameters()
import sys
# Change path to wherever Diametrics is
path = "../../diametrics" #### CHANGE
sys.path.append(path)
import metrics

start_day = dt.strptime('05:00', '%H:%M').time()
noon =  dt.strptime('12:00', '%H:%M').time()
five = dt.strptime('17:00', '%H:%M').time()

# Lists of strings for each form of exercise
mixed = ['cycling & push ups', 'box', 'Weights+ cross training', 'gym',
         'workout', 'press-ups & sprints', 'work out', 'excersise class',
         'fitness test', 'rpe 13', 'muay thai', 'video', 'surf', 'baseball',
         'stair', 'martial arts', 'interval', 'equestrian', 'les mills',
         'orange', 'beachbody', 'physical therapy', 'otf', 'exercise class',
         'mma', 'pe class', 'mixed exercise', 'free play', 'undefined exercise',
         'recess', 'performing arts', 'horseback',
         ]

aerobic = ['walk', 'run', 'cycl', 'football', 'cross', 'tread',
           'cardio', 'aerobic', 'jog', 'tennis', 'golf', 'bik', 'squash', 'ride',
           'rugby', 'danc', 'skip', 'zumba', 'cricket', 'ski','swim',
           'spin', 'row', 'hik', 'badmin', 'aquafit', 'walik', 'clycling',
           'body jam', 'soccer',  'basketball', 'rope', 'volleyball', 'rafting',
           'trampolin', 'kayak', 'essentrics', 'park',  'skating', 'roller',
           'paddle', 'jumping', 'hockey', 'frisbee', 'backpacking', 'snow shoe', 
           'umpiring', 'peloton', 'laser tag', 'longboarding', 'jazzercise',
           'snowshoeing', 'skateboarding', 'kickball', 'fencing', 'racquet sports',
           'bowling', 'track & field']
           
anaerobic = ['weight', 'yoga', 'hiit', 'stretch', 'resist', 'climb', 'strengt',
             'legs', 'physio', 'pilates', 'punch', 'pump', 'calisthen',
             'anarobic', 'core', 'barre', 'functional', 'pushup', 'archery',
             'trx', 'body blast', 'bosu', 'ski', 'snowboard', 'bouldering', 
             'burpees', 'calesthenics', 'strrength training', 'ab-', 'ab ',]

to_remove = ['garden', 'house', 'diy', 'kids play centre', 'hoover', 'lambing',
             'sax', 'farm', 'sax', 'manual work', 'physical labor', 
             'shopping', 'band', 'bathroom deep clean', 'herd', ]

def check_time(x):
    """
    Checks and cleans a given input to ensure it's a valid time format (HH:MM). If not,
    it attempts to correct it using common replacements. If still invalid, it returns NaN.
    
    Parameters:
    - x (any): Input value to be checked and potentially corrected.
    
    Returns:
    - datetime.time or NaN: A valid datetime.time instance if the input can be corrected, otherwise NaN.
    
    Notes:
    - This function primarily handles string inputs with incorrect characters for time formats.
      It uses a predefined dictionary of replacements to attempt correction.
    """
    if (not isinstance(x, datetime.time)) & (pd.notnull(x)):
        replacements = {' ':'', ';': ':', '.': ':', '~':'', '1(:45':'13:45'}.items()
        for key, value in replacements:
            x = str(x).replace(key, value)
        try:
            x = datetime.datetime.strptime(x, '%H:%M').time()
            return x
        except:
            return np.nan
    else:
        return x
    
def clean(df):
    """
    Cleans and renames the columns of a given dataframe, specifically tailored for exercise diaries.
    
    Parameters:
    - df (DataFrame): Input dataframe to be cleaned.
    
    Returns:
    - DataFrame: The cleaned dataframe with renamed columns.
    
    Notes:
    - This function is designed to handle a specific structure of dataframe columns, ensuring only the first 14 
      columns are considered. If the dataframe structure changes, this function will need to be updated.
    - The expected input columns are not explicitly stated in the function. Users must ensure that they provide 
      data in the expected order.
    """
    df = df.iloc[:, :14]
    df.columns = ['ID', 'date', 'exercise_on_day', 'type_of_exercise',
                 'start_time', 'starting_glucose', 'finish_time',
                 'finishing_glucose', 'duration', 'borg', 'comment',
                 'hours_in_mins', 'mins', 'duration_mins']
    return df


def clean_diaries(df):
    """
    Rearranges data from two columns into one column. It's designed to handle cases where 
    data from a single "logical" column has been split across two physical columns in the dataframe.
    
    Parameters:
    - df (DataFrame): The input dataframe containing the diary data to be cleaned.
    
    Returns:
    - DataFrame: A cleaned dataframe where data is stacked vertically.
    
    Notes:
    - This function is specifically tailored for a dataframe structure where columns have been 
      duplicated with a ".1" suffix. If the dataframe structure changes, this function will need to be updated.
    """
    if 'last meal' not in df.columns:
        df['Last meal'] = np.nan
    if 'Lastmeal' not in df.columns:
        df['Lastmeal'] = np.nan
    first_df = df[['Date', 'Exercise', 'What time', 'how long',
                   'Type of exericse', 'Intensity', 'Glucose before',
                   'Glucose After', 'low during', 'high during',
                   'Fast acting insulin change', 'basel Insulin change',
                   'Take extra carb?', 'how much', 'Last meal', 'comment']] 
    try:
        secnd_df = df[['Date', 'Exercise.1', 'What time.1', 'how long.1',
                       'Type of exericse.1', 'Intensity.1', 'Glucose before.1',
                       'Glucose After.1', 'low during.1', 'high during.1',
                       'Fast acting insulin change.1', 'basel Insulin change.1',
                       'Take extra carb?.1', 'how much.1', 'Lastmeal',
                       'comment.1']]
    except:
        return first_df
    
    secnd_df.columns = first_df.columns
    merge_df = pd.concat([first_df, secnd_df], axis=0, ignore_index=True)
    return merge_df


def categorise_time_of_day(time):
    """
    Categorize a given time into one of three times of day: morning, afternoon, or evening.
    
    Parameters:
    - time (datetime.time): The time to be categorized.
    
    Returns:
    - str: One of the categories 'morning', 'afternoon', or 'evening'.
    
    Notes:
    - This function relies on predefined global constants: 'start_day', 'noon', and 'five' 
      to determine the thresholds for morning, afternoon, and evening.
    """
    if time > start_day and time < noon:
        return 'morning'
    if time >= start_day and time <= five:
        return 'afternoon'
    if time > five or time < start_day:
        return 'evening'


def date_preprocessing(dataset, variable, use_time=False, use_dow=False, use_tod=False):
    """
    Preprocess a datetime column, extracting various components like month, day, hour, etc.
    
    Parameters:
    - dataset (DataFrame): The DataFrame containing the datetime column to be processed.
    - variable (str): The name of the datetime column in the dataset.
    - use_time (bool): Whether to extract the hour and minute components.
    - use_dow (bool): Whether to extract the day of the week.
    - use_tod (bool): Whether to categorize the time into 'morning', 'afternoon', or 'evening'.
    
    Returns:
    - DataFrame: The original DataFrame with additional columns for extracted date components.
    """
    #dataset[variable+'_year'] = dataset[variable].dt.year
    dataset['month'] = dataset[variable].dt.month
    dataset['day'] = dataset[variable].dt.day
    #dataset[variable+'_timestamp'] = dataset[variable].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
    if use_time:
        dataset['hour'] = dataset[variable].dt.hour
        dataset['minute'] = dataset[variable].dt.minute
    if use_dow:
        dataset['day_of_week'] = dataset[variable].dt.day_of_week
    if use_tod:
        dataset['time_of_day'] = dataset[variable].dt.time.apply(lambda x: categorise_time_of_day(x))
    return dataset

def get_lab_results(id_frame):
    """
    Fills the missing lab results of the first visit using the results from the second visit, if available.
    
    Parameters:
    - id_frame (DataFrame): A DataFrame containing lab results for a particular ID.
    
    Returns:
    - DataFrame: The input DataFrame with missing values of the first row potentially filled by the second row.
    """
    if id_frame.shape[0]>1:
        id_frame.iloc[0] = id_frame.iloc[0].fillna(id_frame.iloc[1])
    return id_frame

def correct_borg(x):
    """
    Cleans and corrects the Borg rating of perceived exertion scores.
    
    Parameters:
    - x (str or float): The Borg score, either as a numeric value or as a descriptive word/phrase.
    
    Returns:
    - float or NaN: A numeric Borg score or NaN if the input is not recognized.
    
    Notes:
    - The function deals with a variety of potential string inputs representing Borg scores and 
      tries to convert them into standardized numeric values.
    """
    if type(x) == str:
        # Return nan
        if x in ['not stated', 'not known']:
            return np.nan
        else:
            # If there's 2 numbers, with a - or / between, take the average
            x = x.lower()
            x = x.replace('-', '\\').replace('/', '\\')
            if '\\' in x:
                split = x.split('\\', 1)
                try:
                    avg = (float(split[0]) + float(split[1]))/2
                    return avg
                except:
                    # This is unnecessary
                    print('Commencing phase 2')
            # Return borg score based on words used
            # If they've used more than 1 word, e.g. moderately hard, take the
            # average of the two scores 
            # Score counter to work out average
            score = 0
            # Sum of borg scores
            borg = 0
            if 'light' in x:
                borg += 11
                score += 1
            elif 'fairly light' in x:
                borg += 12
            elif ('medium' in x) | ('moderate' in x):
                borg += 13
                score += 1
            elif 'hard' in x:
                borg += 15
                score += 1
            elif 'somewhat hard' in x:
                borg += 14
            else:
                return np.nan
            borg = borg/score
            return borg
    else:
        return x

def replace_borg(x):
    if pd.notnull(x):
        if x<=12:
            return 0
        elif x>=16:
            return 2
        else:
            return 1
    else:
        return x


def divide_exercise_into_type(x):
    """
    Categorizes a given exercise type as either aerobic, anaerobic, or mixed.
    
    Parameters:
    - x (str): The exercise type to be categorized.
    
    Returns:
    - str or NaN: One of the categories 'aer', 'ana', 'mix', or NaN if the input is not recognized.
    
    Notes:
    - The function relies on predefined global lists: 'to_remove', 'aerobic', 'anaerobic', and 'mixed'
      to determine the category of the exercise.
    """
    aer = False
    ana = False
    mix = False
    x = x.lower()
    # Remove things like gardening etc
    if any(substring in x for substring in to_remove):
        return np.nan
    # Check if there's substrings from any of the lists
    if any(substring in x for substring in aerobic):
        aer = True
    elif any(substring in x for substring in anaerobic):
        ana = True
    elif any(substring in x for substring in mixed):
        mix = True
    else:
        print(x)
        return np.nan
    # If mix is True or both aerobic & anaerobic return mix
    if mix | (aer & ana):
        return 'mix'
    # Return ana if just anaerobic
    elif ana:
        return 'ana'
    # Return aer if just aerobic
    elif aer:
        return 'aer'
    
def round_time(dt=None, roundTo=60):
    """Round a datetime object to any time lapse in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    """
    dt = dt.to_pydatetime()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)

def combine_frame(filename, directory):
    """
    Combine data from a participant's file into a standardized DataFrame format.

    Parameters:
    - filename (str): Name of the file containing participant data.
    - directory (str): Path to the directory where the file is located.

    Returns:
    - DataFrame: Processed DataFrame with columns 'time', 'glc', and 'ID'.
                 The DataFrame will contain glucose readings ('glc') indexed by time ('time')
                 and tagged with the participant's ID ('ID').
    
    Notes:
    - The function expects the CSV file to have 'timestamp' and 'sensorglucose' columns.
    - Any 'Low' and 'High' values in the 'sensorglucose' column will be replaced with 2.22 and 22.22, respectively.
    """
    # set filepath for each file in directory
    filepath = directory + '/' + filename
    # Upload metrics dataset for the file
    df = pd.read_csv(filepath)
    df = df[['timestamp', 'sensorglucose']].dropna(how='all')
    df.columns = ['time', 'glc']
    # Set ID from filename
    ID = filename.replace('.csv', '')
    df['ID'] = ID
    df['time'] = pd.to_datetime(df['time'])
    # Round seconds to zero so minute align
    df.time = df.time.apply(lambda x: round_time(x))
    # Replace low and high with values
    df.glc = pd.to_numeric(df.glc.replace({'Low':2.22, 'High':22.22}))\
        .apply(lambda x: 22.22 if x > 22.22 else (2.22 if x < 2.22 else x))
    return df

def fill_missing(row):
    """
    Fill missing 'start_datetime' and 'finish_datetime' values in a row using the 'duration_mins' column.

    Parameters:
    - row (Series): A row from a DataFrame containing columns 'start_datetime', 'finish_datetime', and 'duration_mins'.

    Returns:
    - Series: The row with missing 'start_datetime' and 'finish_datetime' values filled, if possible.
    
    Notes:
    - If 'start_datetime' is missing, it will be filled by subtracting 'duration_mins' from 'finish_datetime'.
    - If 'finish_datetime' is missing, it will be filled by adding 'duration_mins' to 'start_datetime'.
    """
    if pd.isnull(row.start_datetime):
        row.start_datetime = row.finish_datetime-timedelta(minutes=row.duration_mins)
    if pd.isnull(row.finish_datetime):
        row.finish_datetime = row.start_datetime - timedelta(minutes=row.duration_mins)
    return row


def try_parsing_date(text):
    '''
    Attempts to parse a string to a datetime object using a set of predefined datetime formats.
    
    Parameters:
    text (str): The string to be parsed to a datetime object.
    
    Returns:
    bool: True if the string can be parsed to a datetime object using one of the predefined formats, False otherwise.
    '''
    text = str(text)
    formats = ("%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M",
               "%d-%m-%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
               "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", "%d/%m/%Y  %H:%M:%S")  # add dot format
    for fmt in formats:
        try:
            dt.strptime(text, fmt)
            return True
        except ValueError:
            pass
    return False


def test_col(col):
    '''
    Analyzes a pandas Series and determines the type of data it contains.
    
    Parameters:
    col (pd.Series): The pandas Series to be analyzed.
    
    Returns:
    str: A string representing the type of data contained in the Series. It can be 'dt' if the Series contains datetime values, 'glc_uk' if it contains UK glucose readings, 'glc_us' if it contains US glucose readings, or 'unknown' if the type of data couldn’t be determined.
    '''
    col = col.dropna()
    datetime_bool = col.apply(lambda x: try_parsing_date(x))
    if datetime_bool.all():
        return 'dt'
    try:
        col_num = pd.to_numeric(col).dropna()
        if ((col_num < 28) & (col_num > 2)).all():
            return 'glc_uk'
        elif ((col_num < 505) & (col_num > 36)).all():
            return 'glc_us'
        else:
            return 'unknown'

    except Exception:
        return 'unknown'


def find_header(df):
    '''
    Processes a DataFrame to find and set the appropriate header based on the data present in the rows.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be processed. 
    
    Returns:
    pd.DataFrame: A new DataFrame with the appropriate header set, based on the first row containing both valid datetime and numeric glucose values.
    
    Raises:
    Exception: An error occurred due to problems with the input data.
    '''
    dropped = df.dropna()
    dropped.columns = ['time', 'glc']
    count = 0
    for i, row in dropped.iterrows():
        is_date = try_parsing_date(row['time'])
        if not is_date:
            count += 1
            continue
        try:
            float(row['glc'])
            break
        except Exception:
            count += 1
    if count == dropped.shape[0]:
        raise Exception('Problem with input data')
    return dropped.iloc[count:]


def preprocess_data(df, id_colname=None):
    '''
    Preprocesses the input dataframe by identifying columns containing datetime and glucose data, 
    and then processing them to output a cleaner DataFrame with these columns.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to preprocess.
    id_colname (str, optional): The column name containing unique identifiers. If provided, this column will be included in the output DataFrame.
    
    Returns:
    pd.DataFrame: A preprocessed DataFrame with relevant datetime and glucose columns.
    
    Raises:
    Exception: If unable to identify datetime and/or glucose columns.
    '''
    
    # Identify the maximum number of rows across columns in the DataFrame
    max_rows = df.count().max()
    
    # Filter columns that have more than 70% non-null values
    cols_to_keep = df.count()[df.count() > max_rows * 0.7].index
    
    # Focus on the bottom 10% of rows for further processing
    footer_rows = df[cols_to_keep].iloc[int(-max_rows * 0.1):]
    
    # Initialize dictionary to keep track of types of columns
    col_type_dict = {'dt': [], 'glc_uk': [], 'glc_us': []}
    
    # Identify and categorize columns based on their type (datetime, UK glucose, or US glucose)
    for i in cols_to_keep:
        col_type = test_col(footer_rows[i])
        if col_type != 'unknown':
            col_type_dict[col_type].append(i)
    
    # If both datetime and UK glucose columns are identified, process them
    if (len(col_type_dict['dt']) > 0) & (len(col_type_dict['glc_uk']) > 0):
        sub_frame = df[[col_type_dict['dt'][-1], col_type_dict['glc_uk'][0]]]
        df_processed = find_header(sub_frame)
    # If both datetime and US glucose columns are identified, process them
    elif (len(col_type_dict['dt']) > 0) & (len(col_type_dict['glc_us']) > 0):
        sub_frame = df[col_type_dict['dt'][-1], col_type_dict['glc_us'][0]]
        df_processed = find_header(sub_frame)
        
        # Adjust for scale difference between UK and US glucose measurements
        try:
            df_processed['time'] = df_processed['time'] / 0.0555
        except Exception:
            print('Problem with input data')
    else:
        raise Exception('Can\'t identify datetime and/or glucose columns')
    
    # If an ID column is provided, merge it with the processed DataFrame
    if id_colname is not None:
        df_processed = df_processed.join(df[id_colname], how='left')
        df_processed.rename({id_colname: 'ID'}, inplace=True)
    
    # Reset index of the final processed DataFrame
    df_processed.reset_index(drop=True, inplace=True)
    
    return df_processed

def format_df(filename, directory):
    '''
    Reads and preprocesses the input file (either Excel or CSV) from the given directory.
    The function will replace specific values (High, Low, HI, LO) with numerical representations.
    If the dataframe doesn't contain an ID column, the filename (without the extension) will be used as the ID.

    Parameters:
    filename (str): Name of the file to process.
    directory (str): Directory path where the file is located.

    Returns:
    pd.DataFrame: A preprocessed DataFrame with the required format.

    Note:
    Ensure preprocess_data function is defined or imported before using this function.
    '''
    
    # Create the complete filepath
    filepath = directory + '/' + filename
    
    # Try to read the file as an Excel file
    try:
        df = pd.read_excel(filepath)
    except Exception:  # TODO: Consider specifying the types of Exceptions
        # If unsuccessful, try reading the file as a CSV
        try:
            df = pd.read_csv(filepath, names=[i for i in range(30)])
        except Exception:
            print('File in wrong format, must be Excel or CSV')
            return  # Exit function if neither format works

    # Replace specific string values with their respective numerical representation
    replacements = {'High': 22.3, 'Low': 2.1, 'HI': 22.3, 'LO': 2.1, 'hi': 22.3, 'lo': 2.1}
    df.replace(replacements, inplace=True)

    # Preprocess the dataframe
    df_preprocessed = preprocess_data(df)

    # Assign filename (without extension) as ID if there's no ID column
    df_preprocessed['ID'] = filename.rsplit('.', 1)[0]

    # Convert glucose values to numeric
    df_preprocessed.glc = pd.to_numeric(df_preprocessed.glc)

    return df_preprocessed


def calc_roc_arrow(df, time):
    """
    Calculate the rate of change (ROC) of glucose levels based on a specified time.

    Parameters:
    - df (DataFrame): Input DataFrame containing glucose data.
    - time (datetime): Specific time for which ROC is calculated.

    Returns:
    - glc (float): Glucose level at the closest time point before the specified time.
    - time_diff (float): Time difference between the closest measurement and the specified time (in hours).
    """
    # treating the exercise period as truth, so only look before
    sub_df = df[(df['time'] <= (time - timedelta(minutes=20))) &
                (df['time'] > time-timedelta(minutes=40))]
    if not sub_df.empty:
        glc = sub_df.iloc[-1].glc
        time_diff = (time - sub_df.iloc[-1].time).total_seconds()/(60**2)
    else:
        glc=np.nan
        time_diff = np.nan
    return glc, time_diff

def calc_glc_roc(df, time, window, libre=False):
    """
    Calculate starting glucose and rate of change (ROC) over a given window.

    Parameters:
    - df (DataFrame): Input DataFrame containing glucose data.
    - time (datetime): End time for the window.
    - window (int): Duration of the window in minutes.
    - libre (bool, optional): If True, processes Libre-specific columns. Default is False.

    Returns:
    - pd.Series: Series containing starting glucose and its ROC.
    """
    time = pd.to_datetime(time)
    
    if libre:
        df.dropna(subset=['glc', 'scan_glc'], how='all', inplace=True)
        df['glc'].fillna(df.scan_glc, inplace=True)
        
    # treating the exercise period as truth, so only look before
    sub_df = df[(df['time'] > (time - timedelta(minutes=window))) &
                (df['time'] < time)]
    
    # Calculate diff
    sub_df['one_time'] = time
    sub_df['diff'] = sub_df[['time', 'one_time']].diff(axis=1)['one_time']
    sub_df['diff'] = sub_df['diff'].apply(lambda x: abs(x.total_seconds()))
    
    if not sub_df.empty:
        ind = sub_df['diff'].idxmin()
        glc = sub_df['glc'].loc[ind]
        prev_glc, time_diff = calc_roc_arrow(df, time)
        if prev_glc != np.nan:
            roc = (prev_glc - glc)/time_diff
    else:
        glc = np.nan
        roc = np.nan
        
    return pd.Series([glc, roc])



def create_bout_id(diaries):
    """
    Create a unique ID for each bout of exercise.

    Parameters:
    - diaries (DataFrame): Input DataFrame containing exercise diary data.

    Returns:
    - DataFrame: DataFrame with added 'bout_id' column representing unique IDs for each exercise bout.
    """
    # Sort by datetime
    sorted_diaries = diaries.sort_values(['ID', 'start_datetime']).reset_index(drop=True)
    # Combine id with bout_number to give unique bout id per person
    sorted_diaries['bout_id'] = sorted_diaries.apply(lambda row: row['ID'] + '_' +
                                 row.start_datetime.strftime('%Y%M%d%H%M%S'), axis=1)
    # Drop bout_number columns
    #sorted_diaries.drop(columns=['bout_number'], inplace=True)
    return sorted_diaries

def store_glucose_data_as_series(dataframe):
    """
    Extract time and glucose data from a DataFrame.

    Parameters:
    - dataframe (DataFrame): Input DataFrame containing glucose data.

    Returns:
    - Series: Series indexed by time with glucose levels as values.
    """
    glc_series = dataframe.set_index('time')['glc']
    return glc_series


def extract_tsfresh(timeseries, y):
    '''
    Extract relevant features from the timeseries using tsfresh.
    
    This function takes a timeseries dataframe and a target dataframe `y` to
    extract features relevant for predicting the target variable.
    The extraction is performed only on intersecting IDs between timeseries and `y`.

    Parameters:
    timeseries (pd.DataFrame): The input dataframe with the time series data.
        It is expected to contain columns: bout_id, time, and glc.
    y (pd.Series or pd.DataFrame): Target variable dataframe indexed by bout_id.

    Returns:
    pd.DataFrame: A dataframe with the extracted features for each bout_id.

    Note:
    Ensure tsfresh's extract_relevant_features function and any required settings
    (like extraction_settings) are imported before using this function.
    '''
    
    # Identify bout_ids that exist in both `timeseries` and `y`
    intersecting_ids = set(timeseries.bout_id).intersection(set(y.index))
    
    # Filter the dataframes to only include intersecting bout_ids
    timeseries = timeseries.loc[timeseries.bout_id.isin(intersecting_ids)]
    y = y.loc[y.index.isin(intersecting_ids)]
    
    # Extract relevant features using tsfresh
    X = extract_relevant_features(timeseries, y, 
                                  column_id="bout_id", column_sort="time",
                                  column_value="glc",
                                  show_warnings=False,
                                  default_fc_parameters=extraction_settings)
    return X


def create_sub_cgm(cgm_df, interval, start, end, ID, bout_id):
    '''
    Extract a subsection of CGM data for a specific time interval and ID.

    Parameters:
    - cgm_df (pd.DataFrame): CGM data containing 'ID', 'time', and 'glc'.
    - interval (timedelta): The interval for which data sufficiency needs to be calculated.
    - start, end (datetime): The start and end of the period.
    - ID (str or int): ID of the patient.
    - bout_id (int): Unique ID for the bout.

    Returns:
    - pd.DataFrame: The subset of CGM data for the given time period and ID.
    - float: Data sufficiency as a percentage.
    '''
    
    # Filter CGM data based on ID and the given time range
    cgm_id = cgm_df.loc[(cgm_df['ID'] == ID) & (cgm_df['time'] >= start) & (cgm_df['time'] <= end)]
    
    # Assign bout_id to this subset
    cgm_id['bout_id'] = bout_id
    
    # Calculate data sufficiency
    if cgm_id.shape[0] == 0:
        data_suff = 0
    else:
        data_suff = metrics.data_sufficiency(cgm_id[['time', 'glc']], interval, start, end)
        data_suff = data_suff['Data Sufficiency (%)']
        
        
    return cgm_id, data_suff


def set_up_dataframes(cgm_df, exercise_df):
    '''
    Prepares a dataframe containing metrics and data sufficiency for each bout based on the CGM and exercise data.

    Parameters:
    - cgm_df (pd.DataFrame): CGM data.
    - exercise_df (pd.DataFrame): Exercise diary data.

    Returns:
    - list of dicts: Each dict contains the bout ID, data sufficiency, and other metrics.
    '''
    
    results = []

    # Identify the common IDs between CGM and exercise data
    ids_intersect = set(cgm_df['ID'].values).intersection(set(exercise_df['ID'].values))
    
    # For each ID, extract the relevant metrics
    for ID in ids_intersect:
        print(ID)
        diary_id = exercise_df.loc[exercise_df['ID'] == ID]

        for i, row in diary_id.iterrows():
            bout_id = row['bout_id']
            interval = timedelta(minutes=row['interval'])
            
            # Extract metrics for the DURING period
            start = pd.to_datetime(row.start_datetime)
            end = pd.to_datetime(row.finish_datetime)
            cgm_id_during, data_suff_during = create_sub_cgm(cgm_df, interval, start, end, ID, bout_id)

            bout_id_dict = {'bout_id': bout_id, 'data_suff_during': data_suff_during}
            
            if data_suff_during >= 50:
                # Extract metrics for the BEFORE period
                start = pd.to_datetime(row.start_datetime) - timedelta(hours=1)
                end = pd.to_datetime(row.start_datetime)
                cgm_id_before, data_suff_before = create_sub_cgm(cgm_df, interval, start, end, ID, bout_id)
                cgm_id_before = cgm_id_before.dropna(subset=['time', 'glc'])

                bout_id_dict['before_data_suff'] = data_suff_before

                if cgm_id_before.shape[0] != 0:
                    std_metrics_before = metrics.all_standard_metrics(cgm_id_before[['time', 'glc']])
                    tsfresh = extract_features(cgm_id_before,
                                               column_id="bout_id", column_sort="time",
                                               column_value="glc",
                                               show_warnings=False,
                                               default_fc_parameters=extraction_settings).to_dict(orient='records')[0]
                    bout_id_dict.update(std_metrics_before)
                    bout_id_dict.update(tsfresh)
                
            results.append(bout_id_dict)

    return results


def split_dataframe(df, chunk_size=100): 
    '''
    Split a dataframe into smaller chunks of a given size.

    Parameters:
    - df (pd.DataFrame): The dataframe to split.
    - chunk_size (int, optional): Size of each chunk. Default is 100.

    Returns:
    - list of pd.DataFrame: Chunks of the input dataframe.
    '''
    
    chunks = []
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])
    return chunks


def make_numeric(time):
    '''
    Convert a time string in the format "HH:MM:SS" to minutes.

    Parameters:
    - time (str or int): A time in the format "HH:MM:SS" or 0 or "0.0".

    Returns:
    - float: Time converted to minutes. If input is 0, "0.0", or NaN, the function returns the input unchanged.
    '''
    
    # If time is non-null and not any of the zero formats
    if (time != '0') and (time != '0.0') and (time != 0) and (pd.notnull(time)):
        hrs = int(time[7:9]) * 60  # Extract hours and convert to minutes
        mins = int(time[10:12])    # Extract minutes
        scs = int(time[13:15]) / 60  # Extract seconds and convert to minutes
        return hrs + mins + scs
    else:
        return time


def overview_results(df, numeric_demo, cat_demo, numeric_bout, cat_bout):
    '''
    Provide a summary of results based on numeric and categorical columns of the input dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe containing data for analysis.

    Returns:
    - pd.DataFrame: Summary of results for the numeric and categorical columns.
    '''
    
    results = []

    # Append the total number of bouts
    results.append(['number of bouts', df.shape[0]])
    print(df.head())
    # Calculate the IQR for bouts per person
    q1, q2, q3 = np.percentile(df['ID'].value_counts(), [25, 50,75])
    iqr = f'{q2} [{q1}, {q3}]'
    results.append(['bouts per person', iqr])

    # Append percentage of bouts ending in hypo
    results.append(['percentage bouts end in hypo', df.y.mean()])

    # For each numeric bout, calculate and append its IQR
    for n in numeric_bout:
        print(n)
        q1, q2, q3 = np.percentile(df[n].dropna(), [25, 50,75])
        q1 = np.round(q1,2)
        q2 = np.round(q2,2)
        q3 = np.round(q3,2)
        iqr = f'{q2} [{q1}, {q3}]'
        results.append([n, iqr])

    # Append value counts for each categorical bout
    for c in cat_bout:
        print(c)
        counts = df[c].value_counts()
        results.append([c, counts])

    # Obtain unique demographic data entries
    demo_df = df[numeric_demo+cat_demo].drop_duplicates()

    # Append total number of unique participants
    results.append(['number of participants', demo_df.shape[0]])

    # For each numeric demographic data, calculate and append its IQR
    for n in numeric_demo:
        q1, q2, q3 = np.percentile(demo_df[n].dropna(), [25, 50,75])
        q1 = np.round(q1,2)
        q2 = np.round(q2,2)
        q3 = np.round(q3,2)
        iqr = f'{q2} [{q1}, {q3}]'
        results.append([n, iqr])

    # Append value counts for each categorical demographic data
    for c in cat_demo:
        counts = demo_df[c].value_counts()
        results.append([c, counts])

    # Return the summary as a dataframe
    return pd.DataFrame(results)


def col_iqr(df, cols):
    '''
    Calculate the Interquartile Range (IQR) for specified columns of a dataframe.

    Parameters:
    - df (pd.DataFrame): The dataframe containing the data.
    - cols (list): A list of column names for which IQR needs to be calculated.

    Returns:
    - pd.DataFrame: A dataframe with each specified column's name and its corresponding IQR in a formatted string.
    '''
    
    results = []  # List to store results

    for n in cols:
        print(n)
        
        # Calculate the 25th, 50th, and 75th percentiles
        q1, q2, q3 = np.percentile(df[n].dropna(), [25, 50, 75])
        
        # Round the percentile values to two decimal places
        q1 = np.round(q1, 2)
        q2 = np.round(q2, 2)
        q3 = np.round(q3, 2)
        
        # Format the median and IQR in a string
        iqr = f'{q2} [{q1}, {q3}]'
        print(iqr)
        
        results.append([n, iqr])

    # Return the results as a dataframe
    return pd.DataFrame(results)
