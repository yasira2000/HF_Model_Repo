# import the necessary libraries
from multiprocessing import Manager, Value
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import math
from scipy import interpolate
from tqdm import tqdm

from utils.drawPlots import plot_ohlc_segment

original_pattern_name_list = [
    'Double Top, Adam and Adam', 
    'Double Top, Adam and Eve', 
    'Double Top, Eve and Eve', 
    'Double Top, Eve and Adam',
    'Double Bottom, Adam and Adam', 
    'Double Bottom, Eve and Adam', 
    'Double Bottom, Eve and Eve', 
    'Double Bottom, Adam and Eve',
    'Triangle, symmetrical', 
    'Head-and-shoulders top', 
    'Head-and-shoulders bottom', 
    'Flag, high and tight'
]

# Updated pattern encoding
pattern_encoding = {
    'Double Top': 0,
    'Double Bottom': 1,
    'Triangle, symmetrical': 2,
    'Head-and-shoulders top': 3,
    'Head-and-shoulders bottom': 4,
    'Flag, high and tight': 5,
    'No Pattern': 6
}

def get_pattern_encoding():
    return pattern_encoding

def get_reverse_pattern_encoding():
    return {v: k for k, v in pattern_encoding.items()}

def get_patetrn_name_by_encoding(encoding):
    """
    Get the pattern name by encoding.
    
    # Input:
    - encoding (int): The encoding of the pattern.
    
    # Returns:
    - str: The name of the pattern.
    """
    return get_reverse_pattern_encoding().get(encoding, 'Unknown Pattern')

def get_pattern_encoding_by_name(name):
    """
    Get the pattern encoding by name.
    
    # Input:
    - name (str): The name of the pattern.
    
    # Returns:
    - int: The encoding of the pattern.
    """
    return get_pattern_encoding().get(name, -1)

def get_pattern_list():
    return list(pattern_encoding.keys())

def filter_to_get_selected_patterns(df):
    # Filter dataframe to only include selected patterns
    df = df[df['Chart Pattern'].isin(original_pattern_name_list)].copy()  # Explicit copy to avoid warning
    
    # Replace all variations of Double Top and Double Bottom with simplified names
    double_top_variations = {
        'Double Top, Adam and Adam': 'Double Top',
        'Double Top, Adam and Eve': 'Double Top',
        'Double Top, Eve and Eve': 'Double Top',
        'Double Top, Eve and Adam': 'Double Top'
    }

    double_bottom_variations = {
        'Double Bottom, Adam and Adam': 'Double Bottom',
        'Double Bottom, Eve and Adam': 'Double Bottom',
        'Double Bottom, Eve and Eve': 'Double Bottom',
        'Double Bottom, Adam and Eve': 'Double Bottom'
    }
    
    # Combine all variations into a single mapping
    pattern_mapping = {**double_top_variations, **double_bottom_variations}

    # Use .loc[] to modify the dataframe safely
    df.loc[:, 'Chart Pattern'] = df['Chart Pattern'].replace(pattern_mapping)
    
    return df

def normalize_dataset(dataset):
    # calculate the min values from Low column and max values from High column for each instance
    min_low = dataset.groupby(level='Instance')['Low'].transform('min')
    max_high = dataset.groupby(level='Instance')['High'].transform('max')
    
    # OHLC columns to normalize
    ohlc_columns = ['Open', 'High', 'Low', 'Close']
    
    dataset_normalized = dataset.copy()
    
    # Apply the normalization formula to all columns in one go
    dataset_normalized[ohlc_columns] = (dataset_normalized[ohlc_columns] - min_low.values[:, None]) / (max_high.values[:, None] - min_low.values[:, None])
    
    # if there is a Volume column normalize it
    if 'Volume' in dataset.columns:
        # calculate the min values from Volume column and max values from Volume column for each instance
        min_volume = dataset.groupby(level='Instance')['Volume'].transform('min')
        max_volume = dataset.groupby(level='Instance')['Volume'].transform('max')
        
        # Normalize the Volume column
        dataset_normalized['Volume'] = (dataset_normalized['Volume'] - min_volume.values) / (max_volume.values - min_volume)
    
    
    return dataset_normalized    

def normalize_ohlc_segment(dataset):
    # calculate the min values from Low column and max values from High column for each instance
    min_low = dataset['Low'].min()
    max_high = dataset['High'].max()
    
    # OHLC columns to normalize
    ohlc_columns = ['Open', 'High', 'Low', 'Close']
    
    dataset_normalized = dataset.copy()
    
    if (max_high - min_low) != 0:
        # Apply the normalization formula to all columns in one go
        dataset_normalized[ohlc_columns] = (dataset_normalized[ohlc_columns] - min_low) / (max_high - min_low)
    else :
        print("Error: Max high and min low are equal")
    
    # if there is a Volume column normalize it
    if 'Volume' in dataset.columns:
        # calculate the min values from Volume column and max values from Volume column for each instance
        min_volume = dataset['Volume'].min()
        max_volume = dataset['Volume'].max()
        
        if (max_volume - min_volume) != 0:
            # Normalize the Volume column
            dataset_normalized['Volume'] = (dataset_normalized['Volume'] - min_volume) / (max_volume - min_volume)
        else:
            print("Error: Max volume and min volume are equal")
    
    
    return dataset_normalized
   
def process_row_improved(idx, row, ohlc_df, instance_counter, lock, successful_instances, instance_index_mapping):
    try:
        # Extract info and filter data
        start_date = pd.to_datetime(row['Start'])
        end_date = pd.to_datetime(row['End'])
        
        symbol_df_filtered = ohlc_df[(ohlc_df['Date'] >= start_date) & 
                                    (ohlc_df['Date'] <= end_date)]
        
        if symbol_df_filtered.empty:
            print(f"Empty result for {row['Symbol']} from {start_date} to {end_date}")
            return None
        
        # Get unique instance ID
        with lock:
            unique_instance = instance_counter.value
            instance_counter.value += 1
            
            # Explicitly add to instance_index_mapping using string key conversion
            instance_index_mapping[unique_instance] = idx
            
            # Track successful instances
            successful_instances.append(unique_instance)
        
        # Setup MultiIndex
        symbol_df_filtered = symbol_df_filtered.reset_index(drop=True)
        multi_index = pd.MultiIndex.from_arrays(
            [[unique_instance] * len(symbol_df_filtered), range(len(symbol_df_filtered))],
            names=["Instance", "Time"]
        )
        symbol_df_filtered.index = multi_index
        
        # Set index levels to proper types
        symbol_df_filtered.index = symbol_df_filtered.index.set_levels(
            symbol_df_filtered.index.levels[0].astype('int'), level=0
        )
        symbol_df_filtered.index = symbol_df_filtered.index.set_levels(
            symbol_df_filtered.index.levels[1].astype('int64'), level=1
        )
        
        # Add pattern and clean up
        symbol_df_filtered['Pattern'] = pattern_encoding[row['Chart Pattern']]
        symbol_df_filtered.drop('Date', axis=1, inplace=True)
        if 'Adj Close' in symbol_df_filtered.columns:
            symbol_df_filtered.drop('Adj Close', axis=1, inplace=True)
        
        # Normalize
        symbol_df_filtered = normalize_ohlc_segment(symbol_df_filtered)
        
        return symbol_df_filtered
    
    except Exception as e:
        print(f"Error processing {row['Symbol']}: {str(e)}")
        return None

def dataset_format(filteredPatternDf, give_instance_index_mapping=False):
    """
    Formats and preprocesses the dataset with better tracking of successful instances.
    """
    # Get symbol list from files
    folder_path = 'Datasets/OHLC data/'
    file_list = os.listdir(folder_path)
    symbol_list = [file[:-4] for file in file_list if file.endswith('.csv')]
    
    # Check for missing symbols
    symbols_in_df = filteredPatternDf['Symbol'].unique()
    missing_symbols = set(symbols_in_df) - set(symbol_list)
    if missing_symbols:
        print("Missing symbols: ", missing_symbols)
    
    # Create a list of tasks (symbol, row pairs)
    tasks = []
    for symbol in symbols_in_df:
        if symbol in symbol_list:  # Skip missing symbols
            filteredPatternDf_for_symbol = filteredPatternDf[filteredPatternDf['Symbol'] == symbol]
            file_path = os.path.join(folder_path, f"{symbol}.csv")
            
            # Pre-load symbol data
            try:
                symbol_df = pd.read_csv(file_path)
                symbol_df['Date'] = pd.to_datetime(symbol_df['Date'])
                symbol_df['Date'] = symbol_df['Date'].dt.tz_localize(None)
                
                for idx, row in filteredPatternDf_for_symbol.iterrows():
                    tasks.append((idx, row, symbol_df))
            except Exception as e:
                print(f"Error loading {symbol}: {str(e)}")
    
    print(f"Processing {len(tasks)} tasks in parallel...")
    
    # Process all tasks with instance tracking
    with Manager() as manager:
        instance_counter = manager.Value('i', 0)
        lock = manager.Lock()
        successful_instances = manager.list()  # Track which instances succeed
        instance_index_mapping = manager.dict()  # Mapping from instance ID to index
        
        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(process_row_improved)(task_idx, row, df, instance_counter, lock, successful_instances, instance_index_mapping) 
            for task_idx, row, df in tasks
        )
        
        # Filter out None results
        results = [result for result in results if result is not None]
        
        print(f"Total tasks: {len(tasks)}, Successful: {len(results)}")
        print(f"Instance counter final value: {instance_counter.value}")
        print(f"Number of successful instances: {len(successful_instances)}")
        
        # # Debug print for mapping
        # print("Debug - Instance Index Mapping:")
        # for k, v in instance_index_mapping.items():
        #     print(f"Key: {k}, Value: {v}")
        
        if len(successful_instances) < instance_counter.value:
            print("Warning: Some instances were assigned but their tasks failed")
        
        # Concatenate results and renumber instances if needed
        if results:
            dataset = pd.concat(results)
            dataset = dataset.sort_index(level=0)
            
            # Replace inf/nan values
            dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
            dataset.fillna(method='ffill', inplace=True)
            
            if give_instance_index_mapping:
                # Convert manager.dict to a regular dictionary
                instance_index_mapping_dict = dict(instance_index_mapping)
                
                print("Converted Mapping:", instance_index_mapping_dict)
                return dataset, instance_index_mapping_dict
            else:
                return dataset
        else:
            return pd.DataFrame()
        



def width_augmentation (filteredPatternDf, min_aug_len , aug_len_fraction, make_duplicates = False , keep_original = False):
    """
    Perform width augmentation on the filtered pattern DataFrame.
    
    # Input: 
    - filteredPatternDf (pd.DataFrame): The filtered pattern DataFrame.
    - min_aug_len (int): The minimum length of the augmented data.
    - aug_len_fraction (float): The fraction of the original data size to determine the maximum length of the augmented data.
    - make_duplicates (bool): Flag to indicate whether to make duplicates of patterns to reduce dataset imbalance.(make this false on test data)
    - keep_original (bool): Flag to indicate whether to keep the original patterns in the augmented DataFrame.
    
    # Returns:
    - filteredPattern_width_aug_df (pd.DataFrame): The DataFrame with width-augmented patterns.

    """
    
    filteredPattern_width_aug_df = pd.DataFrame(columns=filteredPatternDf.columns)
    
    print('Performing width augmentation...')
    # print('Number of patterns:', len(filteredPatternDf))

    # loop through the rows of filteredPatternDf
    for index, row in tqdm(filteredPatternDf.iterrows(), total=len(filteredPatternDf), desc="Processing"):

        symbol = row['Symbol']
        start_date = row['Start']
        end_date = row['End']
        pattern = row['Chart Pattern']
        
        ohlc_df = pd.read_csv(f'Datasets/OHLC data/{symbol}.csv')
        # Ensure all datetime objects are timezone-naive
        ohlc_df['Date'] = pd.to_datetime(ohlc_df['Date']).dt.tz_localize(None)

        # Convert start_date and end_date to timezone-naive if they have a timezone
        start_date = pd.to_datetime(start_date).tz_localize(None)
        end_date = pd.to_datetime(end_date).tz_localize(None)

        ohlc_of_interest = ohlc_df[(ohlc_df['Date'] >= start_date) & (ohlc_df['Date'] <= end_date)]
        data_size = len(ohlc_of_interest)
        
        if data_size <= 0:
            print (f'No data for {symbol} between {start_date} and {end_date}')
            continue
        
        # index of ohlc data on the start date and end date
        start_index = ohlc_of_interest.index[0]
        end_index = ohlc_of_interest.index[-1]
        
        min_possible_index = 0
        max_possible_index = len(ohlc_df) - 1
        
        number_of_rows_for_pattern= filteredPatternDf['Chart Pattern'].value_counts()[pattern]
        max_num_of_rows_for_pattern = filteredPatternDf['Chart Pattern'].value_counts().max()
        
        # to make the number of rows for each pattern equal to reduce the imbalance in the dataset
        if make_duplicates:
            num_row_diff = (max_num_of_rows_for_pattern - number_of_rows_for_pattern)*2
            
            multiplier = math.ceil(num_row_diff / number_of_rows_for_pattern) +2
            # print ('Pattern :', pattern , 'Multiplier :' , multiplier , 'Number of rows for pattern :', number_of_rows_for_pattern)
            # get a random mvalue between 1 to multiplier
            m = np.random.randint(1, multiplier)
        else:
            m = 1
            
        for i in range(m):
            max_aug_len = math.ceil(data_size * aug_len_fraction)
            if max_aug_len < min_aug_len:
                max_aug_len = min_aug_len
            aug_len_l = np.random.randint(1, max_aug_len)
            aug_len_r = np.random.randint(1, max_aug_len)
            
            # get the start and end index of the augmented data
            start_index_aug = start_index - aug_len_l
            end_index_aug = end_index + aug_len_r
            
            if start_index_aug < min_possible_index:
                start_index_aug = min_possible_index
            if end_index_aug > max_possible_index:
                end_index_aug = max_possible_index
            
            # get the date of the start and end index of the augmented data
            start_date_aug = ohlc_df.iloc[start_index_aug]['Date']
            end_date_aug = ohlc_df.iloc[end_index_aug]['Date']
            
            # create a new row for the augmented data
            new_row = row.copy()
            new_row['Start'] = start_date_aug
            new_row['End'] = end_date_aug
            filteredPattern_width_aug_df = pd.concat([filteredPattern_width_aug_df, pd.DataFrame([new_row])], ignore_index=True)
        
        if keep_original:
            # concat the original row too
            filteredPattern_width_aug_df = pd.concat([filteredPattern_width_aug_df, pd.DataFrame([row])], ignore_index=True)
            
    return filteredPattern_width_aug_df

def normalize_ohlc_len(df, target_len=30 , plot_count= 0):
    
    instances_list = df.index.get_level_values(0).unique()
    normalized_df_list = []
    
    # pick 10 random instances from the list of instances to plot
    random_indices = np.random.choice(len(instances_list), plot_count, replace=False)
    
    for instance in instances_list:
    
        sample = df.loc[instance]
        
        pattern_df = sample.copy()
        new_data = {} 
        orig_indices = pattern_df.index.values  # Changed this line
        new_indices = np.linspace(0, len(orig_indices) - 1, target_len) 

        # First interpolate all numerical columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            # Determine the best interpolation method based on data length
            if len(orig_indices) >= 4:  # Enough points for cubic
                kind = 'cubic'
            elif len(orig_indices) >= 3:  # Can use quadratic
                kind = 'quadratic'
            elif len(orig_indices) >= 2:  # Can use linear
                kind = 'linear'
            else:  # Not enough points, use nearest
                kind = 'nearest'
                
            f = interpolate.interp1d(np.arange(len(orig_indices)), pattern_df[col].values,  
                                  kind=kind, bounds_error=False, fill_value='extrapolate') 
            # Apply interpolation function to get new values 
            new_data[col] = f(new_indices)

        # Ensure all OHLC values are positive
        for col in ['Open', 'High', 'Low', 'Close']:
            new_data[col] = np.maximum(new_data[col], 0.001)  # Small positive value instead of zero

        # Fix OHLC relationships
        for i in range(len(new_indices)):
            # Ensure High is the maximum
            new_data['High'][i] = max(new_data['High'][i], new_data['Open'][i], new_data['Close'][i])
            
            # Ensure Low is the minimum
            new_data['Low'][i] = min(new_data['Low'][i], new_data['Open'][i], new_data['Close'][i])

        # Handle categorical data separately
        if 'Pattern' in pattern_df.columns:
            f = interpolate.interp1d(np.arange(len(orig_indices)), pattern_df['Pattern'].values,  
                                   kind='nearest', bounds_error=False, fill_value=pattern_df['Pattern'].iloc[0])
            new_data['Pattern'] = f(new_indices)

        result_df = pd.DataFrame(new_data)
        result_df.index = pd.MultiIndex.from_product([[instance], result_df.index])
        normalized_df_list.append(result_df)
        
        if instance in instances_list[random_indices]:  # Fixed this line
            # plot results
            plot_ohlc_segment(pattern_df)
            plot_ohlc_segment(result_df)
    
    combined_result_df = pd.concat(normalized_df_list, axis=0)  # Fixed this line
    return combined_result_df

# Define features, target, and desired series length
features = ['Open', 'High', 'Low', 'Close', 'Volume']
target   = 'Pattern'
series_length = 100

# This function pads or truncates every instance to length=100,
# then stacks into an array of shape (n_instances, n_features, series_length)
def prepare_rocket_data(dataset, features = features, target = target, series_length = series_length):
    def adjust_series_length(group):
        arr = group[features].values
        if len(arr) > series_length:
            return arr[:series_length]
        padding = np.zeros((series_length - len(arr), arr.shape[1]))
        return np.vstack([arr, padding])
    
    # Apply per-instance adjustment
    adjusted = dataset.groupby(level=0).apply(adjust_series_length)
    X = np.stack(adjusted.values)              # (n_instances, series_length, n_features)
    X = np.transpose(X, (0, 2, 1))              # → (n_instances, n_features, series_length)
    
    y = dataset.groupby(level=0)[target].first().values
    return X, y