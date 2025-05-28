# handler.py
import joblib
import pandas as pd
import numpy as np
import math
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
import os # For accessing model path

# Import your utility functions
# Make sure your utils directory is alongside handler.py
# and contains __init__.py, eval.py, formatAndPreprocessNewPatterns.py
from utils.eval import intersection_over_union
from utils.formatAndPreprocessNewPatterns import get_patetrn_name_by_encoding, get_pattern_encoding_by_name, get_reverse_pattern_encoding

# --- Global Model Loading (Crucial for performance) ---
# This model will be loaded ONLY ONCE when the server starts.
# Ensure the path is correct relative to where handler.py runs in the container.
# The `MODEL_DIR` env var is automatically set by Inference Endpoints.
# If you place 'Models/' directly in your repo root, it will be at /repository/Models/
# If you place it outside (not recommended), you'd need to adjust paths.
# For simplicity, assume `Models/` is in the root of your HF repo.
MODEL_PATH = os.path.join(os.environ.get("MODEL_DIR", "."), "Models", "Width Aug OHLC_mini_rocket_xgb.joblib")

# Load the model globally
try:
    print(f"Loading model from: {MODEL_PATH}")
    rocket_model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # In a real scenario, you might want to raise an exception to prevent the server from starting
    rocket_model = None

# --- Helper functions (from your provided code) ---
# Paste your `process_window`, `parallel_process_sliding_window`,
# `prepare_dataset_for_cluster`, `cluster_windows` here.
# Make sure they are defined before `locate_patterns`
# because locate_patterns depends on them.

# Make sure these globals are outside functions if they are truly global constants
pattern_encoding_reversed = get_reverse_pattern_encoding()
# model is now `rocket_model` loaded globally
# plot_count is handled by the API input now
win_size_proportions = np.round(np.logspace(0, np.log10(20), num=10), 2).tolist()
padding_proportion = 0.6
stride = 1
probab_threshold_list = 0.5
prob_threshold_of_no_pattern_to_mark_as_no_pattern = 0.5
target_len = 30 # Not used in your current code

eps=0.04
min_samples=3
win_width_proportion=10 # Not used in your current code


def process_window(i, ohlc_data_segment, rocket_model, probability_threshold, pattern_encoding_reversed,seg_start, seg_end, window_size, padding_proportion,prob_threshold_of_no_pattern_to_mark_as_no_pattern=1):
    start_index = i - math.ceil(window_size * padding_proportion)
    end_index = start_index + window_size

    start_index = max(start_index, 0)
    end_index = min(end_index, len(ohlc_data_segment))

    ohlc_segment = ohlc_data_segment[start_index:end_index]
    if len(ohlc_segment) == 0:
        return None  # Skip empty segments
    win_start_date = ohlc_segment['Date'].iloc[0]
    win_end_date = ohlc_segment['Date'].iloc[-1]

    ohlc_array_for_rocket = ohlc_segment[['Open', 'High', 'Low', 'Close','Volume']].to_numpy().reshape(1, len(ohlc_segment), 5)
    ohlc_array_for_rocket = np.transpose(ohlc_array_for_rocket, (0, 2, 1))
    try:
        pattern_probabilities = rocket_model.predict_proba(ohlc_array_for_rocket)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None
    max_probability = np.max(pattern_probabilities)
    no_pattern_proba = pattern_probabilities[0][get_pattern_encoding_by_name ('No Pattern')]
    pattern_index = np.argmax(pattern_probabilities)

    pred_proba = max_probability
    pred_pattern = get_patetrn_name_by_encoding(pattern_index)
    if no_pattern_proba > prob_threshold_of_no_pattern_to_mark_as_no_pattern:
        pred_proba = no_pattern_proba
        pred_pattern = 'No Pattern'

    new_row = {
        'Start': win_start_date, 'End': win_end_date,  'Chart Pattern': pred_pattern,  'Seg_Start': seg_start, 'Seg_End': seg_end ,
        'Probability': pred_proba
    }
    return new_row


def parallel_process_sliding_window(ohlc_data_segment, rocket_model, probability_threshold, stride, pattern_encoding_reversed, window_size, padding_proportion,prob_threshold_of_no_pattern_to_mark_as_no_pattern=1,parallel=True,num_cores=-1):
    seg_start = ohlc_data_segment['Date'].iloc[0]
    seg_end = ohlc_data_segment['Date'].iloc[-1]

    # Render.com's worker environment for the HF endpoint will have limited cores for single instances.
    # Parallel processing (`joblib.Parallel`) within the *single* HF endpoint worker
    # might not yield significant benefits or might even cause issues if not configured carefully.
    # It's generally better to rely on HF's scaling for multiple requests.
    # Consider setting `parallel=False` or `num_cores=1` for initial deployment if you hit issues.
    # For now, let's keep it as is, but be mindful of resource constraints.

    if parallel:
        with Parallel(n_jobs=num_cores, verbose=0) as parallel: # verbose=0 to reduce log spam
            results = parallel(
                delayed(process_window)(
                    i=i,
                    ohlc_data_segment=ohlc_data_segment,
                    rocket_model=rocket_model,
                    probability_threshold=probability_threshold,
                    pattern_encoding_reversed=pattern_encoding_reversed,
                    window_size=window_size,
                    seg_start=seg_start,
                    seg_end=seg_end,
                    padding_proportion=padding_proportion,
                    prob_threshold_of_no_pattern_to_mark_as_no_pattern=prob_threshold_of_no_pattern_to_mark_as_no_pattern
                )
                for i in range(0, len(ohlc_data_segment), stride)
            )
        return pd.DataFrame([res for res in results if res is not None])
    else:
        results = []
        for i_idx, i in enumerate(range(0, len(ohlc_data_segment), stride)):
            res = process_window(i, ohlc_data_segment, rocket_model, probability_threshold, pattern_encoding_reversed, seg_start, seg_end, window_size, padding_proportion)
            if res is not None:
                results.append(res)
        return pd.DataFrame(results)

def prepare_dataset_for_cluster(ohlc_data_segment, win_results_df):
    predicted_patterns = win_results_df.copy()
    # origin_date = ohlc_data_segment['Date'].min() # Not used
    for index, row in predicted_patterns.iterrows():
        pattern_start = row['Start']
        pattern_end = row['End']
        start_point_index = len(ohlc_data_segment[ohlc_data_segment['Date'] < pattern_start])
        pattern_len = len(ohlc_data_segment[(ohlc_data_segment['Date'] >= pattern_start) & (ohlc_data_segment['Date'] <= pattern_end)])
        pattern_mid_index = start_point_index + (pattern_len / 2)
        predicted_patterns.at[index, 'Center'] = pattern_mid_index
        predicted_patterns.at[index, 'Pattern_Start_pos'] = start_point_index
        predicted_patterns.at[index, 'Pattern_End_pos'] = start_point_index + pattern_len
    return predicted_patterns

def cluster_windows(predicted_patterns , probability_threshold, window_size,eps = 0.05 , min_samples = 2):
    df = predicted_patterns.copy()

    if isinstance(probability_threshold, list):
        for i in range(len(probability_threshold)):
            pattern_name = get_patetrn_name_by_encoding(i)
            df.drop(df[(df['Chart Pattern'] == pattern_name) & (df['Probability'] < probability_threshold[i])].index, inplace=True)
    else:
        df = df[df['Probability'] > probability_threshold]

    cluster_labled_windows = []
    interseced_clusters = []

    if df.empty: # Handle case where df might be empty after filtering
        return None, None

    min_center = df['Center'].min()
    max_center = df['Center'].max()

    for pattern, group in df.groupby('Chart Pattern'):
        centers = group['Center'].values.reshape(-1, 1)

        if min_center < max_center:
            norm_centers = (centers - min_center) / (max_center - min_center)
        else:
            norm_centers = np.ones_like(centers)

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(norm_centers)
        group['Cluster'] = db.labels_
        cluster_labled_windows.append(group)

        for cluster_id, cluster_group in group[group['Cluster'] != -1].groupby('Cluster'):
            expanded_dates = []
            for _, row in cluster_group.iterrows():
                dates = pd.date_range(row["Start"], row["End"])
                expanded_dates.extend(dates)

            date_counts = pd.Series(expanded_dates).value_counts().sort_index()
            cluster_start = date_counts[date_counts >= 2].index.min()
            cluster_end = date_counts[date_counts >= 2].index.max()

            interseced_clusters.append({
                'Chart Pattern': pattern,
                'Cluster': cluster_id,
                'Start': cluster_start,
                'End': cluster_end,
                'Seg_Start': cluster_group['Seg_Start'].iloc[0],
                'Seg_End': cluster_group['Seg_End'].iloc[0],
                'Avg_Probability': cluster_group['Probability'].mean(),
            })

    if len(cluster_labled_windows) == 0 or len(interseced_clusters) == 0:
        return None, None

    cluster_labled_windows_df = pd.concat(cluster_labled_windows)
    interseced_clusters_df = pd.DataFrame(interseced_clusters)
    cluster_labled_windows_df = cluster_labled_windows_df.sort_index()
    return cluster_labled_windows_df, interseced_clusters_df


# ========================= locate_patterns function ==========================

# This will be your primary inference function called by the HF endpoint.
class InferenceHandler:
    def __init__(self):
        # Model is loaded globally, so it's accessible here
        self.model = rocket_model
        if self.model is None:
            raise ValueError("ML model failed to load during initialization.")

        # Initialize other global parameters here as well
        self.pattern_encoding_reversed = pattern_encoding_reversed
        self.win_size_proportions = win_size_proportions
        self.padding_proportion = padding_proportion
        self.stride = stride
        self.probab_threshold_list = probab_threshold_list
        self.prob_threshold_of_no_pattern_to_mark_as_no_pattern = prob_threshold_of_no_pattern_to_mark_as_no_pattern
        self.eps = eps
        self.min_samples = min_samples

    def __call__(self, inputs):
        """
        Main inference method for the Hugging Face Inference Endpoint.
        Args:
            inputs: A dictionary or list of dictionaries representing the input data.
                    For your case, this will be the OHLC data sent from Django.
                    Expected format: [{"Date": "YYYY-MM-DD", "Open": ..., "High": ..., ...}, ...]
        Returns:
            A list of dictionaries representing the detected patterns.
        """
        if not self.model:
            raise ValueError("ML model is not loaded. Cannot perform inference.")

        # Ensure inputs is a list of dictionaries if not already
        if isinstance(inputs, dict):
            inputs = [inputs] # Handle single input dict if needed

        # Convert input (list of dicts) to pandas DataFrame
        try:
            ohlc_data = pd.DataFrame(inputs)
            # Ensure 'Date' is datetime, it might come as string from JSON
            ohlc_data['Date'] = pd.to_datetime(ohlc_data['Date'])
            # Ensure proper columns exist
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in ohlc_data.columns for col in required_cols):
                raise ValueError(f"Missing required columns in input data. Expected: {required_cols}, Got: {ohlc_data.columns.tolist()}")

        except Exception as e:
            print(f"Error processing input data: {e}")
            raise ValueError(f"Invalid input data format: {e}")


        ohlc_data_segment = ohlc_data.copy()
        seg_len = len(ohlc_data_segment)

        if ohlc_data_segment.empty:
            raise ValueError("OHLC Data segment is empty or invalid after processing.")

        win_results_for_each_size = []
        located_patterns_and_other_info_for_each_size = []
        cluster_labled_windows_list = []

        used_win_sizes = []
        win_iteration = 0

        for win_size_proportion in self.win_size_proportions:
            window_size = seg_len // win_size_proportion
            if window_size < 10:
                window_size = 10
            window_size = int(window_size)
            if window_size in used_win_sizes:
                continue
            used_win_sizes.append(window_size)

            # Pass the globally loaded model `self.model`
            win_results_df = parallel_process_sliding_window(
                ohlc_data_segment,
                self.model,
                self.probab_threshold_list,
                self.stride,
                self.pattern_encoding_reversed,
                window_size,
                self.padding_proportion,
                self.prob_threshold_of_no_pattern_to_mark_as_no_pattern,
                parallel=True, # You might want to test with False/num_cores=1 on HF to avoid internal parallelism issues
                num_cores=-1 # -1 means all available cores; on HF, this will be limited by the instance type
            )

            if win_results_df is None or win_results_df.empty:
                print(f"Window results dataframe is empty for window size {window_size}")
                continue
            win_results_df['Window_Size'] = window_size
            win_results_for_each_size.append(win_results_df)

            predicted_patterns = prepare_dataset_for_cluster(ohlc_data_segment, win_results_df)
            if predicted_patterns is None or predicted_patterns.empty:
                print("Predicted patterns dataframe is empty")
                continue

            # Pass eps and min_samples from handler's state
            cluster_labled_windows_df , interseced_clusters_df = cluster_windows(
                predicted_patterns,
                self.probab_threshold_list,
                window_size,
                eps=self.eps,
                min_samples=self.min_samples
            )

            if cluster_labled_windows_df is None or interseced_clusters_df is None or cluster_labled_windows_df.empty or interseced_clusters_df.empty:
                print("Clustered windows dataframe is empty")
                continue
            mask = cluster_labled_windows_df['Cluster'] != -1
            cluster_labled_windows_df.loc[mask, 'Cluster'] = cluster_labled_windows_df.loc[mask, 'Cluster'].astype(int) + win_iteration
            interseced_clusters_df['Cluster'] = interseced_clusters_df['Cluster'].astype(int) + win_iteration
            num_of_unique_clusters = interseced_clusters_df[interseced_clusters_df['Cluster']!=-1]['Cluster'].nunique()
            win_iteration += num_of_unique_clusters
            cluster_labled_windows_list.append(cluster_labled_windows_df)

            interseced_clusters_df['Calc_Start'] = interseced_clusters_df['Start']
            interseced_clusters_df['Calc_End'] = interseced_clusters_df['End']
            located_patterns_and_other_info = interseced_clusters_df.copy()

            if located_patterns_and_other_info is None or located_patterns_and_other_info.empty:
                print("Located patterns and other info dataframe is empty")
                continue
            located_patterns_and_other_info['Window_Size'] = window_size

            located_patterns_and_other_info_for_each_size.append(located_patterns_and_other_info)

        if located_patterns_and_other_info_for_each_size is None or not located_patterns_and_other_info_for_each_size:
            print("Located patterns and other info for each size is empty")
            return [] # Return empty list if no patterns found

        located_patterns_and_other_info_for_each_size_df = pd.concat(located_patterns_and_other_info_for_each_size)

        unique_window_sizes = located_patterns_and_other_info_for_each_size_df['Window_Size'].unique()
        unique_patterns = located_patterns_and_other_info_for_each_size_df['Chart Pattern'].unique()
        unique_window_sizes = np.sort(unique_window_sizes)[::-1]

        filtered_loc_pat_and_info_rows_list = []

        for chart_pattern in unique_patterns:
            located_patterns_and_other_info_for_each_size_df_chart_pattern = located_patterns_and_other_info_for_each_size_df[located_patterns_and_other_info_for_each_size_df['Chart Pattern'] == chart_pattern]
            for win_size in unique_window_sizes:
                located_patterns_and_other_info_for_each_size_df_win_size_chart_pattern = located_patterns_and_other_info_for_each_size_df_chart_pattern[located_patterns_and_other_info_for_each_size_df_chart_pattern['Window_Size'] == win_size]
                for idx , row in located_patterns_and_other_info_for_each_size_df_win_size_chart_pattern.iterrows():
                    start_date = row['Calc_Start']
                    end_date = row['Calc_End']
                    is_already_included = False
                    intersecting_rows = located_patterns_and_other_info_for_each_size_df_chart_pattern[
                                                        (located_patterns_and_other_info_for_each_size_df_chart_pattern['Calc_Start'] <= end_date) &
                                                        (located_patterns_and_other_info_for_each_size_df_chart_pattern['Calc_End'] >= start_date)
                                                    ]
                    is_already_included = False
                    for idx2, row2 in intersecting_rows.iterrows():
                        iou = intersection_over_union(start_date, end_date, row2['Calc_Start'], row2['Calc_End'])

                        if iou > 0.6:
                            if row2['Window_Size'] > row['Window_Size']:
                                if (row['Avg_Probability'] - row2['Avg_Probability']) > 0.1:
                                    is_already_included = False
                                else:
                                    is_already_included = True
                                    break
                            elif row['Window_Size'] >= row2['Window_Size']:
                                if (row2['Avg_Probability'] - row['Avg_Probability']) > 0.1:
                                    is_already_included = True
                                    break
                                else:
                                    is_already_included = False

                    if not is_already_included:
                        filtered_loc_pat_and_info_rows_list.append(row)

        filtered_loc_pat_and_info_df = pd.DataFrame(filtered_loc_pat_and_info_rows_list)

        # Convert datetime columns to string format for serialization before returning
        datetime_columns = ['Start', 'End', 'Seg_Start', 'Seg_End', 'Calc_Start', 'Calc_End']
        for col in datetime_columns:
            if col in filtered_loc_pat_and_info_df.columns:
                if pd.api.types.is_datetime64_any_dtype(filtered_loc_pat_and_info_df[col]):
                    filtered_loc_pat_and_info_df[col] = pd.to_datetime(filtered_loc_pat_and_info_df[col]).dt.strftime('%Y-%m-%d')
                elif not filtered_loc_pat_and_info_df[col].empty and isinstance(filtered_loc_pat_and_info_df[col].iloc[0], str):
                    pass
                else:
                    filtered_loc_pat_and_info_df[col] = filtered_loc_pat_and_info_df[col].astype(str)

        # Return as a list of dictionaries (JSON serializable)
        return filtered_loc_pat_and_info_df.to_dict('records')