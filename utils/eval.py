import sys
# from matplotlib import pyplot as plt
# from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


def intersection_over_union(start1, end1, start2, end2):
    """
    Compute Intersection over Union (IoU) between two date ranges.
    """
    latest_start = max(start1, start2)
    earliest_end = min(end1, end2)
    overlap = max(0, (earliest_end - latest_start).days + 1)
    union = (end1 - start1).days + (end2 - start2).days + 2 - overlap
    return overlap / union if union > 0 else 0  # Avoid division by zero

def mean_abselute_error(start1, end1, start2, end2):
    """
    Compute Mean Absolute Error (MAE) between two date ranges.
    """
    # check if start or end are NAT 
    if start1 is pd.NaT or end1 is pd.NaT or start2 is pd.NaT or end2 is pd.NaT:
        print("One of the dates is NaT")
        print(f"start1: {start1}, end1: {end1}, start2: {start2}, end2: {end2}")
        return None
    return (abs(start1 - start2).days + abs(end1 - end2).days) / 2


def get_model_eval_res(located_patterns_and_other_info_updated_dict,window_results_dict,selected_models,selected_test_patterns_without_no_pattern):
    model_eval_results_dict = {}
    for model_name in selected_models:
        print(f"\n Selected model: {model_name}")
        
        located_patterns_and_other_info_updated_df = located_patterns_and_other_info_updated_dict[model_name]
        window_results_df = window_results_dict[model_name]

        # dictionary to store the count of properly located patterns , iou and mae for each properly detected pattern for each model
        

        # Dictionary to store the count of properly located patterns
        number_of_properly_located_patterns = {}
        iou_for_each_properly_detected_pattern = {}
        mae_for_each_properly_detected_pattern = {}

        # Convert date columns to datetime (once, outside the loop for efficiency)
        located_patterns_and_other_info_updated_df['Calc_Start'] = pd.to_datetime(located_patterns_and_other_info_updated_df['Calc_Start'])
        located_patterns_and_other_info_updated_df['Calc_End'] = pd.to_datetime(located_patterns_and_other_info_updated_df['Calc_End'])

        # Iterate over test patterns with progress bar
        for index, row in selected_test_patterns_without_no_pattern.iterrows():
            sys.stdout.write(f"\rProcessing row {index + 1}/{len(selected_test_patterns_without_no_pattern)}")
            sys.stdout.flush() 
            symbol = row['Symbol']
            chart_pattern = row['Chart Pattern']
            start_date = pd.to_datetime(row['Start']).tz_localize(None)
            end_date = pd.to_datetime(row['End']).tz_localize(None)
            
            # Filter for matching symbol and chart pattern
            located_patterns_for_this = located_patterns_and_other_info_updated_df[
                (located_patterns_and_other_info_updated_df['Symbol'] == symbol) &
                (located_patterns_and_other_info_updated_df['Chart Pattern'] == chart_pattern)
            ].copy()  # Use `.copy()` to avoid SettingWithCopyWarning
            
            if located_patterns_for_this.empty:
                continue  # Skip if no matching rows
            
            # Compute IoU for each row using .loc to avoid warnings
            located_patterns_for_this.loc[:, 'IoU'] = located_patterns_for_this.apply(
                lambda x: intersection_over_union(start_date, end_date, x['Calc_Start'], x['Calc_End']),
                axis=1
            )
            
            # Compute MAE for each row using .loc to avoid warnings
            located_patterns_for_this.loc[:, 'MAE'] = located_patterns_for_this.apply(
                lambda x: mean_abselute_error(start_date, end_date, x['Calc_Start'], x['Calc_End']),
                axis=1
            )

            
            # Filter based on IoU threshold (≥ 0.8)
            located_patterns_for_this_proper = located_patterns_for_this[located_patterns_for_this['IoU'] >= 0.25]
            
            if not located_patterns_for_this_proper.empty:
                number_of_properly_located_patterns[chart_pattern] = number_of_properly_located_patterns.get(chart_pattern, 0) + 1
                iou_for_each_properly_detected_pattern[chart_pattern] = iou_for_each_properly_detected_pattern.get(chart_pattern, 0) + max(located_patterns_for_this_proper['IoU'])
                mae_for_each_properly_detected_pattern[chart_pattern] = mae_for_each_properly_detected_pattern.get(chart_pattern, 0) + min(located_patterns_for_this_proper['MAE'])

        number_of_properly_located_patterns
        
        model_eval_results_dict[model_name] = {
            'number_of_properly_located_patterns': number_of_properly_located_patterns,
            'iou_for_each_properly_detected_pattern': iou_for_each_properly_detected_pattern,
            'mae_for_each_properly_detected_pattern': mae_for_each_properly_detected_pattern
        }
    return model_eval_results_dict

############################################################################################
# Evaluate multiple models and plot
############################################################################################
# Commenting out plotting functions
"""
def create_comprehensive_model_comparison(all_models_metrics):

    Create a comprehensive visualization comparing all models across all metrics,
    using nested concentric pie charts for Precision and Recall.
    
    Parameters:
    -----------
    all_models_metrics : dict
        Dictionary containing metrics for each model

    models = list(all_models_metrics.keys())
    n_models = len(models)
    
    # Define the metrics to include
    key_metrics = {
        'total_recall': 'Recall',
        'total_precision': 'Precision', 
        'overall_f1': 'F1 Score',
        'overall_iou': 'IoU',
        'overall_mae': 'MAE'
    }
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(20, 14))
    
    # Add main title with enough space for legend below it
    plt.suptitle('Comprehensive Model Evaluation', fontsize=16, y=0.98)
    
    # Define a color palette for models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    # Create a master legend below the title
    legend_handles = [plt.Line2D([0], [0], color=colors[i], lw=4, label=model) for i, model in enumerate(models)]
    fig.legend(
        handles=legend_handles,
        labels=models,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.93),  # Moved down from 0.98 to 0.93
        ncol=n_models,
        fontsize=12
    )
    
    # Adjust GridSpec to account for the title and legend
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1.2, 1.2, 1], top=0.88)  # Reduced top from 0.95 to 0.88
    

    
    # 1. Precision Nested Pie Chart - top left
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create a multi-layer nested pie chart for precision
    # Each ring represents a different model
    precision_values = [metrics['total_precision'] for metrics in all_models_metrics.values()]
    
    # Calculate radii for each ring (outermost ring is largest)
    radii = np.linspace(0.5, 1.0, n_models+1)[1:]  # start from second element to skip 0.5
    
    # Plot each model as a ring, outermost = first model
    for i, model in enumerate(models):
        # Create data for this model's ring [precision, 1-precision]
        data = [precision_values[i], 1-precision_values[i]]
        colors_ring = [colors[i], 'lightgray']
        
        # Create pie chart for this ring
        wedges, texts = ax1.pie(
            data, 
            radius=radii[i],
            colors=colors_ring,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.15, edgecolor='w')
        )
        
        # Add only the value (no model name) to the pie chart wedge
        angle = (wedges[0].theta1 + wedges[0].theta2) / 2
        x = (radii[i] - 0.075) * np.cos(np.radians(angle))
        y = (radii[i] - 0.075) * np.sin(np.radians(angle))
        ax1.text(x, y, f"{precision_values[i]:.3f}", 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Create center circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.25, fc='white')
    ax1.add_patch(centre_circle)
    
    ax1.set_title('Precision Comparison (Higher is Better)')
    ax1.set_aspect('equal')
    
    # 2. Recall Nested Pie Chart - top middle
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create a multi-layer nested pie chart for recall
    recall_values = [metrics['total_recall'] for metrics in all_models_metrics.values()]
    
    # Plot each model as a ring, outermost = first model
    for i, model in enumerate(models):
        # Create data for this model's ring [recall, 1-recall]
        data = [recall_values[i], 1-recall_values[i]]
        colors_ring = [colors[i], 'lightgray']
        
        # Create pie chart for this ring
        wedges, texts = ax2.pie(
            data, 
            radius=radii[i],
            colors=colors_ring,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.15, edgecolor='w')
        )
        
        # Add only the value (no model name) to the pie chart wedge
        angle = (wedges[0].theta1 + wedges[0].theta2) / 2
        x = (radii[i] - 0.075) * np.cos(np.radians(angle))
        y = (radii[i] - 0.075) * np.sin(np.radians(angle))
        ax2.text(x, y, f"{recall_values[i]:.3f}", 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Create center circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.25, fc='white')
    ax2.add_patch(centre_circle)
    
    ax2.set_title('Recall Comparison (Higher is Better)')
    ax2.set_aspect('equal')
    
    # 3. F1 Score and IoU - top right
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Prepare data for grouped bar chart
    metrics_to_plot = ['overall_f1', 'overall_iou']
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / n_models
    
    # Plot grouped bars for each model
    for i, (model_name, metrics) in enumerate(all_models_metrics.items()):
        values = [metrics[key] for key in metrics_to_plot]
        bars = ax3.bar(x + i*width - width*(n_models-1)/2, values, width, color=colors[i])
        
        # Add value labels above each bar
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9, rotation=0)
    
    # Customize the plot
    ax3.set_xticks(x)
    ax3.set_xticklabels([key_metrics[key] for key in metrics_to_plot])
    ax3.set_ylabel('Score')
    ax3.set_title('F1 Score & IoU Comparison (Higher is Better)')
    ax3.set_ylim(0, 1.0)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. MAE comparison (separate bar chart) - middle left
    ax4 = fig.add_subplot(gs[1, 0])
    
    mae_values = [metrics['overall_mae'] for metrics in all_models_metrics.values()]
    bars = ax4.bar(models, mae_values, color=colors)
    
    # Add value labels above MAE bars
    for bar, value in zip(bars, mae_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax4.set_ylabel('Error')
    ax4.set_title('Mean Absolute Error (Lower is Better)')
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 5. Model metrics radar chart - middle center
    ax5 = fig.add_subplot(gs[1, 1], polar=True)
    
    # Setup for radar chart
    metrics_for_radar = ['total_recall', 'total_precision', 'overall_f1', 'overall_iou']
    num_vars = len(metrics_for_radar)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Plot each model on the radar chart
    for i, (model_name, metrics) in enumerate(all_models_metrics.items()):
        values = [metrics[metric] for metric in metrics_for_radar]
        values += values[:1]  # Close the loop
        
        ax5.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i])
        ax5.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Set radar chart labels
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels([key_metrics[metric] for metric in metrics_for_radar])
    ax5.set_ylim(0, 1)
    ax5.set_title('Model Performance Radar Chart')
    
    # 6. Model comparison bar - middle right
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Calculate the average of the four main metrics for an overall score
    # (excluding MAE which is inverse, lower is better)
    overall_scores = []
    for model_name, metrics in all_models_metrics.items():
        score = (metrics['total_recall'] + metrics['total_precision'] + 
                metrics['overall_f1'] + metrics['overall_iou']) / 4
        overall_scores.append(score)
    
    # Create horizontal bar chart
    y_pos = np.arange(len(models))
    ax6.barh(y_pos, overall_scores, color=colors)
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(models)
    ax6.invert_yaxis()  # labels read top-to-bottom
    ax6.set_xlabel('Overall Performance Score')
    ax6.set_title('Overall Model Comparison (Higher is Better)')
    
    # Add value labels
    for i, v in enumerate(overall_scores):
        ax6.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    # 7. Detailed per-model metrics table - bottom span all columns
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('tight')
    ax7.axis('off')
    
    # Prepare table data
    table_data = []
    for model_name, metrics in all_models_metrics.items():
        row = [model_name]
        for key in key_metrics:
            row.append(f"{metrics[key]:.4f}")
        table_data.append(row)
    
    # Create table
    column_labels = ['Model'] + list(key_metrics.values())
    table = ax7.table(
        cellText=table_data,
        colLabels=column_labels,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax7.set_title('Model Metrics Summary Table')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.88])  # Adjusted rect to account for title and legend
    
    plt.show()
    
    return fig

# The evaluate_model and evaluate_all_models functions remain unchanged
# The evaluate_model and evaluate_all_models functions remain unchanged
# The evaluate_model function remains unchanged from your second code snippet
def evaluate_model(model_name, model_eval_results_dict, pattern_row_count, test_patterns, located_patterns_and_other_info_updated_dict):
   Evaluate a model and calculate metrics without redundant plots
    print(f"\n{'='*20} Model: {model_name} {'='*20}")
    
    # Extract model results
    number_of_properly_located_patterns = model_eval_results_dict[model_name]['number_of_properly_located_patterns']
    located_patterns_df = located_patterns_and_other_info_updated_dict[model_name]
    mae_for_each_properly_detected_pattern = model_eval_results_dict[model_name]['mae_for_each_properly_detected_pattern']
    iou_for_each_properly_detected_pattern = model_eval_results_dict[model_name]['iou_for_each_properly_detected_pattern']
    
    # Calculate metrics without plotting
    # Recall
    total_number_of_all_patterns = sum(pattern_row_count.values())
    total_number_of_properly_located_patterns = sum(number_of_properly_located_patterns.values())
    total_recall = total_number_of_properly_located_patterns / total_number_of_all_patterns if total_number_of_all_patterns > 0 else 0
    
    per_pattern_recall = {}
    for pattern, count in number_of_properly_located_patterns.items():
        pattern_count = test_patterns[test_patterns['Chart Pattern'] == pattern].shape[0]
        if pattern_count > 0:
            per_pattern_recall[pattern] = count / pattern_count
        else:
            per_pattern_recall[pattern] = 0
    
    # Precision
    total_number_of_all_located_patterns = len(located_patterns_df)
    total_precision = total_number_of_properly_located_patterns / total_number_of_all_located_patterns if total_number_of_all_located_patterns > 0 else 0
    
    per_pattern_precision = {}
    for pattern, count in number_of_properly_located_patterns.items():
        pattern_predictions = located_patterns_df[located_patterns_df['Chart Pattern'] == pattern].shape[0]
        if pattern_predictions > 0:
            per_pattern_precision[pattern] = count / pattern_predictions
        else:
            per_pattern_precision[pattern] = 0
    
    # F1 Score
    per_pattern_f1 = {}
    for pattern in per_pattern_recall.keys():
        precision = per_pattern_precision.get(pattern, 0)
        recall = per_pattern_recall.get(pattern, 0)
        if precision + recall > 0:
            per_pattern_f1[pattern] = 2 * (precision * recall) / (precision + recall)
        else:
            per_pattern_f1[pattern] = 0
    
    all_precisions = list(per_pattern_precision.values())
    all_recalls = list(per_pattern_recall.values())
    avg_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0
    avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
    
    if avg_precision + avg_recall == 0:
        overall_f1 = 0
    else:
        overall_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    
    # MAE
    per_pattern_mae = {}
    for pattern, count in number_of_properly_located_patterns.items():
        if count > 0:
            per_pattern_mae[pattern] = mae_for_each_properly_detected_pattern.get(pattern, 0) / count
        else:
            per_pattern_mae[pattern] = 0
    
    total_mae_sum = sum(mae_for_each_properly_detected_pattern.values())
    total_proper_patterns = sum(number_of_properly_located_patterns.values())
    overall_mae = total_mae_sum / total_proper_patterns if total_proper_patterns > 0 else 0
    
    # IoU
    per_pattern_iou = {}
    for pattern, count in number_of_properly_located_patterns.items():
        if count > 0:
            per_pattern_iou[pattern] = iou_for_each_properly_detected_pattern.get(pattern, 0) / count
        else:
            per_pattern_iou[pattern] = 0
    
    total_iou_sum = sum(iou_for_each_properly_detected_pattern.values())
    overall_iou = total_iou_sum / total_proper_patterns if total_proper_patterns > 0 else 0
    
    # Print summary of metrics
    print(f"Overall Recall: {total_recall:.4f}")
    print(f"Overall Precision: {total_precision:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    print(f"Overall Mean Absolute Error: {overall_mae:.4f}")
    print(f"Overall Mean Intersection over Union: {overall_iou:.4f}")
    
    # Store all metrics in one place for easy access
    metrics_summary = {
        'total_recall': total_recall,
        'per_pattern_recall': per_pattern_recall,
        'total_precision': total_precision,
        'per_pattern_precision': per_pattern_precision,
        'overall_f1': overall_f1,
        'per_pattern_f1': per_pattern_f1,
        'overall_mae': overall_mae,
        'per_pattern_mae': per_pattern_mae,
        'overall_iou': overall_iou,
        'per_pattern_iou': per_pattern_iou
    }
    
    return metrics_summary

# Updated evaluate_all_models function that only creates the comprehensive plot
def evaluate_all_models(model_eval_results_dict, pattern_row_count, test_patterns, located_patterns_and_other_info_updated_dict):
   Evaluate all models and return metrics summary with comprehensive plot only
    all_models_metrics = {}
    
    for model_name in model_eval_results_dict.keys():
        all_models_metrics[model_name] = evaluate_model(
            model_name,
            model_eval_results_dict,
            pattern_row_count,
            test_patterns,
            located_patterns_and_other_info_updated_dict
        )
    
    # Only create the comprehensive visualization
    if len(model_eval_results_dict) > 0:
        print("\n--- Comprehensive Model Comparison ---")
        # figure = create_comprehensive_model_comparison(all_models_metrics)
    
    return all_models_metrics, None  # Return None instead of figure
"""
###########################################################################################################