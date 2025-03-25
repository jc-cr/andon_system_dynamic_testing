#!/usr/bin/env python3
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# Define the base directory for your data
BASE_DIR = "logs/dynamic_test"
BASE_DIR = os.path.expanduser(BASE_DIR)  # Expand the ~ to the actual home directory

def load_and_analyze_directory(dir_path):
    """Load and analyze a single experiment directory using config.json"""
    dir_name = os.path.basename(dir_path)
    print(f"Analyzing {dir_name}...")
    
    try:
        # Load config file
        config_path = os.path.join(dir_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"No config file found in {dir_name}")
    
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        angle = config.get('angle', 0)
        direction = config.get('direction', 'unknown')
        
        # Get time window from config if available
        start_time = config.get('start_time', None)  # in seconds
        end_time = config.get('end_time', None)  # in seconds
        
        # If no time window specified, raise error
        if start_time is None and end_time is None:
            raise ValueError("No time window specified in config.json")
        
        # Load data files
        rs_path = os.path.join(dir_path, "realsense_data.json")
        andon_path = os.path.join(dir_path, "andon_data.json")
        
        # Check if files exist
        if not os.path.exists(rs_path):
            raise ValueError(f"RealSense data file not found in {dir_name}")
        if not os.path.exists(andon_path):
            raise ValueError(f"Andon data file not found in {dir_name}")
        
        # Load JSON data
        with open(rs_path, 'r') as f:
            rs_data = json.load(f)
        with open(andon_path, 'r') as f:
            andon_data = json.load(f)
            
        rs_df = pd.DataFrame(rs_data)
        andon_df = pd.DataFrame(andon_data)
        
        # Ensure we have data
        if len(rs_df) == 0:
            raise ValueError(f"No RealSense data in {dir_name}")
        if len(andon_df) == 0:
            raise ValueError(f"No Andon data in {dir_name}")
        
        # For debug purposes, print data shapes before filtering
        print(f"  Initial data: RealSense {len(rs_df)} records, Andon {len(andon_df)} records")
        
        filtered_rs_df = rs_df.copy()
        filtered_andon_df = andon_df.copy()
        
        if start_time is not None or end_time is not None:
            # Use absolute timestamps, not relative ones
            global_min_time = min(rs_df['timestamp'].min(), andon_df['timestamp'].min())
            
            # Calculate absolute start/end times
            abs_start_time = global_min_time + start_time if start_time is not None else None
            abs_end_time = global_min_time + end_time if end_time is not None else None
            
            # Apply filtering using absolute timestamps
            if abs_start_time is not None:
                rs_df = rs_df[rs_df['timestamp'] >= abs_start_time]
                andon_df = andon_df[andon_df['timestamp'] >= abs_start_time]
                
            if abs_end_time is not None:
                rs_df = rs_df[rs_df['timestamp'] <= abs_end_time]
                andon_df = andon_df[andon_df['timestamp'] <= abs_end_time]
                
            print(f"  After time filtering: RealSense {len(filtered_rs_df)} records, Andon {len(filtered_andon_df)} records")
        
        # Ensure we still have data after filtering
        if len(filtered_rs_df) == 0:
            raise ValueError(f"No RealSense data after time filtering in {dir_name}")
        if len(filtered_andon_df) == 0:
            raise ValueError(f"No Andon data after time filtering in {dir_name}")
        
        # Merge datasets based on closest timestamp
        # Set a maximum allowed time difference (100ms)
        MAX_TIME_DIFF = 0.1
        
        # Replace the entire merged data generation with this:
        merged_data = []
        for _, rs_row in rs_df.iterrows():
            rs_time = rs_row['timestamp']
            
            # Use a more reliable approach for finding closest timestamp
            time_diffs = (andon_df['timestamp'] - rs_time).abs()
            
            if len(time_diffs) > 0:
                # Get the index of the minimum time difference
                closest_idx = time_diffs.idxmin()
                closest_diff = time_diffs.loc[closest_idx]
                
                # Only check time difference threshold, NOT detection status
                if closest_diff <= MAX_TIME_DIFF:
                    andon_row = andon_df.loc[closest_idx]
                    
                    merged_data.append({
                        'timestamp': rs_time,
                        'rs_detected': rs_row['detected'],
                        'andon_detected': andon_row['detected'],
                        'rs_depth': rs_row.get('depth', np.nan),
                        'andon_depth': andon_row.get('depth', np.nan),
                        'rs_confidence': rs_row.get('confidence', 0),
                        'andon_confidence': andon_row.get('confidence', 0),
                        'time_diff': closest_diff
                    })
                
        merged_df = pd.DataFrame(merged_data)
        if len(merged_df) == 0:
            print(f"  No matching data points in {dir_name}")
            return None
        
        # Calculate detection metrics
        # Filter for frames where RealSense detected (our ground truth)
        rs_detected_df = merged_df[merged_df['rs_detected']]
        if len(rs_detected_df) == 0:
            print(f"  No RealSense detections in {dir_name}")
            return None
        
        rs_detected_count = len(rs_detected_df)
        both_detected_count = sum(rs_detected_df['andon_detected'])
        detection_rate = (both_detected_count / rs_detected_count) * 100
        
        # Debug print detection metrics
        print(f"  Detection metrics: {both_detected_count}/{rs_detected_count} frames ({detection_rate:.1f}%)")
        
        # Get frames where both systems detected for depth analysis
        both_detected_df = rs_detected_df[rs_detected_df['andon_detected']]
        
        # Initialize depth metrics
        rmse = np.nan
        mae = np.nan
        relative_error = np.nan
        min_depth = np.nan
        max_depth = np.nan
        avg_depth = np.nan
        valid_depth_count = 0
        
        # If we have frames where both systems detected, calculate depth metrics
        if len(both_detected_df) > 0:
            # Debug - check the depth values before filtering
            print(f"  Both detected: {len(both_detected_df)} frames")
            
            # Get subset with valid depth values
            valid_depth_df = both_detected_df[
                (both_detected_df['rs_depth'] > 0) & 
                (both_detected_df['andon_depth'] > 0) &
                (both_detected_df['rs_depth'] <= 4000) & 
                (both_detected_df['andon_depth'] <= 4000)
            ]
            
            valid_depth_count = len(valid_depth_df)
            print(f"  Valid depth measurements: {valid_depth_count} frames")
            
            if valid_depth_count > 0:
                # Calculate depth errors
                errors = valid_depth_df['andon_depth'] - valid_depth_df['rs_depth']
                abs_errors = abs(errors)
                
                rmse = np.sqrt(np.mean(errors**2))
                mae = np.mean(abs_errors)
                avg_depth = np.mean(valid_depth_df['rs_depth'])
                relative_error = (mae / avg_depth) * 100 if avg_depth > 0 else np.nan
                min_depth = valid_depth_df['rs_depth'].min()
                max_depth = valid_depth_df['rs_depth'].max()
                
                print(f"  Depth metrics: RMSE = {rmse:.2f}mm, MAE = {mae:.2f}mm, Rel Error = {relative_error:.1f}%")
        
        # Generate plots
        output_dir = os.path.join(BASE_DIR, "results")
        os.makedirs(output_dir, exist_ok=True)
        
        test_label = f"{angle}° {direction}"
        
        # Always plot detection data
        plot_detection_comparison(merged_df, test_label, output_dir, angle, direction)
        
        # Plot depth data if we have valid measurements
        if valid_depth_count > 0:
            plot_depth_comparison(both_detected_df, valid_depth_df, test_label, output_dir, angle, direction)
        else:
            create_placeholder_depth_plot(test_label, output_dir, angle, direction, detection_rate)
        
        # Return results including all key metrics
        return {
            'dir_name': dir_name,
            'angle': angle,
            'direction': direction,
            'rmse': rmse,
            'mae': mae,
            'relative_error': relative_error,
            'detection_rate': detection_rate,
            'rs_detected_count': rs_detected_count,
            'both_detected_count': both_detected_count,
            'valid_depth_count': valid_depth_count,
            'min_depth': min_depth,
            'max_depth': max_depth,
            'avg_depth': avg_depth
        }
        
    except Exception as e:
        print(f"  Error analyzing {dir_name}: {e}")
        traceback.print_exc()
        return None

def plot_depth_comparison(both_detected_df, valid_depth_df, test_label, output_dir, angle, direction):
    """Create a depth comparison plot showing RealSense vs Andon measurements"""
    plt.figure(figsize=(10, 6))
    
    # Normalize time to start at 0
    min_time = both_detected_df['timestamp'].min()
    both_detected_df = both_detected_df.copy()
    both_detected_df['relative_time'] = both_detected_df['timestamp'] - min_time
    
    valid_depth_df = valid_depth_df.copy()
    valid_depth_df['relative_time'] = valid_depth_df['timestamp'] - min_time
    
    # Sort by time
    both_detected_df = both_detected_df.sort_values('relative_time')
    valid_depth_df = valid_depth_df.sort_values('relative_time')
    
    # Plot all detected points (even if depths are invalid)
    plt.plot(both_detected_df['relative_time'], both_detected_df['rs_depth']/1000, 'bo', alpha=0.3, label='RealSense (all)')
    plt.plot(both_detected_df['relative_time'], both_detected_df['andon_depth']/1000, 'ro', alpha=0.3, label='Andon (all)')
    
    # Plot valid depth measurements with lines connecting them
    if len(valid_depth_df) > 0:
        plt.plot(valid_depth_df['relative_time'], valid_depth_df['rs_depth']/1000, 'b-', linewidth=2, label='RealSense (valid)')
        plt.plot(valid_depth_df['relative_time'], valid_depth_df['andon_depth']/1000, 'r-', linewidth=2, label='Andon (valid)')
        
        # Plot error for valid measurements
        plt.plot(valid_depth_df['relative_time'], 
                (valid_depth_df['andon_depth'] - valid_depth_df['rs_depth'])/1000, 
                'g--', label='Error', alpha=0.7)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Depth (meters)')
    plt.title(f'Depth Comparison - {test_label}')
    plt.legend()
    plt.grid(True)
    
    # Calculate statistics for annotation (only using valid measurements)
    if len(valid_depth_df) > 0:
        errors = valid_depth_df['andon_depth'] - valid_depth_df['rs_depth']
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(abs(errors))
        avg_depth = np.mean(valid_depth_df['rs_depth'])
        rel_error = (mae / avg_depth) * 100
        
        # Add statistics text
        plt.figtext(0.15, 0.02, 
                    f"RMSE: {rmse/1000:.3f}m, MAE: {mae/1000:.3f}m, Rel Error: {rel_error:.1f}%\n" +
                    f"Valid depth points: {len(valid_depth_df)}/{len(both_detected_df)}", 
                    fontsize=10)
    else:
        plt.figtext(0.15, 0.02, "No valid depth measurements for metrics", fontsize=10)
    
    # Save figure with a clean filename
    filename = f"depth_{angle}_{direction}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def create_placeholder_depth_plot(test_label, output_dir, angle, direction, detection_rate):
    """Create a placeholder depth plot when no valid depth data exists"""
    plt.figure(figsize=(10, 6))
    
    if detection_rate == 0:
        message = "No Valid Depth Data\n(Andon failed to detect the person)"
    else:
        message = "No Valid Depth Data\n(Both systems detected but depth values were invalid)"
    
    # Create a simple text annotation
    plt.text(0.5, 0.5, message,
             horizontalalignment='center',
             verticalalignment='center',
             transform=plt.gca().transAxes,
             fontsize=14)
    
    plt.title(f'Depth Comparison - {test_label}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Depth (meters)')
    
    # Add annotation with detection rate
    plt.figtext(0.15, 0.02, f"Detection Rate: {detection_rate:.1f}%", fontsize=10)
    
    # Remove axis ticks for cleaner look
    plt.xticks([])
    plt.yticks([])
    
    # Save figure with a clean filename
    filename = f"depth_{angle}_{direction}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def plot_detection_comparison(merged_df, test_label, output_dir, angle, direction):
    """Create a plot showing detection performance over time"""
    if len(merged_df) == 0:
        print(f"  No detection data to plot for {test_label}")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Normalize time to start at 0
    min_time = merged_df['timestamp'].min()
    merged_df = merged_df.copy()
    merged_df['relative_time'] = merged_df['timestamp'] - min_time
    
    # Sort by time
    merged_df = merged_df.sort_values('relative_time')
    
    # Create binary detection indicators (0 or 1)
    rs_detect = merged_df['rs_detected'].astype(int)
    andon_detect = merged_df['andon_detected'].astype(int)
    
    # Plot detections (offset Andon slightly for visibility)
    plt.plot(merged_df['relative_time'], rs_detect, 'b-', label='RealSense Detection', linewidth=2)
    plt.plot(merged_df['relative_time'], andon_detect - 0.05, 'r-', label='Andon Detection', linewidth=2)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Detection (1 = Yes, 0 = No)')
    plt.title(f'Detection Comparison - {test_label}')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.1, 1.1)  # Give some margin for visibility
    
    # Calculate detection rate for annotation
    rs_detected_count = sum(merged_df['rs_detected'])
    both_detected_count = sum(merged_df['rs_detected'] & merged_df['andon_detected'])
    detection_rate = (both_detected_count / rs_detected_count) * 100 if rs_detected_count > 0 else 0
    
    # Add statistics text
    plt.figtext(0.15, 0.02, 
                f"Detection Rate: {detection_rate:.1f}% ({both_detected_count}/{rs_detected_count} frames)", 
                fontsize=10)
    
    # Check if Andon detection improves over time
    andon_first5 = merged_df.head(5)['andon_detected'].mean()
    andon_last5 = merged_df.tail(5)['andon_detected'].mean()
    
    if andon_first5 < 0.5 and andon_last5 > 0.5:
        plt.figtext(0.5, 0.95, "Note: Andon detection improves over time", 
                   fontsize=10, ha='center')
    
    # Save figure with a clean filename
    filename = f"detection_{angle}_{direction}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

def create_summary_plots(results):
    """Create summary plots comparing metrics across all test configurations"""
    if not results:
        print("No valid results to plot")
        return
    
    output_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame from results
    df = pd.DataFrame(results)
    
    # Sort by angle and then direction for consistent presentation
    df = df.sort_values(['angle', 'direction'])
    
    # Create labels for plots
    df['test_label'] = df.apply(lambda row: f"{row['angle']}° {row['direction']}", axis=1)
    
    # 1. Detection rate comparison (THE CRITICAL METRIC)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df['test_label'], df['detection_rate'], color='r')
    
    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height:.1f}%',
                 ha='center', va='bottom', rotation=0)
    
    plt.xlabel('Test Configuration')
    plt.ylabel('Detection Rate (%)')
    plt.title('Detection Rate by Test Configuration (% of RealSense detections that Andon also detected)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.ylim(0, 105)  # Leave room for percentage labels
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detection_rate_summary.png"))
    plt.close()
    
    # 2. RMSE and MAE comparison (only for tests with valid depth data)
    depth_df = df[~df['rmse'].isna()]
    
    if len(depth_df) > 0:
        plt.figure(figsize=(12, 6))
        x = np.arange(len(depth_df))
        width = 0.35
        
        plt.bar(x - width/2, depth_df['rmse']/1000, width, label='RMSE', color='blue', alpha=0.7)
        plt.bar(x + width/2, depth_df['mae']/1000, width, label='MAE', color='green', alpha=0.7)
        
        plt.xlabel('Test Configuration')
        plt.ylabel('Error (meters)')
        plt.title('Depth Measurement Error by Test Configuration (when Andon detected)')
        plt.xticks(x, depth_df['test_label'], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "depth_error_summary.png"))
        plt.close()
    
    # 3. Relative error comparison (only for tests with valid depth data)
    if len(depth_df) > 0:
        plt.figure(figsize=(12, 6))
        bars = plt.bar(depth_df['test_label'], depth_df['relative_error'], color='purple', alpha=0.7)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom', rotation=0)
        
        plt.xlabel('Test Configuration')
        plt.ylabel('Relative Error (%)')
        plt.title('Relative Depth Error by Test Configuration (when Andon detected)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "relative_error_summary.png"))
        plt.close()
    
    # 4. Sample count comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.3
    
    plt.bar(x - width, df['rs_detected_count'], width, label='RealSense Detections', color='blue')
    plt.bar(x, df['both_detected_count'], width, label='Both Detected', color='green')
    plt.bar(x + width, df['valid_depth_count'], width, label='Valid Depth Points', color='orange')
    
    # Add data labels
    for i, value in enumerate(df['rs_detected_count']):
        plt.text(i - width, value + 5, str(value), ha='center')
    
    for i, value in enumerate(df['both_detected_count']):
        plt.text(i, value + 5, str(value), ha='center')
        
    for i, value in enumerate(df['valid_depth_count']):
        plt.text(i + width, value + 5, str(value), ha='center')
    
    plt.xlabel('Test Configuration')
    plt.ylabel('Count')
    plt.title('Detection and Valid Data Counts by Test Configuration')
    plt.xticks(x, df['test_label'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "data_count_summary.png"))
    plt.close()
    
    # 5. Angle comparison plots grouped by direction
    # For detection rate
    plt.figure(figsize=(12, 6))
    
    # Get forward and backward data
    forward_df = df[df['direction'].str.contains('fwd|forward', case=False)]
    backward_df = df[df['direction'].str.contains('bck|back|backward', case=False)]
    
    if len(forward_df) > 0:
        plt.plot(forward_df['angle'], forward_df['detection_rate'], 'bo-', linewidth=2, markersize=8, label='Forward Detection Rate')
    if len(backward_df) > 0:
        plt.plot(backward_df['angle'], backward_df['detection_rate'], 'ro-', linewidth=2, markersize=8, label='Backward Detection Rate')
    
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Detection Rate (%)')
    plt.title('Detection Rate by Angle and Direction')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "angle_detection_comparison.png"))
    plt.close()
    
    # For RMSE if we have valid depth data
    plt.figure(figsize=(12, 6))
    
    # Get forward and backward data with valid RMSE
    forward_depth_df = forward_df[~forward_df['rmse'].isna()]
    backward_depth_df = backward_df[~backward_df['rmse'].isna()]
    
    if len(forward_depth_df) > 0:
        plt.plot(forward_depth_df['angle'], forward_depth_df['rmse']/1000, 'bo-', linewidth=2, markersize=8, label='Forward RMSE')
    if len(backward_depth_df) > 0:
        plt.plot(backward_depth_df['angle'], backward_depth_df['rmse']/1000, 'ro-', linewidth=2, markersize=8, label='Backward RMSE')
    
    plt.xlabel('Angle (degrees)')
    plt.ylabel('RMSE (meters)')
    plt.title('RMSE by Angle and Direction (when Andon detected)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "angle_rmse_comparison.png"))
    plt.close()
    
    # Save detailed summary to CSV
    summary_df = df[['angle', 'direction', 'rmse', 'mae', 'relative_error', 
                    'detection_rate', 'rs_detected_count', 'both_detected_count', 
                    'valid_depth_count', 'min_depth', 'max_depth', 'avg_depth']]
    summary_df.to_csv(os.path.join(output_dir, "summary_results.csv"), index=False)
    print(f"Summary saved to {os.path.join(output_dir, 'summary_results.csv')}")
    
    # Print summary table with formatted values
    print("\nSummary of Results:")
    summary_table = summary_df.copy()
    
    # Format values for display
    summary_table['rmse'] = summary_table['rmse'].apply(lambda x: f"{x:.1f} mm" if not np.isnan(x) else "N/A")
    summary_table['mae'] = summary_table['mae'].apply(lambda x: f"{x:.1f} mm" if not np.isnan(x) else "N/A")
    summary_table['relative_error'] = summary_table['relative_error'].apply(lambda x: f"{x:.1f}%" if not np.isnan(x) else "N/A")
    summary_table['detection_rate'] = summary_table['detection_rate'].apply(lambda x: f"{x:.1f}%" if not np.isnan(x) else "N/A")
    summary_table['min_depth'] = summary_table['min_depth'].apply(lambda x: f"{x/1000:.2f} m" if not np.isnan(x) else "N/A")
    summary_table['max_depth'] = summary_table['max_depth'].apply(lambda x: f"{x/1000:.2f} m" if not np.isnan(x) else "N/A")
    summary_table['avg_depth'] = summary_table['avg_depth'].apply(lambda x: f"{x/1000:.2f} m" if not np.isnan(x) else "N/A")
    
    print(summary_table.to_string())
    
    # Additional summary: Detection rate by angle
    print("\nDetection Rate by Angle:")
    angle_summary = df.groupby('angle')['detection_rate'].mean().reset_index()
    angle_summary['detection_rate'] = angle_summary['detection_rate'].apply(lambda x: f"{x:.1f}%")
    print(angle_summary.to_string(index=False))
    
    # Direction summary
    print("\nDetection Rate by Direction:")
    dir_summary = df.groupby('direction')['detection_rate'].mean().reset_index()
    dir_summary['detection_rate'] = dir_summary['detection_rate'].apply(lambda x: f"{x:.1f}%")
    print(dir_summary.to_string(index=False))

def main():
    print(f"Starting analysis of data in {BASE_DIR}")
    
    # Get all experiment directories (exclude the results directory)
    data_dirs = [d for d in glob.glob(os.path.join(BASE_DIR, "*")) 
                if os.path.isdir(d) and not os.path.basename(d) == "results"]
    
    if not data_dirs:
        print(f"No data directories found in {BASE_DIR}")
        return
    
    print(f"Found {len(data_dirs)} test directories to analyze")
    
    # Process each directory
    results = []
    for dir_path in data_dirs:
        result = load_and_analyze_directory(dir_path)
        if result is not None:
            results.append(result)
            dir_name = os.path.basename(dir_path)
            
            # Print summary of results
            detection_msg = f"Detection Rate = {result['detection_rate']:.1f}% ({result['both_detected_count']}/{result['rs_detected_count']})"
            
            if result['valid_depth_count'] > 0:
                depth_msg = f"RMSE = {result['rmse']:.2f}mm ({result['valid_depth_count']} valid pts)"
            else:
                depth_msg = "No valid depth data"
            
            print(f"  Processed {dir_name}: {detection_msg}, {depth_msg}")
    
    # Create summary plots and reports
    if results:
        create_summary_plots(results)
        print(f"Analysis complete. Results saved to {os.path.join(BASE_DIR, 'results')}")
    else:
        print("No valid results found. Check if directories contain the required data files.")

if __name__ == "__main__":
    main()