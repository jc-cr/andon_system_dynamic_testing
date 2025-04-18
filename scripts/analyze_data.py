import os
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager


plt.rcParams['figure.figsize'] = (12, 8)  # Larger default figure size
plt.rcParams["figure.autolayout"]  = True
plt.rcParams["figure.dpi"] = 300



# Set font with fallbacks to system fonts
plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.size'] = 14  # Base font size
plt.rcParams['axes.titlesize'] = 18  # Title font size
plt.rcParams['axes.labelsize'] = 16  # Axes labels font size
plt.rcParams['xtick.labelsize'] = 14  # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 14  # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 14  # Legend font size
plt.rcParams['figure.titlesize'] = 20  # Figure title size



def load_test_data(test_dir):
    """Load data from a test directory."""
    try:
        # Load RealSense data
        with open(os.path.join(test_dir, 'realsense_data.json'), 'r') as f:
            realsense_data = json.load(f)
        
        # Load Andon data
        with open(os.path.join(test_dir, 'andon_data.json'), 'r') as f:
            andon_data = json.load(f)
        
        # Load config
        with open(os.path.join(test_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Extract test name and angle from directory name
        dir_name = os.path.basename(test_dir)
        parts = dir_name.split('_')
        
        # Standard format is: {angle}_{direction}_{side}_{timestamp} or similar
        test_name = parts[0]
        
        # Handle potential numerical suffixes in test name
        if test_name.isdigit():
            # This is likely an angle
            if len(parts) > 1 and parts[1] in ['l', 'r']:
                # Include the left/right designation in the test name
                test_name = f"{test_name}_{parts[1]}"
        
        # If config uses old-style relative timestamps, upgrade it to absolute timestamps
        if ('start_time' in config or 'end_time' in config) and 'start_timestamp' not in config:
            print(f"  - Converting from relative to absolute timestamps in config")
            # Get the reference timestamp for conversion
            rs_first_time = realsense_data[0]['timestamp']
            rs_last_time = realsense_data[-1]['timestamp']
            
            # Create absolute timestamps
            if 'start_time' in config:
                config['start_timestamp'] = rs_first_time + config['start_time']
            else:
                config['start_timestamp'] = rs_first_time
                
            if 'end_time' in config:
                config['end_timestamp'] = rs_first_time + config['end_time']
            else:
                config['end_timestamp'] = rs_last_time
            
            # Save the updated config
            with open(os.path.join(test_dir, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"  - Updated config.json with absolute timestamps")
        
        return {
            'test_name': test_name,
            'realsense_data': realsense_data,
            'andon_data': andon_data,
            'config': config
        }
    except Exception as e:
        print(f"Error loading data from {test_dir}: {e}")
        return None

def align_timestamps(realsense_data, andon_data, config, max_time_diff=0.1):
    """
    Align RealSense and Andon data based on timestamps.
    Uses absolute timestamps from config.
    """
    # Check if data exists
    if not realsense_data or not andon_data:
        return []  # No data to align
    
    # Get the time window from config
    start_timestamp = config.get('start_timestamp')
    end_timestamp = config.get('end_timestamp')
    
    # If not specified, use first/last timestamps
    if start_timestamp is None:
        start_timestamp = realsense_data[0]['timestamp']
    
    if end_timestamp is None:
        end_timestamp = realsense_data[-1]['timestamp']
    
    # Print time window info with 2 decimal precision
    rs_first_time = realsense_data[0]['timestamp']
    print(f"  - Using time window: {start_timestamp:.2f} to {end_timestamp:.2f}")
    print(f"  - Relative to start: {(start_timestamp - rs_first_time):.2f}s to {(end_timestamp - rs_first_time):.2f}s")
    
    # Filter data within the specified time window
    filtered_realsense = [
        record for record in realsense_data 
        if start_timestamp <= record['timestamp'] <= end_timestamp
    ]
    
    filtered_andon = [
        record for record in andon_data 
        if start_timestamp <= record['timestamp'] <= end_timestamp
    ]
    
    # Count detections in window for diagnostics
    rs_detected = sum(1 for r in filtered_realsense if r['detected'])
    andon_detected = sum(1 for a in filtered_andon if a['detected'])
    
    print(f"  - Records in time window: {len(filtered_realsense)} RealSense, {len(filtered_andon)} Andon")
    print(f"  - Detections in time window: {rs_detected} RealSense, {andon_detected} Andon")
    
    # Now align the filtered data
    pairs = []
    realsense_idx = 0
    andon_idx = 0
    
    while realsense_idx < len(filtered_realsense) and andon_idx < len(filtered_andon):
        rs_time = filtered_realsense[realsense_idx]['timestamp']
        andon_time = filtered_andon[andon_idx]['timestamp']
        
        time_diff = abs(rs_time - andon_time)
        
        if time_diff <= max_time_diff:
            # Timestamps are close enough, consider them a pair
            pairs.append((filtered_realsense[realsense_idx], filtered_andon[andon_idx]))
            realsense_idx += 1
            andon_idx += 1
        elif rs_time < andon_time:
            realsense_idx += 1
        else:
            andon_idx += 1
    
    # Print alignment summary
    print(f"  - Aligned pairs: {len(pairs)}")
    both_detected = sum(1 for rs, andon in pairs if rs['detected'] and andon['detected'])
    print(f"  - Pairs where both systems detected: {both_detected}")
    
    return pairs

def calculate_detection_metrics(aligned_pairs):
    """Calculate detection performance metrics."""
    total_pairs = len(aligned_pairs)
    
    # Counters for detection scenarios
    true_positives = 0  # Both detected
    false_positives = 0  # Andon detected, RealSense didn't
    false_negatives = 0  # RealSense detected, Andon didn't
    true_negatives = 0   # Neither detected
    
    # Lists for latency and confidence analysis
    detection_latencies = []
    true_positive_confidences = []
    false_positive_confidences = []
    
    for rs_record, andon_record in aligned_pairs:
        rs_detected = rs_record['detected']
        andon_detected = andon_record['detected']
        
        if rs_detected and andon_detected:
            true_positives += 1
            # Calculate latency (time difference)
            true_positive_confidences.append(andon_record['confidence'])
        elif not rs_detected and andon_detected:
            false_positives += 1
            false_positive_confidences.append(andon_record['confidence'])
        elif rs_detected and not andon_detected:
            false_negatives += 1
        else:
            true_negatives += 1
    
    # Calculate detection metrics
    detection_rate = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    false_positive_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    false_negative_rate = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Average confidence scores
    avg_true_positive_confidence = np.mean(true_positive_confidences) if true_positive_confidences else 0
    avg_false_positive_confidence = np.mean(false_positive_confidences) if false_positive_confidences else 0
    
    return {
        'total_pairs': total_pairs,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'detection_rate': round(detection_rate, 2),
        'false_positive_rate': round(false_positive_rate, 2),
        'false_negative_rate': round(false_negative_rate, 2),
        'avg_true_positive_confidence': round(avg_true_positive_confidence, 2),
        'avg_false_positive_confidence': round(avg_false_positive_confidence, 2)
    }

def calculate_depth_metrics(aligned_pairs):
    """Calculate depth estimation performance metrics."""
    # Filter to include only pairs where both systems detected a person
    detected_pairs = [(rs, andon) for rs, andon in aligned_pairs if rs['detected'] and andon['detected']]
    
    if not detected_pairs:
        print(f"  - No pairs where both systems detected a person")
        return {
            'count': 0,
            'mae': 0,
            'rmse': 0,
            'mean_relative_error': 0,
            'bias': 0,
            'correlation': 0,
            'p_value': 0
        }
    
    print(f"  - Found {len(detected_pairs)} pairs where both systems detected a person")
    
    # Get direction from test type (first pair with detection)
    is_forward = False
    for rs, andon in aligned_pairs:
        if rs['detected']:
            # Look at depth value of first detection to estimate direction
            if rs['depth'] > 4000:
                is_forward = True
                print("  - High initial depth values detected, likely a forward test")
            break
    
    # Set depth validation range based on test type
    max_valid_depth = 5000 if is_forward else 4000
    print(f"  - Using depth validation range: 0 to {max_valid_depth} mm")
    
    # Extract all depth values
    all_realsense_depths = [pair[0]['depth'] for pair in detected_pairs]
    all_andon_depths = [pair[1]['depth'] for pair in detected_pairs]
    
    # Print depth range info with 2 decimal precision
    rs_min, rs_max = min(all_realsense_depths), max(all_realsense_depths)
    andon_min, andon_max = min(all_andon_depths), max(all_andon_depths)
    print(f"  - RealSense depth range: {rs_min:.2f} to {rs_max:.2f} mm")
    print(f"  - Andon depth range: {andon_min:.2f} to {andon_max:.2f} mm")
    
    # Count how many values are outside valid range
    rs_invalid = sum(1 for d in all_realsense_depths if not (0 <= d <= max_valid_depth))
    andon_invalid = sum(1 for d in all_andon_depths if not (0 <= d <= max_valid_depth))
    print(f"  - Invalid RealSense depths (outside 0-{max_valid_depth}mm): {rs_invalid}/{len(all_realsense_depths)}")
    print(f"  - Invalid Andon depths (outside 0-{max_valid_depth}mm): {andon_invalid}/{len(all_andon_depths)}")
    
    # Filter out invalid depth values (< 0 or > max_valid_depth)
    valid_pairs = []
    for rs, andon in detected_pairs:
        rs_depth = rs['depth']
        andon_depth = andon['depth']
        
        if (0 <= rs_depth <= max_valid_depth) and (0 <= andon_depth <= max_valid_depth):
            valid_pairs.append((rs, andon))
    
    if not valid_pairs:
        print(f"  - WARNING: No valid depth pairs found (all depths outside 0-{max_valid_depth}mm range)")
        
        # Even with no valid pairs, include original depth data for visualization
        return {
            'count': 0,
            'mae': 0,
            'rmse': 0,
            'mean_relative_error': 0,
            'bias': 0,
            'correlation': 0,
            'p_value': 0,
            'realsense_depths': all_realsense_depths[:10],  # Include some for debugging
            'andon_depths': all_andon_depths[:10],          # Include some for debugging
            'errors': []
        }
    
    print(f"  - Using {len(valid_pairs)} valid depth pairs for metrics calculation")
    
    # Extract depth values from valid pairs
    realsense_depths = [pair[0]['depth'] for pair in valid_pairs]
    andon_depths = [pair[1]['depth'] for pair in valid_pairs]
    
    # Calculate depth metrics
    mae = mean_absolute_error(realsense_depths, andon_depths)
    rmse = np.sqrt(mean_squared_error(realsense_depths, andon_depths))
    
    # Calculate relative errors
    relative_errors = [abs(a - r) / r if r != 0 else float('inf') for r, a in zip(realsense_depths, andon_depths)]
    mean_relative_error = np.mean([e for e in relative_errors if not np.isinf(e)])
    
    # Calculate bias (signed error)
    errors = [a - r for r, a in zip(realsense_depths, andon_depths)]
    bias = np.mean(errors)
    
    # Calculate correlation between RealSense and Andon depths
    correlation, p_value = pearsonr(realsense_depths, andon_depths) if len(valid_pairs) > 1 else (0, 1)
    
    return {
        'count': len(valid_pairs),
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'mean_relative_error': round(mean_relative_error, 2),
        'bias': round(bias, 2),
        'correlation': round(correlation, 2),
        'p_value': round(p_value, 2),
        'realsense_depths': realsense_depths,
        'andon_depths': andon_depths,
        'errors': errors,
        'max_depth_threshold': max_valid_depth
    }

def analyze_test_data(test_data):
    """Analyze a single test's data and compute metrics."""
    # Print timestamp ranges with 2 decimal precision
    rs_first_time = test_data['realsense_data'][0]['timestamp']
    rs_last_time = test_data['realsense_data'][-1]['timestamp']
    andon_first_time = test_data['andon_data'][0]['timestamp']
    andon_last_time = test_data['andon_data'][-1]['timestamp']
    
    print(f"  - RealSense data range: {rs_first_time:.2f} to {rs_last_time:.2f} ({(rs_last_time - rs_first_time):.2f}s)")
    print(f"  - Andon data range: {andon_first_time:.2f} to {andon_last_time:.2f} ({(andon_last_time - andon_first_time):.2f}s)")
    
    # Align timestamps between RealSense and Andon data using the config time window
    aligned_pairs = align_timestamps(
        test_data['realsense_data'], 
        test_data['andon_data'],
        test_data['config']
    )
    
    # Calculate detection metrics
    detection_metrics = calculate_detection_metrics(aligned_pairs)
    
    # Calculate depth metrics
    depth_metrics = calculate_depth_metrics(aligned_pairs)
    
    return {
        'test_name': test_data['test_name'],
        'config': test_data['config'],
        'detection_metrics': detection_metrics,
        'depth_metrics': depth_metrics
    }

def analyze_detection_vs_distance(all_results):
    """Analyze how detection probability varies with distance."""
    distance_bins = []
    detection_rates = []
    
    # Create distance bins (in meters)
    bin_edges = np.arange(0, 5, 0.25)  # Bins from 0 to 5 meters in 0.25m increments
    
    # Collect all aligned pairs from all tests
    all_pairs = []
    for result in all_results:
        test_data = result['original_data']
        pairs = align_timestamps(
            test_data['realsense_data'], 
            test_data['andon_data'],
            test_data['config']
        )
        all_pairs.extend(pairs)
    
    # Group by distance and calculate detection rate
    for i in range(len(bin_edges)-1):
        min_dist = bin_edges[i] * 1000  # Convert to mm
        max_dist = bin_edges[i+1] * 1000
        
        # Filter pairs in this distance range (and ensure depth is valid: 0-4000mm)
        pairs_in_range = [p for p in all_pairs if p[0]['detected'] and 
                          min_dist <= p[0]['depth'] < max_dist and
                          0 <= p[0]['depth'] <= 4000]
        
        if pairs_in_range:
            detected_by_andon = [p[1]['detected'] for p in pairs_in_range].count(True)
            detection_rate = detected_by_andon / len(pairs_in_range)
            
            bin_center = (min_dist + max_dist) / 2
            distance_bins.append(bin_center / 1000)  # Convert back to meters for display
            detection_rates.append(round(detection_rate, 2))
    
    return {
        'distance_bins': distance_bins,
        'detection_rates': detection_rates
    }

def analyze_by_config_params(all_results):
    """Analyze how performance varies with test configuration parameters."""
    angles = sorted(set(result['config'].get('angle', 0) for result in all_results))
    directions = sorted(set(result['config'].get('direction', '') for result in all_results))
    
    # Analysis by angle
    angle_analysis = {}
    for angle in angles:
        angle_results = [r for r in all_results if r['config'].get('angle', 0) == angle]
        if angle_results:
            detection_rates = [r['detection_metrics']['detection_rate'] for r in angle_results]
            maes = [r['depth_metrics']['mae'] for r in angle_results]
            angle_analysis[angle] = {
                'count': len(angle_results),
                'avg_detection_rate': round(np.mean(detection_rates), 2),
                'avg_mae': round(np.mean([mae for mae in maes if mae > 0]), 2)  # Skip zero MAEs
            }
    
    # Analysis by direction
    direction_analysis = {}
    for direction in directions:
        direction_results = [r for r in all_results if r['config'].get('direction', '') == direction]
        if direction_results:
            detection_rates = [r['detection_metrics']['detection_rate'] for r in direction_results]
            maes = [r['depth_metrics']['mae'] for r in direction_results]
            direction_analysis[direction] = {
                'count': len(direction_results),
                'avg_detection_rate': round(np.mean(detection_rates), 2),
                'avg_mae': round(np.mean([mae for mae in maes if mae > 0]), 2)  # Skip zero MAEs
            }
    
    return {
        'angle_analysis': angle_analysis,
        'direction_analysis': direction_analysis
    }


def add_direction_comparison_visualizations(all_results, output_dir):
    """Generate visualizations comparing forward vs backward approaches."""
    # Separate results by direction
    forward_results = [r for r in all_results if r['config'].get('direction', '').lower() in ['forward', 'fwd']]
    backward_results = [r for r in all_results if r['config'].get('direction', '').lower() in ['backward', 'back', 'bck']]
    
    # 1. Detection rate by angle comparison
    if forward_results and backward_results:
        # Get all unique angles
        all_angles = sorted(set([r['config'].get('angle', 0) for r in all_results]))
        
        # Prepare data
        fwd_rates = []
        back_rates = []
        for angle in all_angles:
            # Get forward results for this angle
            angle_fwd = [r for r in forward_results if r['config'].get('angle', 0) == angle]
            if angle_fwd:
                fwd_rates.append(round(np.mean([r['detection_metrics']['detection_rate'] for r in angle_fwd]), 2))
            else:
                fwd_rates.append(0)
            
            # Get backward results for this angle
            angle_back = [r for r in backward_results if r['config'].get('angle', 0) == angle]
            if angle_back:
                back_rates.append(round(np.mean([r['detection_metrics']['detection_rate'] for r in angle_back]), 2))
            else:
                back_rates.append(0)
        
        # Create the comparison bar chart
        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        index = np.arange(len(all_angles))
        
        plt.bar(index - bar_width/2, fwd_rates, bar_width, label='Forward', color='#ff7f0e', alpha=0.8)
        plt.bar(index + bar_width/2, back_rates, bar_width, label='Backward', color='#1f77b4', alpha=0.8)
        
        # Add value labels on top of bars
        for i, v in enumerate(fwd_rates):
            if v > 0:
                plt.text(i - bar_width/2, v + 0.03, f"{v:.2f}", ha='center', va='bottom')
        
        for i, v in enumerate(back_rates):
            if v > 0:
                plt.text(i + bar_width/2, v + 0.03, f"{v:.2f}", ha='center', va='bottom')
        
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Detection Rate')
        plt.title('Detection Rate Comparison: Forward vs. Backward')
        plt.xticks(index, all_angles)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_rate_direction_comparison.png'))
        plt.close()
        
        # 2. Create a similar comparison for depth MAE
        fwd_mae = []
        back_mae = []
        for angle in all_angles:
            # Get forward results for this angle
            angle_fwd = [r for r in forward_results if r['config'].get('angle', 0) == angle]
            if angle_fwd:
                valid_maes = [r['depth_metrics']['mae'] for r in angle_fwd if r['depth_metrics']['mae'] > 0]
                fwd_mae.append(round(np.mean(valid_maes), 2) if valid_maes else 0)
            else:
                fwd_mae.append(0)
            
            # Get backward results for this angle
            angle_back = [r for r in backward_results if r['config'].get('angle', 0) == angle]
            if angle_back:
                valid_maes = [r['depth_metrics']['mae'] for r in angle_back if r['depth_metrics']['mae'] > 0]
                back_mae.append(round(np.mean(valid_maes), 2) if valid_maes else 0)
            else:
                back_mae.append(0)
        
        # Create the comparison bar chart
        plt.figure(figsize=(12, 6))
        
        plt.bar(index - bar_width/2, fwd_mae, bar_width, label='Forward', color='#ff7f0e', alpha=0.8)
        plt.bar(index + bar_width/2, back_mae, bar_width, label='Backward', color='#1f77b4', alpha=0.8)
        
        # Add value labels on top of bars
        for i, v in enumerate(fwd_mae):
            if v > 0:
                plt.text(i - bar_width/2, v + 1, f"{v:.2f}", ha='center', va='bottom')
        
        for i, v in enumerate(back_mae):
            if v > 0:
                plt.text(i + bar_width/2, v + 1, f"{v:.2f}", ha='center', va='bottom')
        
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Mean Absolute Error (mm)')
        plt.title('Depth Error Comparison: Forward vs. Backward')
        plt.xticks(index, all_angles)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'depth_mae_direction_comparison.png'))
        plt.close()
        
        # 3. Compare error distributions with box plots
        forward_errors = []
        backward_errors = []
        forward_angles = []
        backward_angles = []
        
        # Collect all error values by direction
        for result in all_results:
            if 'errors' in result['depth_metrics'] and result['depth_metrics']['errors']:
                direction = result['config'].get('direction', '').lower()
                angle = result['config'].get('angle', 0)
                
                if direction in ['forward', 'fwd']:
                    forward_errors.extend(result['depth_metrics']['errors'])
                    forward_angles.extend([angle] * len(result['depth_metrics']['errors']))
                elif direction in ['backward', 'back', 'bck']:
                    backward_errors.extend(result['depth_metrics']['errors'])
                    backward_angles.extend([angle] * len(result['depth_metrics']['errors']))
        
        if forward_errors and backward_errors:
            # Create boxplot comparison
            plt.figure(figsize=(10, 6))
            
            # Prepare data for boxplot
            data = [
                [error for error, angle in zip(forward_errors, forward_angles) if abs(error) < 2000],
                [error for error, angle in zip(backward_errors, backward_angles) if abs(error) < 2000]
            ]
            
            # Create the boxplot
            box = plt.boxplot(data, patch_artist=True, labels=['Forward', 'Backward'], showfliers=False)
            
            # Set colors
            colors = ['#ff7f0e', '#1f77b4']
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add stats
            for i, d in enumerate(data):
                if d:
                    median = np.median(d)
                    mean = np.mean(d)
                    plt.text(i+1, np.max(d) + 50, f"Mean: {mean:.2f}", ha='center')
                    plt.text(i+1, np.max(d) + 150, f"Median: {median:.2f}", ha='center')
            
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
            plt.ylabel('Depth Error (mm)')
            plt.title('Distribution of Depth Errors by Approach Direction')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'error_distribution_by_direction.png'))
            plt.close()
            
            # 4. Create a scatter plot of relative error vs. distance, colored by direction
            plt.figure(figsize=(12, 8))
            
            # Collect all depth data points with direction info
            all_rs_depths = []
            all_rel_errors = []
            all_directions = []
            
            for result in all_results:
                if ('realsense_depths' in result['depth_metrics'] and 
                    result['depth_metrics']['realsense_depths'] and 
                    'andon_depths' in result['depth_metrics']):
                    
                    rs_depths = result['depth_metrics']['realsense_depths']
                    andon_depths = result['depth_metrics']['andon_depths']
                    direction = result['config'].get('direction', '').lower()
                    
                    # Calculate relative errors
                    for rs, andon in zip(rs_depths, andon_depths):
                        if rs > 0:
                            rel_error = (andon - rs) / rs * 100
                            # Filter extreme values
                            if abs(rel_error) < 100:
                                all_rs_depths.append(rs)
                                all_rel_errors.append(round(rel_error, 2))
                                all_directions.append(direction)
            
            # Plot forward points
            fwd_indices = [i for i, d in enumerate(all_directions) if d in ['forward', 'fwd']]
            fwd_depths = [all_rs_depths[i] for i in fwd_indices]
            fwd_errors = [all_rel_errors[i] for i in fwd_indices]
            
            plt.scatter(fwd_depths, fwd_errors, alpha=0.5, label='Forward', color='#ff7f0e')
            
            # Plot backward points
            back_indices = [i for i, d in enumerate(all_directions) if d in ['backward', 'back', 'bck']]
            back_depths = [all_rs_depths[i] for i in back_indices]
            back_errors = [all_rel_errors[i] for i in back_indices]
            
            plt.scatter(back_depths, back_errors, alpha=0.5, label='Backward', color='#1f77b4')
            
            # Add trend lines for each direction
            try:
                from scipy.signal import savgol_filter
                
                # Forward trend line
                if len(fwd_depths) > 3:
                    # Sort points for trend line
                    sort_idx = np.argsort(fwd_depths)
                    sorted_depths = [fwd_depths[i] for i in sort_idx]
                    sorted_errors = [fwd_errors[i] for i in sort_idx]
                    
                    # Apply smoothing
                    window_length = min(51, len(sorted_depths) - 1)
                    if window_length % 2 == 0:  # Must be odd
                        window_length -= 1
                    
                    if window_length > 3:
                        fwd_smoothed = savgol_filter(sorted_errors, window_length, 3)
                        plt.plot(sorted_depths, fwd_smoothed, 'r-', linewidth=2, 
                                label='Forward Trend')
                
                # Backward trend line
                if len(back_depths) > 3:
                    # Sort points for trend line
                    sort_idx = np.argsort(back_depths)
                    sorted_depths = [back_depths[i] for i in sort_idx]
                    sorted_errors = [back_errors[i] for i in sort_idx]
                    
                    # Apply smoothing
                    window_length = min(51, len(sorted_depths) - 1)
                    if window_length % 2 == 0:  # Must be odd
                        window_length -= 1
                    
                    if window_length > 3:
                        back_smoothed = savgol_filter(sorted_errors, window_length, 3)
                        plt.plot(sorted_depths, back_smoothed, 'b-', linewidth=2, 
                                label='Backward Trend')
            except Exception as e:
                print(f"Could not add trend lines: {e}")
            
            plt.axhline(y=0, color='g', linestyle='--', linewidth=1.5)
            plt.xlabel('RealSense Depth (mm)')
            plt.ylabel('Relative Error (%)')
            plt.title('Relative Depth Error by Approach Direction')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'relative_error_by_direction.png'))
            plt.close()


def generate_visualizations(all_results, output_dir):
    """Generate and save improved visualization figures for depth and detection accuracy."""
    os.makedirs(output_dir, exist_ok=True)

    add_direction_comparison_visualizations(all_results, output_dir)
    
    # Keep the existing detection probability vs. distance plot
    detection_distance_analysis = analyze_detection_vs_distance(all_results)
    
    if detection_distance_analysis['distance_bins']:
        plt.figure(figsize=(10, 6))
        plt.plot(detection_distance_analysis['distance_bins'], 
                detection_distance_analysis['detection_rates'], 
                marker='o', linewidth=2)
        plt.xlabel('Distance (m)')
        plt.ylabel('Detection Rate')
        plt.title('Detection Probability vs. Distance')
        plt.grid(True)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_vs_distance.png'))
        plt.close()
    
    # Combine all depth data for more comprehensive analysis
    all_realsense_depths = []
    all_andon_depths = []
    all_errors = []
    all_test_angles = []
    all_test_directions = []
    
    for result in all_results:
        if ('realsense_depths' in result['depth_metrics'] and 
            result['depth_metrics']['realsense_depths'] and 
            'andon_depths' in result['depth_metrics']):
            
            rs_depths = result['depth_metrics']['realsense_depths']
            andon_depths = result['depth_metrics']['andon_depths']
            
            all_realsense_depths.extend(rs_depths)
            all_andon_depths.extend(andon_depths)
            all_errors.extend(result['depth_metrics']['errors'])
            
            # Extend the test information for each data point
            test_angle = result['config'].get('angle', 0)
            test_direction = result['config'].get('direction', '')
            all_test_angles.extend([test_angle] * len(rs_depths))
            all_test_directions.extend([test_direction] * len(rs_depths))
    
    if all_realsense_depths and all_andon_depths:
        # 1. NEW: Direct comparison scatter plot with regression line
        plt.figure(figsize=(10, 8))
        plt.scatter(all_realsense_depths, all_andon_depths, alpha=0.5)
        
        # Add perfect agreement line (45 degrees)
        max_depth = max(max(all_realsense_depths), max(all_andon_depths))
        min_depth = min(min(all_realsense_depths), min(all_andon_depths))
        plt.plot([min_depth, max_depth], [min_depth, max_depth], 'r--', label='Perfect Agreement')
        
        # Add regression line with 2 decimal precision
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_realsense_depths, all_andon_depths)
        plt.plot([min_depth, max_depth], [intercept + slope*min_depth, intercept + slope*max_depth], 
                 'g-', label=f'Regression Line (r²={r_value**2:.2f})')
        
        plt.xlabel('RealSense Depth (mm)')
        plt.ylabel('Andon Depth (mm)')
        plt.title('Depth Measurement Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'depth_comparison.png'))
        plt.close()
        
        # 2. NEW: Bland-Altman plot (shows agreement between two measurement methods)
        plt.figure(figsize=(10, 8))
        
        mean_depths = [(rs + andon)/2 for rs, andon in zip(all_realsense_depths, all_andon_depths)]
        diff_depths = [andon - rs for rs, andon in zip(all_realsense_depths, all_andon_depths)]
        
        # Calculate statistics for Bland-Altman with 2 decimal precision
        mean_diff = np.mean(diff_depths)
        std_diff = np.std(diff_depths)
        upper_limit = mean_diff + 1.96 * std_diff
        lower_limit = mean_diff - 1.96 * std_diff
        
        plt.scatter(mean_depths, diff_depths, alpha=0.5)
        plt.axhline(y=mean_diff, color='r', linestyle='-', label=f'Mean Difference: {mean_diff:.2f} mm')
        plt.axhline(y=upper_limit, color='g', linestyle='--', 
                   label=f'Upper 95% Limit: {upper_limit:.2f} mm')
        plt.axhline(y=lower_limit, color='g', linestyle='--', 
                   label=f'Lower 95% Limit: {lower_limit:.2f} mm')
        
        plt.xlabel('Mean of RealSense and Andon Measurements (mm)')
        plt.ylabel('Difference: Andon - RealSense (mm)')
        plt.title('Bland-Altman Plot of Depth Measurements')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bland_altman_plot.png'))
        plt.close()
        
        # 3. NEW: Histogram of depth errors
        plt.figure(figsize=(10, 6))
        
        # Use a reasonable bin count
        num_bins = min(30, int(len(all_errors) / 5))
        
        # Calculate percentiles for better bin ranges
        p1 = np.percentile(all_errors, 1)
        p99 = np.percentile(all_errors, 99)
        
        counts, bins, patches = plt.hist(all_errors, bins=num_bins, 
                                         range=(p1, p99), alpha=0.75, edgecolor='black')
        
        plt.axvline(x=0, color='r', linestyle='--', linewidth=1.5, 
                   label='Zero Error')
        plt.axvline(x=np.mean(all_errors), color='g', linestyle='-', linewidth=1.5, 
                   label=f'Mean Error: {np.mean(all_errors):.2f} mm')
        
        plt.xlabel('Depth Error: Andon - RealSense (mm)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Depth Measurement Errors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_histogram.png'))
        plt.close()
        
        # 4. NEW: Relative error vs distance
        plt.figure(figsize=(10, 6))
        
        # Calculate relative errors (as percentage) with 2 decimal precision
        relative_errors = [(andon - rs) / rs * 100 if rs != 0 else np.nan 
                          for rs, andon in zip(all_realsense_depths, all_andon_depths)]
        
        # Remove NaN values
        filtered_depths = []
        filtered_rel_errors = []
        for depth, rel_err in zip(all_realsense_depths, relative_errors):
            if not np.isnan(rel_err) and abs(rel_err) < 100:  # Filter extreme outliers
                filtered_depths.append(depth)
                filtered_rel_errors.append(round(rel_err, 2))
        
        plt.scatter(filtered_depths, filtered_rel_errors, alpha=0.5)
        
        # Add a smooth trend line using LOWESS or polynomial fit
        try:
            from scipy.signal import savgol_filter
            
            # Sort points by x-value for smooth line
            sorted_indices = np.argsort(filtered_depths)
            sorted_depths = [filtered_depths[i] for i in sorted_indices]
            sorted_errors = [filtered_rel_errors[i] for i in sorted_indices]
            
            # Apply Savitzky-Golay filter for smoothing
            window_length = min(51, len(sorted_depths) - 1)
            if window_length % 2 == 0:  # Must be odd
                window_length -= 1
                
            if window_length > 3:  # Need at least 3 points
                smoothed = savgol_filter(sorted_errors, window_length, 3)
                plt.plot(sorted_depths, smoothed, 'r-', linewidth=2, 
                        label='Trend Line')
        except Exception as e:
            print(f"Could not add trend line: {e}")
            
        plt.axhline(y=0, color='g', linestyle='--', linewidth=1.5)
        plt.xlabel('RealSense Depth (mm)')
        plt.ylabel('Relative Error (%)')
        plt.title('Relative Depth Error vs. Distance')
        plt.grid(True, alpha=0.3)
        if 'smoothed' in locals():
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'relative_error_vs_distance.png'))
        plt.close()
    
    # 5. NEW: Detection performance heatmap by angle and distance
    # Prepare data for the heatmap
    try:
        # Collect all aligned pairs from all tests
        all_pairs = []
        for result in all_results:
            test_data = result['original_data']
            pairs = align_timestamps(
                test_data['realsense_data'], 
                test_data['andon_data'],
                test_data['config']
            )
            # Tag each pair with the test angle
            test_angle = test_data['config'].get('angle', 0)
            for pair in pairs:
                if pair[0]['detected']:  # Only include pairs where RealSense detected
                    pair = (pair[0], pair[1], test_angle)
                    all_pairs.append(pair)
        
        if all_pairs:
            # Define distance bins (in mm)
            max_dist = 5000  # 5 meters in mm
            dist_bin_size = 500  # 0.5 meters in mm
            dist_bins = list(range(0, max_dist + dist_bin_size, dist_bin_size))
            
            # Define angle bins
            angles = sorted(set(pair[2] for pair in all_pairs if isinstance(pair[2], (int, float))))
            
            # Initialize data matrix for heatmap
            heatmap_data = np.zeros((len(angles), len(dist_bins)-1))
            counts_matrix = np.zeros((len(angles), len(dist_bins)-1))
            
            # Populate the matrices
            for rs, andon, angle in all_pairs:
                if rs['depth'] > max_dist:
                    continue
                    
                angle_idx = angles.index(angle)
                dist_idx = min(int(rs['depth'] / dist_bin_size), len(dist_bins)-2)
                
                # Increment count
                counts_matrix[angle_idx, dist_idx] += 1
                
                # Increment detection count if Andon detected too
                if andon['detected']:
                    heatmap_data[angle_idx, dist_idx] += 1
            
            # Calculate detection rates
            with np.errstate(divide='ignore', invalid='ignore'):
                detection_rate_matrix = np.divide(heatmap_data, counts_matrix)
                detection_rate_matrix = np.nan_to_num(detection_rate_matrix)  # Replace NaN with 0
            
            # Only create heatmap if we have enough data
            if np.sum(counts_matrix) > 10:  # Arbitrary threshold
                plt.figure(figsize=(12, 8))
                
                # Set up the heatmap
                plt.imshow(detection_rate_matrix, cmap='viridis', aspect='auto', 
                          vmin=0, vmax=1, interpolation='nearest')
                
                # Configure axes
                plt.colorbar(label='Detection Rate')
                
                # Set x-axis ticks and labels (distance bins) with 2 decimal precision
                x_ticks = np.arange(len(dist_bins)-1)
                x_labels = [f"{dist_bins[i]/1000:.1f}-{dist_bins[i+1]/1000:.1f}" for i in range(len(dist_bins)-1)]
                plt.xticks(x_ticks, x_labels, rotation=45)
                
                # Set y-axis ticks and labels (angles)
                y_ticks = np.arange(len(angles))
                plt.yticks(y_ticks, angles)
                
                plt.xlabel('Distance Range (m)')
                plt.ylabel('Angle (degrees)')
                plt.title('Detection Rate by Angle and Distance')
                
                # Add count numbers to cells with 2 decimal precision
                for i in range(len(angles)):
                    for j in range(len(dist_bins)-1):
                        count = counts_matrix[i, j]
                        if count > 0:
                            rate = detection_rate_matrix[i, j]
                            text_color = 'white' if rate < 0.6 else 'black'
                            plt.text(j, i, f"{int(count)}\n{rate:.2f}", ha='center', va='center', 
                                    color=text_color, fontsize=8)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'detection_heatmap.png'))
                plt.close()
    except Exception as e:
        print(f"Could not create detection heatmap: {e}")
    
    # 6. NEW: 3D surface plot of depth error by distance and angle
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        if all_realsense_depths and len(all_test_angles) > 0:
            # Convert angle strings to numbers if needed
            numeric_angles = []
            for angle in all_test_angles:
                if isinstance(angle, (int, float)):
                    numeric_angles.append(angle)
                elif isinstance(angle, str) and angle.strip('-').isdigit():
                    numeric_angles.append(float(angle))
                else:
                    numeric_angles.append(0)  # Default for non-numeric
            
            # Create a figure for 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Create the scatter plot with rounded error values
            rounded_errors = [round(err, 2) for err in all_errors]
            scatter = ax.scatter(
                all_realsense_depths, 
                numeric_angles, 
                rounded_errors,
                c=rounded_errors,  # Color by error
                cmap='coolwarm',
                alpha=0.7
            )
            
            # Add a color bar with 2 decimal precision
            colorbar = fig.colorbar(scatter, ax=ax, label='Depth Error (mm)', format='%.2f')
            
            # Set labels
            ax.set_xlabel('Distance (mm)')
            ax.set_ylabel('Angle (degrees)')
            ax.set_zlabel('Depth Error (mm)')
            
            ax.set_title('3D Visualization of Depth Error by Distance and Angle')
            
            # Adjust view angle
            ax.view_init(elev=30, azim=135)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'depth_error_3d.png'))
            plt.close()
    except Exception as e:
        print(f"Could not create 3D error plot: {e}")

def export_to_csv(all_results, output_dir):
    """Export analysis results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Summary metrics by test - All numerical values rounded to 2 decimal places
    summary_data = []
    for result in all_results:
        summary_data.append({
            'test_name': result['test_name'],
            'angle': result['config'].get('angle', 'N/A'),
            'direction': result['config'].get('direction', 'N/A'),
            'detection_rate': round(result['detection_metrics']['detection_rate'], 2),
            'false_positive_rate': round(result['detection_metrics']['false_positive_rate'], 2),
            'false_negative_rate': round(result['detection_metrics']['false_negative_rate'], 2),
            'depth_mae': round(result['depth_metrics']['mae'], 2),
            'depth_rmse': round(result['depth_metrics']['rmse'], 2),
            'depth_mean_relative_error': round(result['depth_metrics']['mean_relative_error'], 2),
            'depth_bias': round(result['depth_metrics']['bias'], 2),
            'depth_correlation': round(result['depth_metrics']['correlation'], 2)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'test_summary_metrics.csv'), index=False)
    
    # 2. Configuration analysis - All numerical values rounded to 2 decimal places
    config_analysis = analyze_by_config_params(all_results)
    
    # Angle analysis
    angle_data = []
    for angle, metrics in config_analysis['angle_analysis'].items():
        angle_data.append({
            'angle': angle,
            'test_count': metrics['count'],
            'avg_detection_rate': round(metrics['avg_detection_rate'], 2),
            'avg_depth_mae': round(metrics['avg_mae'], 2)
        })
    
    angle_df = pd.DataFrame(angle_data)
    angle_df.to_csv(os.path.join(output_dir, 'angle_analysis.csv'), index=False)
    
    # Direction analysis
    direction_data = []
    for direction, metrics in config_analysis['direction_analysis'].items():
        direction_data.append({
            'direction': direction,
            'test_count': metrics['count'],
            'avg_detection_rate': round(metrics['avg_detection_rate'], 2),
            'avg_depth_mae': round(metrics['avg_mae'], 2)
        })
    
    direction_df = pd.DataFrame(direction_data)
    direction_df.to_csv(os.path.join(output_dir, 'direction_analysis.csv'), index=False)
    
    # 3. Detection vs Distance - All numerical values rounded to 2 decimal places
    detection_distance_analysis = analyze_detection_vs_distance(all_results)
    
    distance_data = []
    for i in range(len(detection_distance_analysis['distance_bins'])):
        distance_data.append({
            'distance_bin': round(detection_distance_analysis['distance_bins'][i], 2),
            'detection_rate': round(detection_distance_analysis['detection_rates'][i], 2)
        })
    
    distance_df = pd.DataFrame(distance_data)
    distance_df.to_csv(os.path.join(output_dir, 'detection_vs_distance.csv'), index=False)

def main():
    # Set up directories
    base_dir = 'logs/dynamic_test'
    output_dir = 'analysis_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a log file for the analysis
    log_file = os.path.join(output_dir, 'analysis_log.txt')
    
    with open(log_file, 'w') as f:
        f.write(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Get all test directories
    test_dirs = glob.glob(os.path.join(base_dir, '*'))
    print(f"Found {len(test_dirs)} test directories")
    
    with open(log_file, 'a') as f:
        f.write(f"Found {len(test_dirs)} test directories\n")
    
    # Analyze each test
    all_results = []
    
    for test_dir in test_dirs:
        try:
            print(f"Analyzing {test_dir}...")
            test_data = load_test_data(test_dir)
            
            if test_data:
                # Get the absolute time range of the data
                rs_first_time = test_data['realsense_data'][0]['timestamp']
                rs_last_time = test_data['realsense_data'][-1]['timestamp']
                total_duration = rs_last_time - rs_first_time
                
                # Get absolute timestamp window from config
                start_timestamp = test_data['config'].get('start_timestamp', rs_first_time)
                end_timestamp = test_data['config'].get('end_timestamp', rs_last_time)
                
                # Calculate the effective time window
                effective_window = end_timestamp - start_timestamp
                
                # Count raw detections in the full dataset
                rs_detections = sum(1 for record in test_data['realsense_data'] if record['detected'])
                andon_detections = sum(1 for record in test_data['andon_data'] if record['detected'])
                
                # Log the time window being used with 2 decimal precision
                with open(log_file, 'a') as f:
                    f.write(f"\nAnalyzing {test_dir}:\n")
                    f.write(f"  - Test name: {test_data['test_name']}\n")
                    f.write(f"  - Angle: {test_data['config'].get('angle', 'Not specified')}\n")
                    f.write(f"  - Direction: {test_data['config'].get('direction', 'Not specified')}\n")
                    f.write(f"  - Total dataset duration: {total_duration:.2f} seconds\n")
                    f.write(f"  - Config absolute timestamps: {start_timestamp:.2f} to {end_timestamp:.2f}\n")
                    f.write(f"  - Effective time window: {effective_window:.2f} seconds\n")
                    f.write(f"  - All RealSense records: {len(test_data['realsense_data'])}\n")
                    f.write(f"  - All Andon records: {len(test_data['andon_data'])}\n")
                    f.write(f"  - Total RealSense detections: {rs_detections}\n")
                    f.write(f"  - Total Andon detections: {andon_detections}\n")
                
                # Perform analysis
                result = analyze_test_data(test_data)
                result['original_data'] = test_data  # Store original data for further analysis
                all_results.append(result)
                
                # Log each test results in more detail with 2 decimal precision
                with open(log_file, 'a') as f:
                    f.write(f"  - Analysis results:\n")
                    total_pairs = result['detection_metrics']['true_positives'] + result['detection_metrics']['false_positives'] + result['detection_metrics']['false_negatives'] + result['detection_metrics']['true_negatives']
                    f.write(f"    - Aligned pairs: {total_pairs}\n")
                    f.write(f"    - Detection rate: {result['detection_metrics']['detection_rate']:.2f}\n")
                    f.write(f"    - Depth MAE: {result['depth_metrics']['mae']:.2f} mm\n")
                    f.write(f"    - Valid depth pairs: {result['depth_metrics']['count']}\n")
        except Exception as e:
            print(f"Error analyzing {test_dir}: {e}")
            with open(log_file, 'a') as f:
                f.write(f"\nError analyzing {test_dir}: {e}\n")
    
    print(f"Analyzed {len(all_results)} tests successfully")
    
    try:
        # Generate visualizations
        print("Generating visualizations...")
        generate_visualizations(all_results, os.path.join(output_dir, 'figures'))
        
        # Export results to CSV
        print("Exporting results to CSV...")
        export_to_csv(all_results, output_dir)
        
        # Log overall statistics with 2 decimal precision
        with open(log_file, 'a') as f:
            f.write(f"\n\nOverall Statistics:\n")
            f.write(f"- Tests analyzed: {len(all_results)}\n")
            
            avg_detection = np.mean([r['detection_metrics']['detection_rate'] for r in all_results])
            avg_mae = np.mean([r['depth_metrics']['mae'] for r in all_results if r['depth_metrics']['count'] > 0])
            
            f.write(f"- Average detection rate: {avg_detection:.2f}\n")
            f.write(f"- Average depth MAE: {avg_mae:.2f} mm\n")
            f.write(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception as e:
        print(f"Error in post-processing: {e}")
        with open(log_file, 'a') as f:
            f.write(f"\nError in post-processing: {e}\n")
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()