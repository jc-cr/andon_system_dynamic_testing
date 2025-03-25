import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    
    # Print time window info
    rs_first_time = realsense_data[0]['timestamp']
    print(f"  - Using time window: {start_timestamp} to {end_timestamp}")
    print(f"  - Relative to start: {start_timestamp - rs_first_time:.2f}s to {end_timestamp - rs_first_time:.2f}s")
    
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
            latency = andon_record['timestamp'] - rs_record['timestamp']
            detection_latencies.append(latency)
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
    
    # Average detection latency
    avg_detection_latency = np.mean(detection_latencies) if detection_latencies else 0
    
    # Average confidence scores
    avg_true_positive_confidence = np.mean(true_positive_confidences) if true_positive_confidences else 0
    avg_false_positive_confidence = np.mean(false_positive_confidences) if false_positive_confidences else 0
    
    return {
        'total_pairs': total_pairs,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'detection_rate': detection_rate,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'avg_detection_latency': avg_detection_latency,
        'avg_true_positive_confidence': avg_true_positive_confidence,
        'avg_false_positive_confidence': avg_false_positive_confidence
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
    
    # Print depth range info
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
        'mae': mae,
        'rmse': rmse,
        'mean_relative_error': mean_relative_error,
        'bias': bias,
        'correlation': correlation,
        'p_value': p_value,
        'realsense_depths': realsense_depths,
        'andon_depths': andon_depths,
        'errors': errors,
        'max_depth_threshold': max_valid_depth
    }

def analyze_test_data(test_data):
    """Analyze a single test's data and compute metrics."""
    # Print timestamp ranges
    rs_first_time = test_data['realsense_data'][0]['timestamp']
    rs_last_time = test_data['realsense_data'][-1]['timestamp']
    andon_first_time = test_data['andon_data'][0]['timestamp']
    andon_last_time = test_data['andon_data'][-1]['timestamp']
    
    print(f"  - RealSense data range: {rs_first_time} to {rs_last_time} ({rs_last_time - rs_first_time:.2f}s)")
    print(f"  - Andon data range: {andon_first_time} to {andon_last_time} ({andon_last_time - andon_first_time:.2f}s)")
    
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

# [All other functions remain the same as in the original script]

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
                # Log basic test information
                with open(log_file, 'a') as f:
                    f.write(f"\nAnalyzing {test_dir}:\n")
                    f.write(f"  - Test name: {test_data['test_name']}\n")
                    f.write(f"  - Angle: {test_data['config'].get('angle', 'Not specified')}\n")
                    f.write(f"  - Direction: {test_data['config'].get('direction', 'Not specified')}\n")
                
                # Perform analysis
                result = analyze_test_data(test_data)
                result['original_data'] = test_data  # Store original data for further analysis
                all_results.append(result)
                
                # Log each test results in more detail
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
        
        # Log overall statistics
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


# This function should be run once to create example config.json files
def generate_example_configs():
    """Generate example config.json files with absolute timestamps"""
    example_config = {
        "angle": 30,
        "direction": "forward",
        "start_timestamp": 1742846099.641373,  # Replace with actual timestamp
        "end_timestamp": 1742846104.858797     # Replace with actual timestamp
    }
    
    with open("example_config.json", "w") as f:
        json.dump(example_config, f, indent=2)
    
    print("Generated example_config.json file")


if __name__ == "__main__":
    main()

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
            detection_rates.append(detection_rate)
    
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
                'avg_detection_rate': np.mean(detection_rates),
                'avg_mae': np.mean([mae for mae in maes if mae > 0])  # Skip zero MAEs
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
                'avg_detection_rate': np.mean(detection_rates),
                'avg_mae': np.mean([mae for mae in maes if mae > 0])  # Skip zero MAEs
            }
    
    return {
        'angle_analysis': angle_analysis,
        'direction_analysis': direction_analysis
    }

def generate_visualizations(all_results, output_dir):
    """Generate and save visualization figures with improved formatting."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create separate lists for forward and backward tests
    forward_tests = [r for r in all_results if r['config'].get('direction', '').lower() in ['forward', 'fwd']]
    backward_tests = [r for r in all_results if r['config'].get('direction', '').lower() in ['backward', 'back', 'bck']]
    
    # 1. Detection Rate by Test (Split by direction)
    # Forward tests
    if forward_tests:
        # Sort by angle for consistent display
        forward_tests = sorted(forward_tests, key=lambda r: r['config'].get('angle', 0))
        test_names = [f"{r['test_name']} ({r['config'].get('angle', 'N/A')}°)" for r in forward_tests]
        detection_rates = [r['detection_metrics']['detection_rate'] for r in forward_tests]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(test_names, detection_rates)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom')
            
        plt.xlabel('Test Name (Angle)')
        plt.ylabel('Detection Rate')
        plt.title('Detection Rate by Test - Forward Direction')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)  # Set y-axis limit with some margin
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_rate_forward_tests.png'))
        plt.close()
    
    # Backward tests
    if backward_tests:
        # Sort by angle for consistent display
        backward_tests = sorted(backward_tests, key=lambda r: r['config'].get('angle', 0))
        test_names = [f"{r['test_name']} ({r['config'].get('angle', 'N/A')}°)" for r in backward_tests]
        detection_rates = [r['detection_metrics']['detection_rate'] for r in backward_tests]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(test_names, detection_rates)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom')
            
        plt.xlabel('Test Name (Angle)')
        plt.ylabel('Detection Rate')
        plt.title('Detection Rate by Test - Backward Direction')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)  # Set y-axis limit with some margin
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_rate_backward_tests.png'))
        plt.close()
        
    # Combined detection rate visualization
    # Sort all results by angle for consistent display
    sorted_results = sorted(all_results, key=lambda r: r['config'].get('angle', 0))
    test_labels = [f"{r['test_name']} ({r['config'].get('angle', 'N/A')}°)" for r in sorted_results]
    directions = [r['config'].get('direction', 'N/A') for r in sorted_results]
    detection_rates = [r['detection_metrics']['detection_rate'] for r in sorted_results]
    
    plt.figure(figsize=(14, 7))
    
    # Create a color map based on direction
    colors = ['#1f77b4' if d.lower() in ['backward', 'back', 'bck'] else '#ff7f0e' for d in directions]
    
    bars = plt.bar(test_labels, detection_rates, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom')
        
    plt.xlabel('Test Name (Angle)')
    plt.ylabel('Detection Rate')
    plt.title('Detection Rate by Test and Direction')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.1)  # Set y-axis limit with some margin
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Backward Direction'),
        Patch(facecolor='#ff7f0e', label='Forward Direction')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_rate_by_test.png'))
    plt.close()
    
    # 2. Depth Error Analysis - Improved with coloring by test
    # Combine all depth data with test information
    all_realsense_depths = []
    all_errors = []
    all_test_names = []  # Track which test each point belongs to
    all_test_angles = []  # Track angle information
    all_test_directions = []  # Track direction information
    
    for result in all_results:
        if 'realsense_depths' in result['depth_metrics'] and result['depth_metrics']['realsense_depths']:
            count = len(result['depth_metrics']['realsense_depths'])
            all_realsense_depths.extend(result['depth_metrics']['realsense_depths'])
            all_errors.extend(result['depth_metrics']['errors'])
            
            # Extend the test information for each data point
            test_name = f"{result['test_name']} ({result['config'].get('angle', 'N/A')}°)"
            all_test_names.extend([test_name] * count)
            all_test_angles.extend([result['config'].get('angle', 0)] * count)
            all_test_directions.extend([result['config'].get('direction', '')] * count)
    
    if all_realsense_depths and all_errors:
        # Create a figure with a larger size for better visibility
        plt.figure(figsize=(14, 8))
        
        # Get unique tests for color mapping
        unique_tests = list(set(all_test_names))
        
        # Create a custom colormap 
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10', len(unique_tests))
        
        # Create color dictionary
        color_dict = {test: cmap(i) for i, test in enumerate(unique_tests)}
        
        # Create a scatter plot with different colors
        for test in unique_tests:
            # Get indices for this test
            indices = [i for i, t in enumerate(all_test_names) if t == test]
            
            # Get depth and error values for these indices
            depths = [all_realsense_depths[i] for i in indices]
            errors = [all_errors[i] for i in indices]
            
            # Plot this test with its assigned color
            plt.scatter(depths, errors, alpha=0.6, label=test, color=color_dict[test])
        
        plt.axhline(y=0, color='r', linestyle='-', linewidth=1.5)
        plt.xlabel('RealSense Depth (mm)')
        plt.ylabel('Error: Andon - RealSense (mm)')
        plt.title('Depth Error vs. Actual Distance (Colored by Test)')
        plt.grid(True, alpha=0.3)
        
        # Add a legend (outside the plot area)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'depth_error_vs_distance.png'), bbox_inches='tight')
        plt.close()
        
        # Also create an alternative version colored by direction
        plt.figure(figsize=(14, 8))
        
        # Get unique directions
        unique_directions = sorted(set(all_test_directions))
        
        # Plot points by direction
        for direction in unique_directions:
            # Get indices for this direction
            indices = [i for i, d in enumerate(all_test_directions) if d.lower() == direction.lower()]
            
            # Get depth and error values for these indices
            depths = [all_realsense_depths[i] for i in indices]
            errors = [all_errors[i] for i in indices]
            
            # Use consistent colors for forward/backward
            color = '#ff7f0e' if 'forward' in direction.lower() or 'fwd' in direction.lower() else '#1f77b4'
            
            # Plot this direction
            plt.scatter(depths, errors, alpha=0.6, label=direction, color=color)
        
        plt.axhline(y=0, color='r', linestyle='-', linewidth=1.5)
        plt.xlabel('RealSense Depth (mm)')
        plt.ylabel('Error: Andon - RealSense (mm)')
        plt.title('Depth Error vs. Actual Distance (Colored by Direction)')
        plt.grid(True, alpha=0.3)
        
        # Add a legend
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'depth_error_vs_distance_by_direction.png'))
        plt.close()
    
    # 3. Detection Probability vs. Distance
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
    
    # 4. Performance by Angle - IMPROVED ORDERING
    config_analysis = analyze_by_config_params(all_results)
    
    if config_analysis['angle_analysis']:
        # Sort angles numerically to ensure proper order from -30 to 30
        angles = sorted(config_analysis['angle_analysis'].keys(), key=lambda x: float(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.strip('-').isdigit()) else 0)
        detection_rates = [config_analysis['angle_analysis'][a]['avg_detection_rate'] for a in angles]
        
        plt.figure(figsize=(10, 6))
        
        # Use different colors for negative and positive angles
        colors = ['#ff7f0e' if (isinstance(a, (int, float)) and a < 0) or (isinstance(a, str) and a.startswith('-')) else '#1f77b4' for a in angles]
        
        bars = plt.bar(angles, detection_rates, color=colors)
        
        # Add value labels
        for bar, rate in zip(bars, detection_rates):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Average Detection Rate')
        plt.title('Detection Rate by Angle')
        plt.ylim(0, 1.05)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Ensure x-axis ticks show all angles
        plt.xticks(angles)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_by_angle.png'))
        plt.close()
        
        # Also create a depth MAE by angle plot
        mae_values = [config_analysis['angle_analysis'][a]['avg_mae'] for a in angles]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(angles, mae_values, color=colors)
        
        # Add value labels
        for bar, mae in zip(bars, mae_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Average Depth MAE (mm)')
        plt.title('Depth Estimation Error by Angle')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Ensure x-axis ticks show all angles
        plt.xticks(angles)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'depth_error_by_angle.png'))
        plt.close()
def export_to_csv(all_results, output_dir):
    """Export analysis results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Summary metrics by test
    summary_data = []
    for result in all_results:
        summary_data.append({
            'test_name': result['test_name'],
            'angle': result['config'].get('angle', 'N/A'),
            'direction': result['config'].get('direction', 'N/A'),
            'detection_rate': result['detection_metrics']['detection_rate'],
            'false_positive_rate': result['detection_metrics']['false_positive_rate'],
            'false_negative_rate': result['detection_metrics']['false_negative_rate'],
            'avg_detection_latency': result['detection_metrics']['avg_detection_latency'],
            'depth_mae': result['depth_metrics']['mae'],
            'depth_rmse': result['depth_metrics']['rmse'],
            'depth_mean_relative_error': result['depth_metrics']['mean_relative_error'],
            'depth_bias': result['depth_metrics']['bias'],
            'depth_correlation': result['depth_metrics']['correlation']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'test_summary_metrics.csv'), index=False)
    
    # 2. Configuration analysis
    config_analysis = analyze_by_config_params(all_results)
    
    # Angle analysis
    angle_data = []
    for angle, metrics in config_analysis['angle_analysis'].items():
        angle_data.append({
            'angle': angle,
            'test_count': metrics['count'],
            'avg_detection_rate': metrics['avg_detection_rate'],
            'avg_depth_mae': metrics['avg_mae']
        })
    
    angle_df = pd.DataFrame(angle_data)
    angle_df.to_csv(os.path.join(output_dir, 'angle_analysis.csv'), index=False)
    
    # Direction analysis
    direction_data = []
    for direction, metrics in config_analysis['direction_analysis'].items():
        direction_data.append({
            'direction': direction,
            'test_count': metrics['count'],
            'avg_detection_rate': metrics['avg_detection_rate'],
            'avg_depth_mae': metrics['avg_mae']
        })
    
    direction_df = pd.DataFrame(direction_data)
    direction_df.to_csv(os.path.join(output_dir, 'direction_analysis.csv'), index=False)
    
    # 3. Detection vs Distance
    detection_distance_analysis = analyze_detection_vs_distance(all_results)
    
    distance_data = []
    for i in range(len(detection_distance_analysis['distance_bins'])):
        distance_data.append({
            'distance_bin': detection_distance_analysis['distance_bins'][i],
            'detection_rate': detection_distance_analysis['detection_rates'][i]
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
                
                # Log the time window being used
                with open(log_file, 'a') as f:
                    f.write(f"\nAnalyzing {test_dir}:\n")
                    f.write(f"  - Test name: {test_data['test_name']}\n")
                    f.write(f"  - Angle: {test_data['config'].get('angle', 'Not specified')}\n")
                    f.write(f"  - Direction: {test_data['config'].get('direction', 'Not specified')}\n")
                    f.write(f"  - Total dataset duration: {total_duration:.2f} seconds\n")
                    f.write(f"  - Config absolute timestamps: {start_timestamp} to {end_timestamp}\n")
                    f.write(f"  - Effective time window: {effective_window:.2f} seconds\n")
                    f.write(f"  - All RealSense records: {len(test_data['realsense_data'])}\n")
                    f.write(f"  - All Andon records: {len(test_data['andon_data'])}\n")
                    f.write(f"  - Total RealSense detections: {rs_detections}\n")
                    f.write(f"  - Total Andon detections: {andon_detections}\n")
                
                # Perform analysis
                result = analyze_test_data(test_data)
                result['original_data'] = test_data  # Store original data for further analysis
                all_results.append(result)
                
                # Log each test results in more detail
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
        
        # Log overall statistics
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