#!/usr/bin/env python3
import sys
sys.path.append('/home/example-object-tracker')
sys.path.append('/home/example-object-tracker/gstreamer')

import tkinter as tk
from tkinter import ttk
import time
import threading
import json
import os
import logging
from datetime import datetime
import cv2
from PIL import Image, ImageTk
import numpy as np
from threading import Lock

import argparse

from realsense_stream import RealSenseStream
from andon_stream import AndonStream


class DynamicTestVisualizer:
    def __init__(self, realsense_stream, andon_stream, start_delay=5):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Stream sources
        self.realsense_stream = realsense_stream
        self.andon_stream = andon_stream
        self.start_delay = start_delay
        
        # Data processing and recording
        self.data_lock = Lock()
        self.is_recording = False
        self.data_dir = None
        self.realsense_json_path = None
        self.andon_json_path = None
        self.countdown_active = False
        self.countdown_value = 0
        
        # Video recording
        self.realsense_video_writer = None
        self.andon_video_writer = None
        self.video_fps = 30
        
        # Setup main window
        self.root = tk.Tk()
        self.root.title("Dynamic Approach Test")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Create GUI elements
        self.create_gui()
        
    def create_gui(self):
        """Create GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top control panel
        control_frame = ttk.LabelFrame(main_frame, text="Test Controls")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Delay setting
        delay_frame = ttk.Frame(control_frame)
        delay_frame.pack(side=tk.LEFT, padx=20, pady=10)
        
        ttk.Label(delay_frame, text="Start Delay (seconds):").grid(row=0, column=0, padx=5, pady=5)
        self.delay_var = tk.IntVar(value=self.start_delay)
        delay_spinner = ttk.Spinbox(delay_frame, from_=1, to=30, textvariable=self.delay_var, width=5)
        delay_spinner.grid(row=0, column=1, padx=5, pady=5)
        
        # Record button
        self.record_button = ttk.Button(control_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Countdown display
        countdown_frame = ttk.Frame(control_frame)
        countdown_frame.pack(side=tk.LEFT, padx=20, pady=10)
        ttk.Label(countdown_frame, text="Countdown:").grid(row=0, column=0, padx=5, pady=5)
        self.countdown_var = tk.StringVar(value="--")
        self.countdown_label = ttk.Label(countdown_frame, textvariable=self.countdown_var, font=("Arial", 14, "bold"))
        self.countdown_label.grid(row=0, column=1, padx=5, pady=5)
        
        # Status indicator
        status_frame = ttk.Frame(control_frame)
        status_frame.pack(side=tk.LEFT, padx=20, pady=10)
        ttk.Label(status_frame, text="Status:").grid(row=0, column=0, padx=5, pady=5)
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.grid(row=0, column=1, padx=5, pady=5)
        
        # Exit button
        ttk.Button(control_frame, text="Exit", command=self.on_close).pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Video display section - side by side
        video_frame = ttk.Frame(main_frame)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        # RealSense (Ground Truth) Display
        realsense_frame = ttk.LabelFrame(video_frame, text="RealSense (Ground Truth)")
        realsense_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.realsense_info_frame = ttk.Frame(realsense_frame)
        self.realsense_info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.realsense_info_frame, text="Detected:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.realsense_detected_var = tk.StringVar(value="No")
        ttk.Label(self.realsense_info_frame, textvariable=self.realsense_detected_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(self.realsense_info_frame, text="Confidence:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.realsense_confidence_var = tk.StringVar(value="--")
        ttk.Label(self.realsense_info_frame, textvariable=self.realsense_confidence_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(self.realsense_info_frame, text="Depth:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.realsense_depth_var = tk.StringVar(value="--")
        ttk.Label(self.realsense_info_frame, textvariable=self.realsense_depth_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.realsense_image_frame = ttk.Frame(realsense_frame)
        self.realsense_image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.realsense_image_label = ttk.Label(self.realsense_image_frame)
        self.realsense_image_label.pack(fill=tk.BOTH, expand=True)
        
        # Andon System Display
        andon_frame = ttk.LabelFrame(video_frame, text="Andon System")
        andon_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.andon_info_frame = ttk.Frame(andon_frame)
        self.andon_info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(self.andon_info_frame, text="Detected:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.andon_detected_var = tk.StringVar(value="No")
        ttk.Label(self.andon_info_frame, textvariable=self.andon_detected_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(self.andon_info_frame, text="Confidence:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.andon_confidence_var = tk.StringVar(value="--")
        ttk.Label(self.andon_info_frame, textvariable=self.andon_confidence_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(self.andon_info_frame, text="Depth:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.andon_depth_var = tk.StringVar(value="--")
        ttk.Label(self.andon_info_frame, textvariable=self.andon_depth_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.andon_image_frame = ttk.Frame(andon_frame)
        self.andon_image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.andon_image_label = ttk.Label(self.andon_image_frame)
        self.andon_image_label.pack(fill=tk.BOTH, expand=True)
    
    
    def update_display(self):
        """Update display with latest data from both streams"""
        try:
            # Get latest data from RealSense
            realsense_data = self.realsense_stream.get_latest_data()
            andon_data = self.andon_stream.get_latest_data()
            
            if realsense_data:
                # Update detection info
                detected = realsense_data.get('detected', False)
                self.realsense_detected_var.set("Yes" if detected else "No")
                
                if detected:
                    confidence = realsense_data.get('confidence', 0)
                    depth = realsense_data.get('depth', 0)
                    self.realsense_confidence_var.set(f"{confidence:.2f}")
                    self.realsense_depth_var.set(f"{depth/1000:.2f} m")
                else:
                    self.realsense_confidence_var.set("--")
                    self.realsense_depth_var.set("--")
                
                # Update image
                frame = realsense_data.get('frame')
                if frame is not None:
                    # Create a copy for the display
                    display_frame = frame.copy()
                    
                    # Fix color if needed (handle BGR to RGB conversion)
                    if display_frame.ndim == 3 and display_frame.shape[2] == 3:
                        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    # Draw bounding box overlay for display
                    cv_frame_with_overlay = self._add_overlay_to_frame(display_frame, realsense_data, is_realsense=True)
                    
                    # Convert to PIL for display
                    pil_frame = Image.fromarray(cv_frame_with_overlay)
                    self.realsense_photo = ImageTk.PhotoImage(pil_frame)
                    self.realsense_image_label.configure(image=self.realsense_photo)
                    
                    # Add frame with overlay to video if recording
                    if self.is_recording and self.realsense_video_writer is not None:
                        # Convert back to BGR for video writer
                        video_frame = cv2.cvtColor(cv_frame_with_overlay, cv2.COLOR_RGB2BGR)
                        self.realsense_video_writer.write(video_frame)
            
            if andon_data:
                # Update detection info
                detected = andon_data.get('detected', False)
                self.andon_detected_var.set("Yes" if detected else "No")
                
                if detected:
                    confidence = andon_data.get('confidence', 0)
                    depth = andon_data.get('depth', 0)
                    self.andon_confidence_var.set(f"{confidence:.2f}")
                    self.andon_depth_var.set(f"{depth/1000:.2f} m")
                else:
                    self.andon_confidence_var.set("--")
                    self.andon_depth_var.set("--")
                
                # Update image
                frame = andon_data.get('frame')
                if frame is not None:
                    # Create a copy for the display
                    display_frame = frame.copy()
                    
                    # Draw bounding box overlay for display
                    cv_frame_with_overlay = self._add_overlay_to_frame(display_frame, andon_data, is_realsense=False)
                    
                    # Convert to PIL for display
                    pil_frame = Image.fromarray(cv_frame_with_overlay)
                    self.andon_photo = ImageTk.PhotoImage(pil_frame)
                    self.andon_image_label.configure(image=self.andon_photo)
                    
                    # Add frame with overlay to video if recording
                    if self.is_recording and self.andon_video_writer is not None:
                        # Convert back to BGR for video writer
                        video_frame = cv2.cvtColor(cv_frame_with_overlay, cv2.COLOR_RGB2BGR)
                        self.andon_video_writer.write(video_frame)
            
            # Record data if active
            if self.is_recording:
                self._record_data(realsense_data, andon_data)
        
        except Exception as e:
            self.logger.error(f"Error updating display: {e}")
        
        # Schedule next update
        self.root.after(33, self.update_display)  # ~30 FPS




    
    def _add_overlay_to_frame(self, frame, data, is_realsense=True):
        """Add detection overlay to frame"""
        try:
            if isinstance(frame, np.ndarray):
                # Create a copy to avoid modifying the original
                display_frame = frame.copy()
                
                # Convert frame if needed
                if display_frame.ndim == 2:  # Grayscale
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2RGB)
                
                # Draw detection box if available
                if data.get('detected', False):
                    bbox = data.get('bbox', {})
                    if bbox:
                        # Get frame dimensions
                        height, width = display_frame.shape[:2]
                        
                        if is_realsense:
                            # RealSense uses normalized coordinates (0-1)
                            xmin = int(bbox.get('xmin', 0) * width)
                            ymin = int(bbox.get('ymin', 0) * height)
                            xmax = int(bbox.get('xmax', 0) * width)
                            ymax = int(bbox.get('ymax', 0) * height)
                        else:
                            # Andon system uses absolute pixel coordinates
                            xmin = int(bbox.get('xmin', 0))
                            ymin = int(bbox.get('ymin', 0))
                            xmax = int(bbox.get('xmax', 0))
                            ymax = int(bbox.get('ymax', 0))
                            
                            # Ensure coordinates are within image bounds
                            xmin = max(0, min(xmin, width-1))
                            ymin = max(0, min(ymin, height-1))
                            xmax = max(0, min(xmax, width-1))
                            ymax = max(0, min(ymax, height-1))
                        
                        # Draw rectangle
                        cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        
                        # Draw depth info
                        depth = data.get('depth', 0)
                        depth_text = f"Depth: {depth/1000:.2f}m"
                        text_y_pos = max(20, ymin - 10)  # Ensure text is visible
                        cv2.putText(display_frame, depth_text, 
                                   (xmin, text_y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Add confidence info if available
                        confidence = data.get('confidence', 0)
                        if confidence > 0:
                            conf_text = f"Conf: {confidence:.2f}"
                            conf_y_pos = max(50, ymin - 30)  # Ensure text is visible
                            cv2.putText(display_frame, conf_text,
                                      (xmin, conf_y_pos),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                return display_frame
            else:
                self.logger.warning(f"Unknown frame type for overlay: {type(frame)}")
                return frame
        
        except Exception as e:
            self.logger.error(f"Error adding overlay to frame: {e}")
            return frame
    
    def _prepare_frame(self, frame, data):
        """Prepare frame for display with detection overlay - legacy method kept for compatibility"""
        is_realsense = 'centroid' not in data  # RealSense has centroid, Andon doesn't
        fixed_frame = frame.copy()
        if fixed_frame.ndim == 3 and fixed_frame.shape[2] == 3:
            fixed_frame = cv2.cvtColor(fixed_frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(self._add_overlay_to_frame(fixed_frame, data, is_realsense=is_realsense))
    
    def toggle_recording(self):
        """Toggle recording state with countdown"""
        if self.is_recording:
            # Stop recording
            self.is_recording = False
            self.record_button.configure(text="Start Recording")
            self.status_var.set("Ready")
            self.countdown_var.set("--")
            
            # Finalize videos
            self._finalize_recording()
            
            self.logger.info(f"Recording stopped. Data saved to {self.data_dir}")
        else:
            # Start countdown then recording
            self.record_button.configure(state="disabled")
            self.start_countdown()
    
    def start_countdown(self):
        """Start countdown before recording"""
        self.countdown_active = True
        self.countdown_value = self.delay_var.get()
        self.countdown_var.set(str(self.countdown_value))
        self.status_var.set("Countdown...")
        
        # Create log directory
        self.data_dir = os.path.join("logs", "dynamic_test", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Setup JSON files for data recording
        self.realsense_json_path = os.path.join(self.data_dir, "realsense_data.json")
        self.andon_json_path = os.path.join(self.data_dir, "andon_data.json")
        
        # Initialize empty lists in the JSON files
        with open(self.realsense_json_path, 'w') as f:
            json.dump([], f)
        
        with open(self.andon_json_path, 'w') as f:
            json.dump([], f)
            
            
        # Start countdown
        self._update_countdown()
    
    def _update_countdown(self):
        """Update countdown timer"""
        if not self.countdown_active:
            return
            
        if self.countdown_value > 0:
            self.countdown_var.set(str(self.countdown_value))
            self.countdown_value -= 1
            self.root.after(1000, self._update_countdown)
        else:
            # Countdown finished, start recording
            self.countdown_active = False
            self.is_recording = True
            self.record_button.configure(text="Stop Recording", state="normal")
            self.status_var.set("Recording...")
            self.countdown_var.set("0")
            
            # Initialize video writers
            self._initialize_video_writers()
            
            self.logger.info(f"Recording started to {self.data_dir}")
    
    def _initialize_video_writers(self):
        """Initialize video writers for both streams"""
        try:
            # Define video paths
            realsense_video_path = os.path.join(self.data_dir, "realsense_video.mp4")
            andon_video_path = os.path.join(self.data_dir, "andon_video.mp4")
            
            # Get sample frames to determine dimensions
            realsense_data = self.realsense_stream.get_latest_data()
            andon_data = self.andon_stream.get_latest_data()
            
            # RealSense video writer
            if realsense_data and 'frame' in realsense_data:
                frame = realsense_data['frame']
                if isinstance(frame, np.ndarray):
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.realsense_video_writer = cv2.VideoWriter(
                        realsense_video_path, fourcc, self.video_fps, (w, h)
                    )
            
            # Andon video writer
            if andon_data and 'frame' in andon_data:
                frame = andon_data['frame']
                if isinstance(frame, np.ndarray):
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.andon_video_writer = cv2.VideoWriter(
                        andon_video_path, fourcc, self.video_fps, (w, h)
                    )
                    
        except Exception as e:
            self.logger.error(f"Error initializing video writers: {e}")
    
    def _finalize_recording(self):
        """Finalize recording and close video writers"""
        try:
            # Close video writers
            if self.realsense_video_writer:
                self.realsense_video_writer.release()
                self.realsense_video_writer = None
            
            if self.andon_video_writer:
                self.andon_video_writer.release()
                self.andon_video_writer = None
                
        except Exception as e:
            self.logger.error(f"Error finalizing recording: {e}")
    
    




    
    def on_close(self):
        """Handle window close event"""
        self.logger.info("Shutting down...")
        
        # Ensure recording is properly finalized
        if self.is_recording:
            self.is_recording = False
            self._finalize_recording()
            
        self.root.destroy()
    
    def run(self):
        """Start visualizer"""
        # Start display updates
        self.root.after(100, self.update_display)
        
        # Start main loop
        self.root.mainloop()


    def _record_data(self, realsense_data, andon_data):
        """Record data from both streams with synchronized timestamps"""
        try:
            # Use the same timestamp for both records to ensure synchronization
            timestamp = datetime.now().timestamp()
            
            # Record RealSense data
            if realsense_data:
                realsense_entry = {
                    'timestamp': timestamp,
                    'detected': realsense_data.get('detected', False),
                    'confidence': float(realsense_data.get('confidence', 0)),
                    'depth': float(realsense_data.get('depth', 0))
                }
                
                # Add bounding box data if available
                if realsense_data.get('detected', False) and 'bbox' in realsense_data:
                    bbox = realsense_data.get('bbox', {})
                    realsense_entry['bbox'] = {
                        'xmin': float(bbox.get('xmin', 0)),
                        'ymin': float(bbox.get('ymin', 0)), 
                        'xmax': float(bbox.get('xmax', 0)),
                        'ymax': float(bbox.get('ymax', 0))
                    }
                
                # Load existing data
                with open(self.realsense_json_path, 'r') as f:
                    realsense_records = json.load(f)
                
                # Append new entry
                realsense_records.append(realsense_entry)
                
                # Save updated data
                with open(self.realsense_json_path, 'w') as f:
                    json.dump(realsense_records, f)
            
            # Record Andon data with the SAME timestamp
            if andon_data:
                andon_entry = {
                    'timestamp': timestamp,  # Use the same timestamp for synchronization
                    'detected': andon_data.get('detected', False),
                    'confidence': float(andon_data.get('confidence', 0)),
                    'depth': float(andon_data.get('depth', 0))
                }
                
                # Add bounding box data if available
                if andon_data.get('detected', False) and 'bbox' in andon_data:
                    bbox = andon_data.get('bbox', {})
                    andon_entry['bbox'] = {
                        'xmin': float(bbox.get('xmin', 0)),
                        'ymin': float(bbox.get('ymin', 0)), 
                        'xmax': float(bbox.get('xmax', 0)),
                        'ymax': float(bbox.get('ymax', 0))
                    }
                
                # Load existing data
                with open(self.andon_json_path, 'r') as f:
                    andon_records = json.load(f)
                
                # Append new entry
                andon_records.append(andon_entry)
                
                # Save updated data
                with open(self.andon_json_path, 'w') as f:
                    json.dump(andon_records, f)
                
        except Exception as e:
            self.logger.error(f"Error recording data: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Dynamic Testing for Person Detection")
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="Delay in seconds before starting data collection (default: 5)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse command line arguments
    args = parse_args()
    delay = args.delay
    
    # Configure simple console logging
    logging.basicConfig(
        level=logging.INFO if not args.debug else logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    logger = logging.getLogger("main")
    logger.info(f"Starting Dynamic Testing with {delay}s start delay")
    
    try:
        # Initialize stream classes
        realsense_stream = RealSenseStream()
        andon_stream = AndonStream()

        # start streams
        realsense_stream.run()
        andon_stream.run()
        
        # Initialize gui
        visualizer = DynamicTestVisualizer(
            realsense_stream=realsense_stream,
            andon_stream=andon_stream,
            start_delay=delay
        )
        
        # Start the visualizer
        visualizer.run()
        
    except Exception as e:
        logger.error(f"Application error: {e}")

        # Cleanup streams
        realsense_stream.stop()
        andon_stream.stop()

        return 1

    finally:
        # Cleanup streams
        realsense_stream.stop()
        andon_stream.stop()

        return 0
    

if __name__ == "__main__":
    sys.exit(main())