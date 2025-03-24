#!/usr/bin/env python3
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
import sys

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
                    frame = self._prepare_frame(frame, realsense_data)
                    self.realsense_photo = ImageTk.PhotoImage(frame)
                    self.realsense_image_label.configure(image=self.realsense_photo)
            
            # Get latest data from Andon system
            andon_data = self.andon_stream.get_latest_data()
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
                    frame = self._prepare_frame(frame, andon_data)
                    self.andon_photo = ImageTk.PhotoImage(frame)
                    self.andon_image_label.configure(image=self.andon_photo)
            
            # Record data if active
            if self.is_recording:
                self._record_data(realsense_data, andon_data)
        
        except Exception as e:
            self.logger.error(f"Error updating display: {e}", exc_info=True)
        
        # Schedule next update
        self.root.after(33, self.update_display)  # ~30 FPS
    
    def _prepare_frame(self, frame, data):
        """Prepare frame for display with detection overlay"""
        try:
            if isinstance(frame, np.ndarray):
                # Convert frame to PIL Image
                if frame.ndim == 2:  # Grayscale
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[2] == 4:  # RGBA
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                else:
                    display_frame = frame.copy()
                
                # Draw detection box if available
                if data.get('detected', False):
                    bbox = data.get('bbox', {})
                    if bbox:
                        # Get frame dimensions
                        height, width = display_frame.shape[:2]
                        
                        # Get normalized coordinates (assuming bbox has xmin, ymin, xmax, ymax)
                        xmin = int(bbox.get('xmin', 0) * width)
                        ymin = int(bbox.get('ymin', 0) * height)
                        xmax = int(bbox.get('xmax', 0) * width)
                        ymax = int(bbox.get('ymax', 0) * height)
                        
                        # Draw rectangle
                        cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        
                        # Draw depth info
                        depth = data.get('depth', 0)
                        depth_text = f"Depth: {depth/1000:.2f}m"
                        cv2.putText(display_frame, depth_text, 
                                   (xmin, ymin - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Convert to PIL Image
                return Image.fromarray(display_frame)
            elif isinstance(frame, Image.Image):
                return frame
            else:
                self.logger.warning(f"Unknown frame type: {type(frame)}")
                return None
        
        except Exception as e:
            self.logger.error(f"Error preparing frame: {e}", exc_info=True)
            return None
    
    def toggle_recording(self):
        """Toggle recording state with countdown"""
        if self.is_recording:
            # Stop recording
            self.is_recording = False
            self.record_button.configure(text="Start Recording")
            self.status_var.set("Ready")
            self.countdown_var.set("--")
            self.logger.info("Recording stopped")
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
            self.logger.info(f"Recording started to {self.data_dir}")
    
    def _record_data(self, realsense_data, andon_data):
        """Record data from both streams"""
        try:
            timestamp = datetime.now().timestamp()
            
            # Prepare RealSense data
            if realsense_data:
                realsense_entry = {
                    'timestamp': timestamp,
                    'detected': realsense_data.get('detected', False),
                    'confidence': realsense_data.get('confidence', 0),
                    'depth': realsense_data.get('depth', 0)
                }
                
                # Load existing data
                with open(self.realsense_json_path, 'r') as f:
                    realsense_records = json.load(f)
                
                # Append new entry
                realsense_records.append(realsense_entry)
                
                # Save updated data
                with open(self.realsense_json_path, 'w') as f:
                    json.dump(realsense_records, f)
            
            # Prepare Andon data
            if andon_data:
                andon_entry = {
                    'timestamp': timestamp,
                    'detected': andon_data.get('detected', False),
                    'confidence': andon_data.get('confidence', 0),
                    'depth': andon_data.get('depth', 0)
                }
                
                # Load existing data
                with open(self.andon_json_path, 'r') as f:
                    andon_records = json.load(f)
                
                # Append new entry
                andon_records.append(andon_entry)
                
                # Save updated data
                with open(self.andon_json_path, 'w') as f:
                    json.dump(andon_records, f)
            
            # Save frames if available
            if realsense_data and 'frame' in realsense_data:
                frame_path = os.path.join(self.data_dir, f"realsense_{timestamp:.3f}.jpg")
                cv2.imwrite(frame_path, realsense_data['frame'])
            
            if andon_data and 'frame' in andon_data:
                frame_path = os.path.join(self.data_dir, f"andon_{timestamp:.3f}.jpg")
                cv2.imwrite(frame_path, andon_data['frame'])
                
        except Exception as e:
            self.logger.error(f"Error recording data: {e}", exc_info=True)
    
    def on_close(self):
        """Handle window close event"""
        self.logger.info("Shutting down...")
        self.root.destroy()
    
    def run(self):
        """Start visualizer"""
        # Start display updates
        self.root.after(100, self.update_display)
        
        # Start main loop
        self.root.mainloop()


# Setup logging
def setup_logging(debug=False):

    """Setup logging configuration"""
    # Create logs directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Setup handlers
    handlers = [logging.StreamHandler()]
    
    # Add file handler
    log_file = os.path.join(log_dir, f"dynamic_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    handlers.append(file_handler)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger("main")


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
    
    # Setup logging
    logger = setup_logging(args.debug)
    logger.info(f"Starting Dynamic Testing with {delay}s start delay")
    
    try:
        # Initialize stream classes
        # Replace these with your actual implementations
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
        logger.error(f"Application error: {e}", exc_info=True)

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