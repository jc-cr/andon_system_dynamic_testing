#!/usr/bin/env python3
"""
RealSense Stream Implementation

Adapts the existing RealSense D435 code to match the format
expected by the dynamic test visualizer with improved depth estimation.
"""

# Add paths to find the modules
import sys
sys.path.append('/home/example-object-tracker')
sys.path.append('/home/example-object-tracker/gstreamer')

import logging
import numpy as np
import cv2
import time
from threading import Thread, Lock
import pyrealsense2 as rs
import collections
import os
import importlib

import gstreamer
import common

Object = collections.namedtuple('Object', ['id', 'score', 'bbox', 'centroid'])

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box with normalized coordinates (0-1)."""
    __slots__ = ()

# Helper function for direct interpreter access (bypassing gstreamer dependency)
def set_input_tensor(interpreter, input_tensor):
    """Sets the input tensor directly without using GStreamer."""
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']
    
    # Check if shape includes batch dimension
    input_shape = input_details['shape']
    if len(input_shape) == 4 and input_shape[0] == 1:
        # Input expects batch dimension
        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, axis=0)
    
    # Set the tensor
    interpreter.set_tensor(tensor_index, input_tensor)

def get_output_tensor(interpreter, index):
    """Get output tensor data directly."""
    output_details = interpreter.get_output_details()[index]
    tensor = interpreter.get_tensor(output_details['index'])
    return tensor


class RealSenseStream:
    """
    RealSense D435 stream implementation.
    
    Uses the RealSense SDK to capture RGB and depth streams, and runs
    object detection with the specified model.
    """
    def __init__(self, 
                 model_path='/home/example-object-tracker/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite',
                 labels_path='/home/example-object-tracker/models/coco_labels.txt',
                 target_class="person",
                 detection_threshold=0.5,
                 depth_radius=1):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing RealSense stream")
        
        # Configuration
        self.model_path = model_path
        self.labels_path = labels_path
        self.target_class = target_class
        self.detection_threshold = detection_threshold
        self.depth_radius = depth_radius  # Radius for depth estimation
        
        # Initialize device and streams
        self.pipeline = None
        self.config = None
        self.profile = None
        self.align = None
        self.depth_scale = 0.001  # Default depth scale (meters)
        
        # TensorFlow interpreter
        self.interpreter = None
        self.labels = None
        
        # Data storage
        self.frame = None
        self.depth_frame = None
        self.detected = False
        self.confidence = 0.0
        self.depth = 0.0
        self.bbox = {'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}
        self.centroid = (0, 0)
        
        # Thread safety
        self.lock = Lock()
        self.running = False
        self.thread = None
    
    def load_labels(self, path):
        """Load label map from file."""
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            labels = {}
            for line in lines:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    labels[int(parts[0])] = parts[1].strip()
        return labels
    
    def run(self):
        """Initialize and start the RealSense stream."""
        try:
            # Initialize RealSense pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Enable streams
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start streaming
            self.profile = self.pipeline.start(self.config)
            
            # Get depth scale
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Align depth to color frame
            self.align = rs.align(rs.stream.color)
            
            # Load TensorFlow model and labels
            self.logger.info(f"Loading model: {self.model_path}")
            self.interpreter = common.make_interpreter(self.model_path)
            self.interpreter.allocate_tensors()
            self.labels = self.load_labels(self.labels_path)
            
            # Start processing thread
            self.running = True
            self.thread = Thread(target=self._update_thread)
            self.thread.daemon = True
            self.thread.start()
            
            self.logger.info("RealSense stream started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start RealSense stream: {e}", exc_info=True)
            return False
            
    def get_depth_at_point(self, depth_image, cx, cy, radius=1):
        """
        Get the average depth in a circular area around a point.
        
        Args:
            depth_image: The depth image
            cx, cy: Center coordinates
            radius: Radius of the circular area
            
        Returns:
            float: Average depth in mm
        """
        height, width = depth_image.shape[:2]
        
        # Create a mask for the circular area
        if radius <= 1:
            # For radius 1 or less, just use the center point
            depth_value = depth_image[cy, cx] if 0 <= cy < height and 0 <= cx < width else 0
            return float(depth_value) * self.depth_scale * 1000 if depth_value > 0 else 0
        
        # For larger radii, use a circular area
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x*x + y*y <= radius*radius
        
        # Get coordinates within the circle
        points_y, points_x = np.where(mask)
        points_y = points_y + cy - radius
        points_x = points_x + cx - radius
        
        # Filter points outside the image
        valid_indices = (points_y >= 0) & (points_y < height) & (points_x >= 0) & (points_x < width)
        points_y = points_y[valid_indices]
        points_x = points_x[valid_indices]
        
        # Get depth values
        depth_values = depth_image[points_y, points_x]
        valid_depths = depth_values[depth_values > 0]
        
        if len(valid_depths) > 0:
            # Use median for robustness
            return np.median(valid_depths).astype(float) * self.depth_scale * 1000
        
        return 0
    
    def _update_thread(self):
        """Background thread for updating detection and depth data."""
        while self.running:
            try:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                
                # Align depth frame to color frame
                aligned_frames = self.align.process(frames)
                
                # Get color and depth frames
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert frames to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Run object detection
                # Prepare input tensor
                input_size = self.interpreter.get_input_details()[0]['shape'][1:3]  # Height, width
                input_tensor = cv2.resize(color_image, (input_size[1], input_size[0]))
                
                # Convert to RGB if needed (TFLite models typically expect RGB)
                input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
                
                # Set input tensor directly instead of using common.set_input
                set_input_tensor(self.interpreter, input_tensor)
                
                # Run inference
                self.interpreter.invoke()
                
                # Get detection results directly instead of using common.output_tensor
                boxes = get_output_tensor(self.interpreter, 0)[0]  # First output tensor, first batch
                class_ids = get_output_tensor(self.interpreter, 1)[0]  # Second output tensor, first batch
                scores = get_output_tensor(self.interpreter, 2)[0]  # Third output tensor, first batch
                
                # Process detections
                detected = False
                best_confidence = 0
                best_bbox = None
                best_centroid = None
                best_depth = 0
                
                for i in range(len(scores)):
                    if scores[i] >= self.detection_threshold:
                        class_id = int(class_ids[i])
                        class_name = self.labels.get(class_id, "unknown")
                        
                        # Skip if not target class (e.g., "person")
                        if self.target_class and class_name != self.target_class:
                            continue
                            
                        # Get bounding box (normalized coordinates)
                        ymin, xmin, ymax, xmax = boxes[i]
                        
                        # Compute centroid
                        centroid_x = (xmin + xmax) / 2
                        centroid_y = (ymin + ymax) / 2
                        
                        # Get actual pixel coordinates for depth measurement
                        img_height, img_width = color_image.shape[:2]
                        cx = int(centroid_x * img_width)
                        cy = int(centroid_y * img_height)
                        
                        # Get depth using the specified radius
                        depth_mm = self.get_depth_at_point(depth_image, cx, cy, self.depth_radius)
                        
                        # Update if this is the highest confidence detection
                        if scores[i] > best_confidence:
                            detected = True
                            best_confidence = scores[i]
                            best_bbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
                            best_centroid = (centroid_x, centroid_y)
                            best_depth = depth_mm
                
                # Update state with thread safety
                with self.lock:
                    self.frame = color_image.copy()
                    self.depth_frame = depth_image.copy()
                    self.detected = detected
                    
                    if detected:
                        self.confidence = best_confidence
                        self.bbox = best_bbox
                        self.centroid = best_centroid
                        self.depth = best_depth
                    else:
                        # Clear detection data when no detections are present
                        self.confidence = 0.0
                        self.bbox = {'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}
                        self.centroid = (0, 0)
                        self.depth = 0.0
                
            except Exception as e:
                self.logger.error(f"Error in RealSense update thread: {e}", exc_info=True)
                time.sleep(0.1)  # Short delay on error
    
    def get_latest_data(self):
        """
        Get the latest frame and detection data.
        
        Returns:
            dict: Data including frame, detection status, confidence, depth, and bounding box
        """
        with self.lock:
            if self.frame is None:
                return None
                
            return {
                'frame': self.frame.copy(),
                'detected': self.detected,
                'confidence': float(self.confidence),
                'depth': float(self.depth),
                'bbox': self.bbox.copy(),
                'centroid': self.centroid,
                'depth_frame': self.depth_frame.copy() if self.depth_frame is not None else None
            }
    
    def set_depth_radius(self, radius):
        """Set the radius for depth estimation."""
        with self.lock:
            self.depth_radius = max(1, radius)  # Ensure radius is at least 1
    
    def stop(self):
        """Stop streaming and release resources."""
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        if self.pipeline:
            try:
                self.pipeline.stop()
                self.logger.info("RealSense pipeline stopped")
            except Exception as e:
                self.logger.error(f"Error stopping RealSense pipeline: {e}")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    model_path = "/home/example-object-tracker/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
    model_labels_path = "/home/example-object-tracker/models/coco_labels.txt"

    # Create RealSense stream with depth radius of 5 pixels
    realsense_stream = RealSenseStream(
        model_path=model_path,
        labels_path=model_labels_path,
        target_class="person",
        detection_threshold=0.5,
        depth_radius=5  # Use a 5-pixel radius for depth estimation
    )
    
    if realsense_stream.run():
        try:
            while True:
                data = realsense_stream.get_latest_data()
                if data:
                    # Print detection data to console for debugging
                    if data['detected']:
                        logging.info(f"Detected {data['confidence']:.2f} at depth {data['depth']:.2f}mm with bbox {data['bbox']} and centroid {data['centroid']}")
                    else:
                        logging.info("No detection")
                    # Process the data (e.g., display frame, handle detections)
                    cv2.imshow("RealSense Frame", data['frame'])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            realsense_stream.stop()
            cv2.destroyAllWindows()
    else:
        logging.error("Failed to start RealSense stream")