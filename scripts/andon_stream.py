#!/usr/bin/env python3
"""
Andon System Stream Implementation

Adapts the existing Coral Micro with TOF sensor code to match
the format expected by the dynamic test visualizer.
"""

import logging
import numpy as np
import cv2
import time
import threading
import json
import base64
import requests
from threading import Thread, Lock


class AndonStream:
    """
    Andon system (Coral Micro) stream implementation.
    
    Connects to the Coral Micro board via HTTP and retrieves
    camera images, detection data, and depth information.
    """
    def __init__(self, ip="10.10.10.1", poll_interval=0.033):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing Andon system stream")
        
        # Configuration
        self.ip = ip
        self.poll_interval = poll_interval
        
        
        # Data storage
        self.frame = None
        self.detected = False
        self.confidence = 0.0
        self.depth = 0.0
        self.bbox = {'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}
        
        # Stats
        self.last_timestamp = 0
        self.last_inference_time = 0
        self.last_depth_estimation_time = 0
        
        # Thread safety
        self.lock = Lock()
        self.running = False
        self.thread = None
    
    def run(self):
        """Start the Andon system stream """
        try:
            self.running = True
            
            # Start polling thread
            self.thread = Thread(target=self._update_thread)
            self.thread.daemon = True
            self.thread.start()
            
            self.logger.info("Andon system stream started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Andon system stream: {e}", exc_info=True)
            return False
    
    
    def _update_thread(self):
        """Background thread for updating detection and depth data."""
        while self.running:
            try:
                # Fetch data from device
                data = self._fetch_data()
                if data:
                    self._process_data(data)
                
                # Sleep to maintain frame rate
                time.sleep(self.poll_interval)
                
            except Exception as e:
                self.logger.error(f"Error in Andon update thread: {e}")
                time.sleep(0.1)  # Short delay on error
    
    def _fetch_data(self):
        """Fetch data from the device using JSON-RPC."""
        try:
            payload = {
                'id': 1,
                'jsonrpc': '2.0',
                'method': 'tx_logs_to_host',
                'params': {}
            }
            
            response = requests.post(
                f'http://{self.ip}/jsonrpc',
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=2.0
            )
            
            if response.status_code != 200:
                self.logger.warning(f"Data fetch HTTP error: {response.status_code}")
                return None
            
            result = response.json()
            if 'error' in result:
                self.logger.warning(f"RPC error: {result['error']}")
                return None
                
            if 'result' not in result:
                self.logger.warning("No result in response")
                return None
                
            return result['result']
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return None
    
    def _process_data(self, data):
        """Process received data from the device."""
        try:
            # Parse basic metadata
            timestamp = data.get('timestamp', 0)
            detection_count = data.get('detection_count', 0)
            inference_time = data.get('inference_time', 0)
            depth_estimation_time = data.get('depth_estimation_time', 0)
            
            # Prepare variables for detection data
            frame = None
            detected = False
            confidence = 0.0
            depth = 0.0
            bbox = {'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0}
            
            # Process image data if available
            if 'image_data' in data and data.get('cam_width') and data.get('cam_height'):
                try:
                    width = data.get('cam_width')
                    height = data.get('cam_height')
                    image_bytes = base64.b64decode(data['image_data'])
                    
                    # Convert bytes to numpy array
                    img_data = np.frombuffer(image_bytes, dtype=np.uint8)
                    
                    # Check if the size matches what we expect
                    expected_size = width * height * 3  # Assuming RGB format (3 bytes per pixel)
                    if len(img_data) == expected_size:
                        frame = img_data.reshape((height, width, 3))
                    else:
                        self.logger.warning(f"Image data size mismatch: got {len(img_data)}, expected {expected_size}")
                
                except Exception as img_err:
                    self.logger.error(f"Error processing image data: {img_err}")
            
            # Process detection data if available
            if detection_count > 0 and 'detections' in data:
                try:
                    # Decode base64 detection data
                    detection_bytes = base64.b64decode(data['detections'])
                    detection_data = self._parse_detections(detection_bytes, detection_count)
                    
                    # Use the first detection (highest confidence)
                    if detection_data and detection_data['count'] > 0:
                        obj = detection_data['objects'][0]
                        detected = True
                        confidence = obj['score']
                        
                        # Get bounding box
                        obj_bbox = obj['bbox']
                        bbox = {
                            'xmin': obj_bbox['xmin'],
                            'ymin': obj_bbox['ymin'],
                            'xmax': obj_bbox['xmax'],
                            'ymax': obj_bbox['ymax']
                        }
                        
                        # Get depth if available
                        if 'depths' in data:
                            depth_bytes = base64.b64decode(data['depths'])
                            depth_data = self._parse_depths(depth_bytes, detection_count)
                            if depth_data and depth_data['count'] > 0:
                                depth = depth_data['depths'][0]
                
                except Exception as det_err:
                    self.logger.error(f"Error processing detection data: {det_err}")
            
            # Update state with thread safety
            with self.lock:
                self.last_timestamp = timestamp
                self.last_inference_time = inference_time
                self.last_depth_estimation_time = depth_estimation_time
                
                if frame is not None:
                    self.frame = frame
                
                self.detected = detected
                self.confidence = confidence
                self.depth = depth
                self.bbox = bbox
            
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
    
    def _parse_detections(self, detection_bytes, count):
        """Parse binary detection data according to tensorflow::Object structure."""
        try:
            detection_data = {
                'count': count,
                'objects': []
            }
            
            # Calculate the size of each Object struct
            # int + float + 4 floats for bbox = 24 bytes (on most platforms)
            object_size = 4 + 4 + (4 * 4)  # int + float + 4 floats
            
            for i in range(count):
                # Calculate offset for this object
                offset = i * object_size
                
                # Ensure we have enough data
                if offset + object_size > len(detection_bytes):
                    self.logger.warning(f"Detection data too short for object {i}")
                    break
                
                # Parse components
                obj_id = int.from_bytes(detection_bytes[offset:offset+4], byteorder='little')
                offset += 4
                
                score = np.frombuffer(detection_bytes[offset:offset+4], dtype=np.float32)[0]
                offset += 4
                
                # Parse BBox
                ymin = np.frombuffer(detection_bytes[offset:offset+4], dtype=np.float32)[0]
                offset += 4
                xmin = np.frombuffer(detection_bytes[offset:offset+4], dtype=np.float32)[0]
                offset += 4
                ymax = np.frombuffer(detection_bytes[offset:offset+4], dtype=np.float32)[0]
                offset += 4
                xmax = np.frombuffer(detection_bytes[offset:offset+4], dtype=np.float32)[0]
                
                # Add to objects list
                detection_data['objects'].append({
                    'id': obj_id,
                    'score': float(score),
                    'bbox': {
                        'ymin': float(ymin),
                        'xmin': float(xmin),
                        'ymax': float(ymax),
                        'xmax': float(xmax)
                    }
                })
            
            return detection_data
            
        except Exception as e:
            self.logger.error(f"Error parsing detection data: {e}")
            return {'count': 0, 'objects': []}
    
    def _parse_depths(self, depth_bytes, count):
        """Parse binary depth data."""
        try:
            # Parse as array of floats
            depths = np.frombuffer(depth_bytes, dtype=np.float32, count=count)
            return {
                'count': count,
                'depths': depths.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing depth data: {e}")
            return {'count': 0, 'depths': []}
    
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
                'timestamp': self.last_timestamp,
                'inference_time': self.last_inference_time,
                'depth_estimation_time': self.last_depth_estimation_time,
            }
    
    def stop(self):
        """Stop streaming and release resources."""
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
            
        self.logger.info("Andon system stream stopped")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    andon_stream = AndonStream(ip="10.10.10.1")

        # print dewtection data and show latest frame with cv2
    if andon_stream.run():
        try:
            while True:
                data = andon_stream.get_latest_data()
                if data:
                    print(f"Detected: {data['detected']}, Confidence: {data['confidence']}, Depth: {data['depth']}")
                    
                    # Show the latest frame
                    cv2.imshow("Frame", data['frame'])
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            andon_stream.stop()
            cv2.destroyAllWindows()
    else:
        print("Failed to start Andon stream.")
    
