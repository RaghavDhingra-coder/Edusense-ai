"""
Configuration file for classroom face detection and tracking system
"""

# Detection Configuration
CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence for face detection (lowered for debugging)
MIN_FACE_SIZE = 30  # Minimum face size in pixels (width or height)
MIN_FACE_WIDTH = 30  # Minimum face width in pixels (lowered for debugging)
MIN_FACE_HEIGHT = 30  # Minimum face height in pixels (lowered for debugging)
IOU_THRESHOLD = 0.45  # IoU threshold for NMS
INFERENCE_SIZE = 640  # Standard inference size (lowered from 1280 for debugging)

# Tracking Configuration - ByteTrack for robust tracking
TRACKER_TYPE = "bytetrack"  # ByteTrack for better persistence
TRACK_HIGH_THRESH = 0.6     # High confidence threshold for tracks
TRACK_LOW_THRESH = 0.3      # Low confidence threshold (keeps tracks alive longer)
TRACK_BUFFER = 60           # Frames to keep lost tracks (increased for classroom)
MATCH_THRESH = 0.8          # Matching threshold for ByteTrack
MAX_AGE = 60                # Maximum frames to keep track alive without detection
MIN_HITS = 3                # Minimum hits before track is confirmed

# Face Re-Identification Configuration
ENABLE_REID = True          # Enable face re-identification
REID_SIMILARITY_THRESHOLD = 0.6  # Cosine similarity threshold (0.6 = 60% match)
REID_EMBEDDING_CACHE_DURATION = 5.0  # Cache embeddings for 5 seconds
REID_MAX_EMBEDDINGS_PER_STUDENT = 5  # Max embeddings to store per student
REID_EXTRACT_INTERVAL = 30  # Extract embedding every N frames for existing tracks
REID_LOW_CONFIDENCE_THRESHOLD = 0.7  # Re-extract if confidence drops below this

# Video Processing Configuration
TARGET_FPS = 15  # Process video at this FPS for performance
DISPLAY_SCALE = 1.0  # Scale factor for display window

# Image Saving Configuration
SAVE_INTERVAL = 1.0  # Save one image per second per student
CROP_PADDING = 15  # Padding around face crop in pixels (reduced for tighter crops)
OUTPUT_DIR = "students"  # Base directory for saving student images
IMAGE_FORMAT = "jpg"  # Image format for saving
IMAGE_QUALITY = 95  # JPEG quality (1-100)
MIN_CROP_SIZE = 30  # Minimum crop dimension to save (width or height)

# Display Configuration
BBOX_COLOR = (0, 255, 0)  # Green bounding box
BBOX_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)
TEXT_THICKNESS = 2
TEXT_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.8
FPS_COLOR = (0, 0, 255)  # Red for FPS counter

# Model Configuration - FACE DETECTION SPECIFIC
YOLO_MODEL = "yolov8n-face.pt"  # YOLOv8 face detection model
DEVICE = "cuda:0"  # Use "cuda:0" for GPU, "cpu" for CPU
USE_HALF_PRECISION = True  # Use FP16 for faster inference on GPU (set False for CPU)
