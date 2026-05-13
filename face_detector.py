"""
Face detection module using YOLOv8-Face
Optimized for classroom environments with face-specific detection
"""

import cv2
import numpy as np
import torch
import os
import warnings

# Suppress PyTorch serialization warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.serialization')

from ultralytics import YOLO
import config


class FaceDetector:
    """
    YOLOv8 Face Detection with tracking capabilities
    Optimized for classroom multi-face detection including back-row students
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize the face detector
        
        Args:
            model_path: Path to YOLOv8 face model (default: uses pretrained face model)
            device: Device to run inference on ('cuda:0' or 'cpu')
        """
        self.device = device or config.DEVICE
        self.model_path = model_path or config.YOLO_MODEL
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.min_face_width = config.MIN_FACE_WIDTH
        self.min_face_height = config.MIN_FACE_HEIGHT
        self.inference_size = config.INFERENCE_SIZE
        self.use_half = config.USE_HALF_PRECISION
        
        # Check if CUDA is available
        if 'cuda' in self.device and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU")
            self.device = 'cpu'
            self.use_half = False  # FP16 only works on GPU
        
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize YOLOv8-Face model for face detection
        Downloads from Hugging Face if not present
        Uses weights_only=False for trusted models (PyTorch 2.6+ compatibility)
        """
        try:
            print(f"🔄 Loading YOLOv8-Face model...")
            
            # Check if model exists locally
            if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000000:
                try:
                    print(f"   Loading local model: {self.model_path}")
                    
                    # SOLUTION: Patch torch.load to use weights_only=False for trusted models
                    # This is safe because we control the model source (Hugging Face official)
                    original_load = torch.load
                    
                    def patched_load(*args, **kwargs):
                        # Force weights_only=False for YOLO models (trusted source)
                        kwargs['weights_only'] = False
                        return original_load(*args, **kwargs)
                    
                    # Temporarily patch torch.load
                    torch.load = patched_load
                    
                    try:
                        self.model = YOLO(self.model_path)
                        print(f"✅ Loaded face detection model: {self.model_path}")
                    finally:
                        # Restore original torch.load
                        torch.load = original_load
                    
                except Exception as e:
                    print(f"⚠️  Could not load {self.model_path}: {str(e)}")
                    self.model = None
            else:
                self.model = None
            
            # Download from Hugging Face if not available
            if self.model is None:
                print(f"⚠️  Face model not found locally")
                print(f"📥 Downloading YOLOv8n-Face from Hugging Face...")
                print(f"   This is a one-time download (~6MB)")
                
                try:
                    import urllib.request
                    
                    # Download from Hugging Face
                    url = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"
                    print(f"   Downloading from: {url}")
                    
                    urllib.request.urlretrieve(url, "yolov8n-face.pt")
                    
                    print(f"✅ Downloaded successfully")
                    
                    # Load model with patched torch.load
                    original_load = torch.load
                    
                    def patched_load(*args, **kwargs):
                        kwargs['weights_only'] = False
                        return original_load(*args, **kwargs)
                    
                    torch.load = patched_load
                    
                    try:
                        self.model = YOLO("yolov8n-face.pt")
                    finally:
                        torch.load = original_load
                    
                    self.model_path = "yolov8n-face.pt"
                    print(f"✅ Loaded YOLOv8n-Face model")
                    
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download face model: {str(e)}\n"
                        f"Please download manually from:\n"
                        f"https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt\n"
                        f"Save as: yolov8n-face.pt"
                    )
            
            # Move model to device
            self.model.to(self.device)
            
            # Enable half precision for GPU
            if self.use_half and 'cuda' in self.device:
                self.model.model.half()
                print(f"✅ Enabled FP16 half precision")
            
            print(f"✅ Model initialized on {self.device}")
            print(f"   Inference size: {self.inference_size}x{self.inference_size}")
            print(f"   Confidence threshold: {self.confidence_threshold}")
            print(f"   Min face size: {self.min_face_width}x{self.min_face_height}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize face detection model: {str(e)}")
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame using face-specific YOLO model
        Optimized for classroom environments with distant faces
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Run inference with larger image size for better distant face detection
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=config.IOU_THRESHOLD,
                imgsz=self.inference_size,  # Larger size for distant faces
                verbose=False,
                device=self.device,
                half=self.use_half
            )
            
            detections = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Calculate face dimensions
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Strict filtering for face-only detection
                    if (width >= self.min_face_width and 
                        height >= self.min_face_height and
                        confidence >= self.confidence_threshold):
                        
                        # Additional validation: aspect ratio check for faces
                        aspect_ratio = width / height if height > 0 else 0
                        
                        # Faces typically have aspect ratio between 0.6 and 1.4
                        if 0.5 <= aspect_ratio <= 1.5:
                            detections.append((
                                int(x1), int(y1), int(x2), int(y2), confidence
                            ))
            
            return detections
            
        except Exception as e:
            print(f"⚠️  Detection error: {str(e)}")
            return []
    
    def track_faces(self, frame, frame_number=0, debug=False):
        """
        Detect and track faces using YOLOv8-Face model with ByteTrack
        Optimized for classroom environments with robust tracking
        
        Args:
            frame: Input frame (BGR format)
            frame_number: Current frame number for tracking age
            debug: Enable debug output
            
        Returns:
            List of tracked detections: [(x1, y1, x2, y2, track_id, confidence, track_age), ...]
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Run tracking with ByteTrack configuration
            results = self.model.track(
                frame,
                conf=self.confidence_threshold,
                iou=config.IOU_THRESHOLD,
                imgsz=self.inference_size,
                persist=True,
                tracker="bytetrack.yaml",  # Use ByteTrack explicitly
                verbose=False,
                device=self.device,
                half=self.use_half
            )
            
            tracked_detections = []
            raw_detection_count = 0
            filtered_count = 0
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                raw_detection_count = len(boxes)
                
                if debug:
                    print(f"🔍 DEBUG: Raw YOLO detections: {raw_detection_count}")
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Get track ID
                    if box.id is not None:
                        track_id = int(box.id[0].cpu().numpy())
                    else:
                        if debug:
                            print(f"   ⚠️  Detection without track ID (conf: {confidence:.2f})")
                        continue  # Skip if no track ID
                    
                    # Calculate track age (frames since first detection)
                    if not hasattr(self, 'track_ages'):
                        self.track_ages = {}
                    
                    if track_id not in self.track_ages:
                        self.track_ages[track_id] = 0
                    else:
                        self.track_ages[track_id] += 1
                    
                    track_age = self.track_ages[track_id]
                    
                    # Calculate face dimensions
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Check filtering criteria
                    passes_size = width >= self.min_face_width and height >= self.min_face_height
                    passes_conf = confidence >= self.confidence_threshold
                    aspect_ratio = width / height if height > 0 else 0
                    passes_aspect = 0.5 <= aspect_ratio <= 1.5
                    
                    if debug and not (passes_size and passes_conf and passes_aspect):
                        print(f"   ❌ Filtered: Track {track_id}, size:{passes_size}, conf:{passes_conf}, aspect:{passes_aspect}")
                        filtered_count += 1
                    
                    # Strict filtering for face-only detection
                    if passes_size and passes_conf and passes_aspect:
                        tracked_detections.append((
                            int(x1), int(y1), int(x2), int(y2), 
                            track_id, confidence, track_age
                        ))
                        
                        if debug:
                            print(f"   ✅ Passed: Track {track_id}, bbox:({int(x1)},{int(y1)},{int(x2)},{int(y2)}), conf:{confidence:.2f}")
            
            if debug:
                print(f"🔍 DEBUG: Tracker output: {len(tracked_detections)} detections (filtered: {filtered_count})")
            
            return tracked_detections
            
        except Exception as e:
            print(f"⚠️  Tracking error: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def is_valid_face_detection(self, bbox, confidence):
        """
        Validate if a detection is a proper face
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Detection confidence
            
        Returns:
            True if valid face, False otherwise
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Check minimum size
        if width < self.min_face_width or height < self.min_face_height:
            return False
        
        # Check confidence
        if confidence < self.confidence_threshold:
            return False
        
        # Check aspect ratio (faces are roughly square to slightly rectangular)
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < 0.5 or aspect_ratio > 1.5:
            return False
        
        return True
