"""
Integrated Flask Server for EduSence AI
Combines camera streaming, detection, tracking, and analytics in one unified dashboard
"""

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import logging
from datetime import datetime
import cv2
import threading
import time
from queue import Queue
import numpy as np
import shutil
from pathlib import Path

# Import existing components - NO CHANGES TO LOGIC
from face_detector import FaceDetector
from image_manager import ImageManager
from face_reid import FaceReID
from hybrid_attentiveness import HybridAttentivenessAnalyzer
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)  # Enable CORS

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global state
analytics_engine = None
camera_system = None
camera_lock = threading.Lock()
current_session_id = None
video_processing_active = False


def generate_session_id(prefix="session"):
    """Generate unique session ID"""
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_session_dir(session_id):
    """Get directory for specific session"""
    return os.path.join('sessions', session_id)


def clear_temp_session_data(session_dir):
    """
    Clear temporary session data while preserving persistent identity database
    
    CLEARS:
    - Session student folders
    - Temporary face crops
    - Session analytics cache
    
    PRESERVES:
    - identity_database.pkl (persistent identity memory)
    - Model files
    - Configuration files
    """
    try:
        if os.path.exists(session_dir):
            logger.info(f"🧹 Clearing temporary session data: {session_dir}")
            shutil.rmtree(session_dir)
            logger.info("✅ Session data cleared")
        
        # Create fresh session directory
        os.makedirs(session_dir, exist_ok=True)
        logger.info(f"📁 Created fresh session directory: {session_dir}")
        
    except Exception as e:
        logger.error(f"❌ Failed to clear session data: {e}")
        raise


class CameraSystem:
    """
    Camera processing system - PRESERVES ALL EXISTING LOGIC
    Runs in background thread, streams to web
    NOW WITH SESSION MANAGEMENT
    Supports both webcam (video_source=0) and video files (video_source=path)
    """
    
    def __init__(self, video_source=0, session_id=None, is_video_file=False):
        """Initialize camera system with existing components"""
        self.video_source = video_source
        self.is_video_file = is_video_file
        self.session_id = session_id or generate_session_id()
        self.session_dir = get_session_dir(self.session_id)
        
        self.running = False
        self.thread = None
        self.frame_queue = Queue(maxsize=2)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Statistics
        self.frame_number = 0
        self.active_tracks = set()
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_counter = 0
        
        # Video file specific
        self.total_frames = 0
        self.processing_complete = False
        self.processing_error = None
        
        # Initialize components - EXACT SAME AS main.py
        logger.info("=" * 60)
        logger.info(f"🎓 Initializing Camera System - Session: {self.session_id}")
        logger.info("=" * 60)
        
        self.detector = None
        self.image_manager = None
        self.reid_system = None
        self.cap = None
        
        # Clear old session data and create fresh session
        self._prepare_session()
        self._initialize_components()
    
    def _prepare_session(self):
        """Prepare fresh session - clear temporary data"""
        logger.info("🔄 Preparing new session...")
        
        # Clear temporary session data
        clear_temp_session_data(self.session_dir)
        
        logger.info("✅ Session prepared")
    
    def _initialize_components(self):
        """Initialize all components - PRESERVES EXISTING LOGIC"""
        try:
            logger.info("📦 Initializing components...")
            
            # Face detector - SAME AS main.py
            self.detector = FaceDetector()
            
            # Image manager - USE SESSION DIRECTORY
            self.image_manager = ImageManager(self.session_dir)
            
            # ReID system - SAME AS main.py
            # NOTE: ReID will use persistent identity database if it exists
            if config.ENABLE_REID:
                logger.info("🔄 Initializing Face Re-Identification...")
                self.reid_system = FaceReID(
                    similarity_threshold=config.REID_SIMILARITY_THRESHOLD,
                    embedding_size=512
                )
                # Reset ReID session state (but keep persistent identity DB)
                self.reid_system.reset_session()
            else:
                logger.info("⚠️  Face ReID disabled")
                self.reid_system = None
            
            logger.info("✅ Components initialized")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            raise
    
    def start(self):
        """Start camera/video processing thread"""
        with camera_lock:
            if self.running:
                logger.warning("Processing already running")
                return False
            
            try:
                # Validate video source
                if self.video_source is None:
                    error_msg = "Video source is None"
                    logger.error(f"[CAMERA] ❌ {error_msg}")
                    self.processing_error = error_msg
                    return False
                
                if self.is_video_file:
                    logger.info(f"[VIDEO] Opening video file: {self.video_source}")
                    logger.info(f"[VIDEO] File exists: {os.path.exists(self.video_source)}")
                    logger.info(f"[VIDEO] File type: {type(self.video_source)}")
                else:
                    logger.info("[CAMERA] Opening camera...")
                
                # Open video source (camera or file)
                self.cap = cv2.VideoCapture(self.video_source)
                if not self.cap.isOpened():
                    error_msg = f"Failed to open {'video file' if self.is_video_file else 'camera'}: {self.video_source}"
                    logger.error(f"[CAMERA] ❌ {error_msg}")
                    self.processing_error = error_msg
                    return False
                
                if self.is_video_file:
                    # Get video properties
                    self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    logger.info(f"[VIDEO] ✅ Video opened: {width}x{height}, {video_fps:.1f} FPS, {self.total_frames} frames")
                else:
                    logger.info("[CAMERA] ✅ Camera opened successfully")
                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Verify video source is working by reading a test frame
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    error_msg = f"Failed to read test frame from {'video file' if self.is_video_file else 'camera'}"
                    logger.error(f"[CAMERA] ❌ {error_msg}")
                    self.cap.release()
                    self.processing_error = error_msg
                    return False
                
                logger.info(f"[CAMERA] ✅ Test frame captured: {test_frame.shape}")
                
                # Reset video to beginning after test
                if self.is_video_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                self.running = True
                self.frame_number = 0
                self.active_tracks = set()
                self.processing_complete = False
                self.processing_error = None
                
                # Start processing thread
                self.thread = threading.Thread(target=self._process_loop, daemon=True)
                self.thread.start()
                
                logger.info("[CAMERA] ✅ Processing thread started")
                logger.info("=" * 60)
                return True
                
            except Exception as e:
                error_msg = f"Failed to start: {str(e)}"
                logger.error(f"[CAMERA] ❌ {error_msg}")
                import traceback
                traceback.print_exc()
                self.processing_error = error_msg
                return False
    
    def stop(self):
        """Stop camera/video processing"""
        with camera_lock:
            if not self.running:
                return False
            
            self.running = False
            
            if self.thread:
                self.thread.join(timeout=2.0)
            
            if self.cap:
                self.cap.release()
                self.cap = None
            
            source_type = "video" if self.is_video_file else "camera"
            logger.info(f"🛑 {source_type.capitalize()} stopped")
            return True
    
    def _process_loop(self):
        """
        Main processing loop - PRESERVES ALL EXISTING LOGIC FROM main.py
        Works for both webcam and video files
        """
        logger.info("[STREAM] 🚀 Processing loop started")
        
        frame_count = 0
        try:
            while self.running:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    if self.is_video_file:
                        # Video file ended
                        logger.info("[STREAM] ✅ Video processing complete")
                        self.processing_complete = True
                        self.running = False
                        break
                    else:
                        # Camera read failed, retry
                        logger.warning("[STREAM] ⚠️ Failed to read frame")
                        time.sleep(0.1)
                        continue
                
                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames
                    logger.info(f"[STREAM] Frame {frame_count} captured: {frame.shape}")
                
                # Process frame - EXACT SAME LOGIC AS main.py
                processed_frame = self._process_frame(frame)
                
                if frame_count % 30 == 0:
                    logger.info(f"[STREAM] Frame {frame_count} processed")
                
                # Update latest frame for streaming - CRITICAL
                with self.frame_lock:
                    self.latest_frame = processed_frame.copy()
                    if frame_count % 30 == 0:
                        logger.info(f"[STREAM] ✅ Shared frame updated (frame {frame_count})")
                
                # Update FPS
                self._update_fps()
                
                # Small delay to prevent CPU overload (only for webcam)
                if not self.is_video_file:
                    time.sleep(0.01)
                # For video files, process as fast as possible
                
        except Exception as e:
            logger.error(f"[STREAM] ❌ Processing loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info("[STREAM] Processing loop stopped")
            logger.info(f"[STREAM] Total frames processed: {frame_count}")
    
    def _process_frame(self, frame):
        """
        Process single frame - EXACT SAME LOGIC AS main.py process_frame()
        """
        self.frame_number += 1
        current_time = time.time()
        
        # Debug for first frames
        debug = self.frame_number <= 50
        
        # Track faces - SAME AS main.py
        detections = self.detector.track_faces(frame, self.frame_number, debug=debug)
        
        # Track current frame's track IDs
        current_tracks = set()
        
        # Process each detection with ReID - SAME AS main.py
        reid_detections = []
        
        for detection in detections:
            x1, y1, x2, y2, track_id, confidence, track_age = detection
            current_tracks.add(track_id)
            
            # Validate detection - SAME AS main.py
            bbox = (x1, y1, x2, y2)
            if not self.detector.is_valid_face_detection(bbox, confidence):
                continue
            
            # Crop face - SAME AS main.py
            face_crop = self.image_manager.crop_face(frame, bbox)
            if face_crop is None:
                continue
            
            # Apply ReID - SAME AS main.py
            if self.reid_system and config.ENABLE_REID:
                should_extract = self.reid_system.should_extract_embedding(
                    track_id, track_age, confidence
                )
                
                student_id, is_new, similarity = self.reid_system.register_or_match_face(
                    track_id, face_crop, current_time, force_extract=should_extract
                )
                
                if is_new and track_age == 0:
                    logger.info(f"✨ New student: Student {student_id} (Track {track_id})")
            else:
                student_id = track_id
            
            # Save face image - SAME AS main.py
            self.image_manager.save_face_image(frame, bbox, student_id, confidence)
            
            # Store for display
            reid_detections.append((x1, y1, x2, y2, student_id, confidence))
        
        # Handle lost tracks - SAME AS main.py
        if self.reid_system and config.ENABLE_REID:
            lost_tracks = self.active_tracks - current_tracks
            for lost_track in lost_tracks:
                self.reid_system.handle_track_lost(lost_track)
            
            if self.frame_number % 100 == 0:
                self.reid_system.cleanup_old_tracks(current_time, timeout=30.0)
        
        self.active_tracks = current_tracks
        
        # Draw detections - SAME AS main.py
        annotated_frame = self._draw_detections(frame, reid_detections)
        
        # Draw info overlay
        annotated_frame = self._draw_info_overlay(annotated_frame, len(reid_detections))
        
        return annotated_frame
    
    def _draw_detections(self, frame, detections):
        """Draw bounding boxes and labels - SAME AS main.py"""
        for x1, y1, x2, y2, student_id, confidence in detections:
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label
            label = f"Student {student_id}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Background for label
            cv2.rectangle(frame, 
                         (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)),
                         color, -1)
            
            # Label text
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Confidence
            conf_text = f"{confidence:.2f}"
            cv2.putText(frame, conf_text, (int(x1), int(y2) + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def _draw_info_overlay(self, frame, num_faces):
        """Draw info overlay on frame"""
        stats = self.image_manager.get_statistics()
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Draw text
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {num_faces}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Students: {stats['total_students']}", (200, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame, f"Images: {stats['total_images']}", (200, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def get_frame(self):
        """Get latest processed frame for streaming"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None
    
    def get_stats(self):
        """Get current statistics"""
        stats = self.image_manager.get_statistics()
        stats['fps'] = self.fps
        stats['frame_number'] = self.frame_number
        stats['active_tracks'] = len(self.active_tracks)
        stats['running'] = self.running
        stats['is_video_file'] = self.is_video_file
        stats['total_frames'] = self.total_frames
        stats['processing_complete'] = self.processing_complete
        stats['processing_error'] = self.processing_error
        stats['progress_percent'] = (self.frame_number / self.total_frames * 100) if self.total_frames > 0 else 0
        return stats


def get_analytics_engine():
    """Get or create analytics engine for CURRENT SESSION"""
    global analytics_engine, camera_system, current_session_id
    
    # Use current session directory if camera is running
    if camera_system and current_session_id:
        students_dir = get_session_dir(current_session_id)
    else:
        # Fallback to default students directory
        students_dir = 'students'
    
    # Always create fresh analytics engine for current session
    logger.info(f"📊 Creating analytics engine for: {students_dir}")
    analytics_engine = HybridAttentivenessAnalyzer(
        students_dir=students_dir,
        focus_threshold=0.65,
        yaw_threshold=30.0,
        pitch_threshold=30.0,
        roll_threshold=35.0,
        consecutive_distraction_threshold=2
    )
    
    return analytics_engine


def get_camera_system():
    """Get or create camera system - returns existing instance if available"""
    global camera_system
    
    # Return existing camera system if available
    # DO NOT create new instance - that would break video streaming
    if camera_system is not None:
        return camera_system
    
    # Only create new instance if none exists (should not happen during normal operation)
    logger.warning("⚠️ No camera system exists, creating placeholder")
    return None


def generate_frames():
    """Generate frames for MJPEG streaming"""
    frame_count = 0
    logger.info("[MJPEG] Stream generator started")
    
    while True:
        # Use global camera_system directly - DO NOT call get_camera_system()
        # Calling get_camera_system() would create a different instance
        global camera_system
        
        frame = None
        if camera_system is not None:
            frame = camera_system.get_frame()
        
        if frame is None:
            # Send placeholder frame
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera Not Started", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame = placeholder
            if frame_count % 30 == 0:
                logger.debug("[MJPEG] Sending placeholder frame")
        else:
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"[MJPEG] ✅ Sending processed frame {frame_count}: {frame.shape}")
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        if not ret:
            logger.warning("[MJPEG] ⚠️ Failed to encode frame")
            continue
        
        if frame_count % 30 == 0:
            logger.info(f"[MJPEG] Frame encoded: {len(buffer)} bytes")
        
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/')
def index():
    """Serve main dashboard"""
    return send_from_directory('frontend', 'index_integrated.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('frontend', path)


@app.route('/api/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera processing with NEW SESSION"""
    try:
        logger.info("=" * 60)
        logger.info("🎥 Starting new camera session...")
        logger.info("=" * 60)
        
        # Stop existing camera if running
        global camera_system, current_session_id, video_processing_active
        if camera_system and camera_system.running:
            logger.info("🛑 Stopping previous session...")
            camera_system.stop()
            time.sleep(0.5)  # Brief pause for cleanup
        
        # Create NEW camera system with fresh session
        new_session_id = generate_session_id("webcam")
        current_session_id = new_session_id
        video_processing_active = False
        
        logger.info(f"🆕 Creating new camera system for session: {new_session_id}")
        camera_system = CameraSystem(video_source=0, session_id=new_session_id, is_video_file=False)
        
        success = camera_system.start()
        
        if success:
            logger.info("=" * 60)
            logger.info(f"✅ New session started: {camera_system.session_id}")
            logger.info(f"📁 Session directory: {camera_system.session_dir}")
            logger.info("=" * 60)
            
            return jsonify({
                'success': True,
                'message': 'Camera started successfully',
                'session_id': camera_system.session_id,
                'session_dir': camera_system.session_dir,
                'source_type': 'webcam'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to start camera'
            }), 500
            
    except Exception as e:
        logger.error(f"Start camera error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/video/upload', methods=['POST'])
def upload_video():
    """Upload video file for processing"""
    try:
        logger.info("=" * 60)
        logger.info("📤 Video upload request received")
        logger.info(f"📋 Request method: {request.method}")
        logger.info(f"📋 Request files: {list(request.files.keys())}")
        logger.info("=" * 60)
        
        # Check if file is present
        if 'video' not in request.files:
            logger.error("❌ No 'video' field in request.files")
            return jsonify({
                'success': False,
                'error': 'No video file provided'
            }), 400
        
        file = request.files['video']
        logger.info(f"📁 File received: {file.filename}")
        
        # Check if filename is empty
        if file.filename == '':
            logger.error("❌ Empty filename")
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            logger.error(f"❌ Invalid file type: {file.filename}")
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Secure filename
        original_filename = file.filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        
        logger.info(f"📝 Original filename: {original_filename}")
        logger.info(f"📝 Secured filename: {filename}")
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"💾 Saving video to: {filepath}")
        logger.info(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
        logger.info(f"📁 Absolute path: {os.path.abspath(filepath)}")
        
        file.save(filepath)
        
        # Verify file was saved
        if not os.path.exists(filepath):
            logger.error(f"❌ File not found after save: {filepath}")
            return jsonify({
                'success': False,
                'error': 'Failed to save file'
            }), 500
        
        # Get file size
        file_size = os.path.getsize(filepath)
        file_size_mb = file_size / (1024 * 1024)
        
        logger.info(f"✅ Video uploaded successfully!")
        logger.info(f"   Filename: {filename}")
        logger.info(f"   Size: {file_size_mb:.2f} MB")
        logger.info(f"   Path: {filepath}")
        logger.info(f"   Exists: {os.path.exists(filepath)}")
        logger.info("=" * 60)
        
        return jsonify({
            'success': True,
            'message': 'Video uploaded successfully',
            'filename': filename,
            'filepath': filepath,
            'size_mb': round(file_size_mb, 2)
        })
        
    except Exception as e:
        logger.error(f"❌ Video upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/video/process', methods=['POST'])
def process_video():
    """Start processing uploaded video"""
    try:
        logger.info("=" * 60)
        logger.info("🎬 VIDEO PROCESSING REQUEST RECEIVED")
        logger.info("=" * 60)
        
        # Log raw request data
        logger.info(f"📋 Request method: {request.method}")
        logger.info(f"📋 Request content type: {request.content_type}")
        logger.info(f"📋 Request data (raw): {request.data}")
        
        # Parse JSON
        data = request.get_json()
        logger.info(f"📋 Parsed JSON: {data}")
        logger.info(f"📋 JSON keys: {list(data.keys()) if data else 'None'}")
        
        if not data:
            logger.error("❌ No JSON data in request")
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        if 'filepath' not in data:
            logger.error(f"❌ No 'filepath' key in JSON. Keys present: {list(data.keys())}")
            return jsonify({
                'success': False,
                'error': 'No filepath provided'
            }), 400
        
        filepath = data['filepath']
        
        logger.info(f"📁 Extracted filepath: {filepath}")
        logger.info(f"📁 Filepath type: {type(filepath)}")
        logger.info(f"📁 Filepath value: '{filepath}'")
        
        # Validate filepath is not None or empty
        if filepath is None:
            logger.error("❌ filepath is None")
            return jsonify({
                'success': False,
                'error': 'Invalid filepath: None'
            }), 400
        
        if not filepath:
            logger.error(f"❌ filepath is empty or falsy: '{filepath}'")
            return jsonify({
                'success': False,
                'error': 'Invalid filepath: empty or None'
            }), 400
        
        if not isinstance(filepath, str):
            logger.error(f"❌ filepath is not a string: {type(filepath)}")
            return jsonify({
                'success': False,
                'error': f'Invalid filepath type: {type(filepath).__name__}'
            }), 400
        
        if filepath.strip() == '':
            logger.error("❌ filepath is empty string after strip")
            return jsonify({
                'success': False,
                'error': 'Invalid filepath: empty string'
            }), 400
        
        logger.info("✅ Filepath validation passed")
        
        # Validate filepath exists
        logger.info(f"📁 Checking if file exists: {filepath}")
        logger.info(f"📁 Absolute path: {os.path.abspath(filepath)}")
        logger.info(f"📁 Current working directory: {os.getcwd()}")
        
        if not os.path.exists(filepath):
            logger.error(f"❌ Video file not found: {filepath}")
            logger.error(f"   Absolute path: {os.path.abspath(filepath)}")
            logger.error(f"   CWD: {os.getcwd()}")
            return jsonify({
                'success': False,
                'error': f'Video file not found: {filepath}'
            }), 404
        
        # Validate it's a file
        if not os.path.isfile(filepath):
            logger.error(f"❌ Path is not a file: {filepath}")
            return jsonify({
                'success': False,
                'error': f'Path is not a file: {filepath}'
            }), 400
        
        logger.info("✅ File exists and is valid")
        logger.info(f"📊 File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
        
        # Stop existing processing if running
        global camera_system, current_session_id, video_processing_active
        if camera_system and camera_system.running:
            logger.info("🛑 Stopping previous session...")
            camera_system.stop()
            time.sleep(0.5)
        
        # Create NEW video processing session
        new_session_id = generate_session_id("video")
        current_session_id = new_session_id
        video_processing_active = True
        
        logger.info(f"🆕 Creating new video processing session: {new_session_id}")
        logger.info(f"📹 Video source: {filepath}")
        
        # Create camera system with video file
        camera_system = CameraSystem(video_source=filepath, session_id=new_session_id, is_video_file=True)
        
        success = camera_system.start()
        
        if success:
            logger.info("=" * 60)
            logger.info(f"✅ Video processing started: {camera_system.session_id}")
            logger.info(f"📁 Session directory: {camera_system.session_dir}")
            logger.info(f"🎞️  Total frames: {camera_system.total_frames}")
            logger.info("=" * 60)
            
            return jsonify({
                'success': True,
                'message': 'Video processing started',
                'session_id': camera_system.session_id,
                'session_dir': camera_system.session_dir,
                'total_frames': camera_system.total_frames,
                'source_type': 'video',
                'video_path': filepath
            })
        else:
            error_msg = camera_system.processing_error or 'Failed to start video processing'
            logger.error(f"❌ Failed to start: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
            
    except Exception as e:
        logger.error(f"❌ Video processing error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/video/stop', methods=['POST'])
def stop_video():
    """Stop video processing"""
    try:
        global camera_system, video_processing_active
        
        if camera_system is None:
            return jsonify({
                'success': True,
                'message': 'No video processing active'
            })
        
        success = camera_system.stop()
        video_processing_active = False
        
        return jsonify({
            'success': True,
            'message': 'Video processing stopped'
        })
        
    except Exception as e:
        logger.error(f"Stop video error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera processing"""
    try:
        global camera_system
        
        if camera_system is None:
            return jsonify({
                'success': True,
                'message': 'Camera not running'
            })
        
        success = camera_system.stop()
        
        return jsonify({
            'success': True,
            'message': 'Camera stopped'
        })
        
    except Exception as e:
        logger.error(f"Stop camera error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    """Get camera status"""
    try:
        global camera_system
        
        if camera_system is None:
            return jsonify({
                'success': True,
                'status': {
                    'running': False,
                    'fps': 0,
                    'active_tracks': 0,
                    'total_students': 0,
                    'total_images': 0
                }
            })
        
        stats = camera_system.get_stats()
        
        return jsonify({
            'success': True,
            'status': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all available sessions"""
    try:
        sessions_dir = 'sessions'
        
        if not os.path.exists(sessions_dir):
            return jsonify({
                'success': True,
                'sessions': [],
                'current_session': current_session_id
            })
        
        sessions = []
        for session_folder in sorted(os.listdir(sessions_dir)):
            session_path = os.path.join(sessions_dir, session_folder)
            if os.path.isdir(session_path) and session_folder.startswith('session_'):
                # Count students and images
                student_folders = [
                    d for d in os.listdir(session_path)
                    if os.path.isdir(os.path.join(session_path, d))
                    and d.startswith('student_')
                ]
                
                total_images = 0
                for folder in student_folders:
                    folder_path = os.path.join(session_path, folder)
                    images = [
                        f for f in os.listdir(folder_path)
                        if f.endswith(('.jpg', '.jpeg', '.png'))
                    ]
                    total_images += len(images)
                
                sessions.append({
                    'session_id': session_folder,
                    'students': len(student_folders),
                    'images': total_images,
                    'is_current': session_folder == current_session_id
                })
        
        return jsonify({
            'success': True,
            'sessions': sessions,
            'current_session': current_session_id
        })
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_classroom():
    """
    Analyze all students - SAME AS EXISTING
    """
    try:
        logger.info("=" * 60)
        logger.info("🔍 Starting classroom analysis...")
        logger.info("=" * 60)
        
        engine = get_analytics_engine()
        
        # Analyze all students
        all_analytics = engine.analyze_all_students()
        logger.info(f"✅ Analyzed {len(all_analytics)} students")
        
        # Compute summary
        summary = engine.compute_classroom_summary(all_analytics)
        
        response = {
            'success': True,
            'summary': summary,
            'students': all_analytics,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("✅ Analysis complete")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/student/<student_id>', methods=['GET'])
def get_student_details(student_id):
    """Get specific student details - SAME AS EXISTING"""
    try:
        engine = get_analytics_engine()
        student_folder = os.path.join(engine.students_dir, student_id)
        
        if not os.path.exists(student_folder):
            return jsonify({
                'success': False,
                'error': f'Student {student_id} not found'
            }), 404
        
        analytics = engine.analyze_student_folder(student_folder)
        
        if analytics is None:
            return jsonify({
                'success': False,
                'error': f'No valid data for {student_id}'
            }), 404
        
        return jsonify({
            'success': True,
            'student': analytics
        })
    
    except Exception as e:
        logger.error(f"Failed to get student details: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/images/<path:filename>', methods=['GET'])
def serve_image(filename):
    """Serve student images from current session or sessions directory"""
    try:
        safe_path = os.path.normpath(filename)
        if '..' in safe_path:
            return jsonify({'error': 'Invalid path'}), 400
        
        # Check if path starts with 'sessions/' (new format)
        if safe_path.startswith('sessions/'):
            # New session-based path
            directory = os.path.dirname(safe_path)
            filename = os.path.basename(safe_path)
        else:
            # Legacy path format
            directory = os.path.dirname(os.path.join('students', safe_path))
            filename = os.path.basename(safe_path)
        
        return send_from_directory(directory, filename)
    
    except Exception as e:
        logger.error(f"Failed to serve image: {e}")
        return jsonify({'error': str(e)}), 404


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics from CURRENT SESSION ONLY"""
    try:
        global camera_system, current_session_id
        
        # Use current session directory if available
        if camera_system and current_session_id:
            students_dir = get_session_dir(current_session_id)
        else:
            # Fallback to default students directory
            students_dir = 'students'
        
        if not os.path.exists(students_dir):
            return jsonify({
                'success': True,
                'total_students': 0,
                'total_images': 0,
                'session_id': current_session_id
            })
        
        student_folders = [
            d for d in os.listdir(students_dir)
            if os.path.isdir(os.path.join(students_dir, d))
            and d.startswith('student_')
        ]
        
        total_images = 0
        for folder in student_folders:
            folder_path = os.path.join(students_dir, folder)
            images = [
                f for f in os.listdir(folder_path)
                if f.endswith(('.jpg', '.jpeg', '.png'))
            ]
            total_images += len(images)
        
        return jsonify({
            'success': True,
            'total_students': len(student_folders),
            'total_images': total_images,
            'session_id': current_session_id
        })
    
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EduSence AI - Integrated Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8080, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🎓 EduSence AI - Integrated Server")
    logger.info("=" * 60)
    logger.info(f"🌐 Server: http://{args.host}:{args.port}")
    logger.info(f"📹 Video Stream: http://{args.host}:{args.port}/api/video_feed")
    logger.info("=" * 60)
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
