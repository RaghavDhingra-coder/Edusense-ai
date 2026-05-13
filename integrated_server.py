"""
Integrated Flask Server for EduSence AI
Combines camera streaming, detection, tracking, and analytics in one unified dashboard
"""

from flask import Flask, jsonify, request, send_from_directory, Response, session as flask_session
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
import uuid

# Import existing components - NO CHANGES TO LOGIC
from face_detector import FaceDetector
from image_manager import ImageManager
from face_reid import FaceReID
from hybrid_attentiveness import HybridAttentivenessAnalyzer
from student_registry import StudentRegistry
from face_reid_recognition import FaceReIDWithRecognition
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, static_folder='frontend', static_url_path='')

# Secret key required for Flask cookie-based sessions
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False   # set True if serving over HTTPS

# Configure CORS - allow all origins so the deployed VM frontend can reach the API
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'}
MAX_UPLOAD_SIZE = 500 * 1024 * 1024  # 500MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Per-user session store ────────────────────────────────────────────────────
# Replaces the old shared globals:
#   analytics_engine, camera_system, current_session_id, video_processing_active
#
# Structure:
#   user_sessions[user_id] = {
#       'camera_system':          CameraSystem | None,
#       'current_session_id':     str | None,
#       'video_processing_active': bool,
#       'last_active':            float,   # time.time() — for cleanup
#   }
user_sessions: dict = {}
user_sessions_lock = threading.Lock()

# ── Legacy globals removed ────────────────────────────────────────────────────
# analytics_engine        → created on-demand inside get_analytics_engine()
# camera_system           → user_sessions[uid]['camera_system']
# current_session_id      → user_sessions[uid]['current_session_id']
# video_processing_active → user_sessions[uid]['video_processing_active']
# ─────────────────────────────────────────────────────────────────────────────

camera_lock = threading.Lock()   # kept for CameraSystem.start() / stop() internals

# Student registry (global, persistent across sessions)
student_registry = StudentRegistry(registry_dir='registered_students')
logger.info(f"📚 Loaded {len(student_registry.students)} registered students")

# Initialize InsightFace for the registry
try:
    import insightface
    from insightface.app import FaceAnalysis
    
    logger.info("🔄 Initializing InsightFace for student registry...")
    face_app = FaceAnalysis(
        name='buffalo_s',
        providers=['CPUExecutionProvider']
    )
    face_app.prepare(ctx_id=0, det_size=(160, 160))
    
    # Connect to registry
    student_registry.set_face_app(face_app)
    logger.info("✅ InsightFace initialized for student registry")
    
except Exception as e:
    logger.error(f"⚠️  InsightFace initialization failed: {e}")
    logger.warning("   Student registration will not work without InsightFace")
    logger.warning("   Install with: pip3 install insightface onnxruntime")


def generate_session_id(prefix="session"):
    """Generate unique session ID"""
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_session_dir(session_id):
    """Get directory for specific session"""
    return os.path.join('sessions', session_id)


# ── Per-user session helpers ──────────────────────────────────────────────────

def get_user_id() -> str:
    """
    Return the stable user-id for the current HTTP request.
    Creates one and stores it in the Flask cookie session if not present.
    """
    if 'user_id' not in flask_session:
        flask_session['user_id'] = str(uuid.uuid4())
        flask_session.permanent = True
    return flask_session['user_id']


def get_user_state(user_id: str) -> dict:
    """
    Return (and lazily create) the runtime state dict for a user.
    Thread-safe.
    """
    with user_sessions_lock:
        if user_id not in user_sessions:
            user_sessions[user_id] = {
                'camera_system': None,
                'current_session_id': None,
                'video_processing_active': False,
                'last_active': time.time(),
            }
        else:
            user_sessions[user_id]['last_active'] = time.time()
        return user_sessions[user_id]


def get_user_upload_dir(user_id: str) -> str:
    """Return (and create) an upload directory isolated to this user."""
    path = os.path.join(UPLOAD_FOLDER, user_id)
    os.makedirs(path, exist_ok=True)
    return path


def cleanup_idle_sessions(max_idle_seconds: int = 3600):
    """
    Remove state for users who have been idle longer than max_idle_seconds.
    Stops any running CameraSystem and removes upload/session directories.
    Called opportunistically on each camera/upload request.
    """
    now = time.time()
    with user_sessions_lock:
        stale = [uid for uid, s in user_sessions.items()
                 if now - s['last_active'] > max_idle_seconds]
    for uid in stale:
        try:
            with user_sessions_lock:
                state = user_sessions.pop(uid, None)
            if state and state['camera_system'] and state['camera_system'].running:
                state['camera_system'].stop()
            # Remove per-user upload folder
            upload_dir = os.path.join(UPLOAD_FOLDER, uid)
            if os.path.exists(upload_dir):
                shutil.rmtree(upload_dir, ignore_errors=True)
            logger.info(f"🧹 Cleaned up idle session for user {uid[:8]}…")
        except Exception as e:
            logger.warning(f"Cleanup error for {uid[:8]}: {e}")

# ─────────────────────────────────────────────────────────────────────────────


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
    Supports:
      - webcam (video_source=0, is_video_file=False, browser_mode=False)
      - video files (video_source=path, is_video_file=True, browser_mode=False)
      - browser webcam (browser_mode=True) — frames pushed via process_browser_frame()
    """
    
    def __init__(self, video_source=0, session_id=None, is_video_file=False, browser_mode=False):
        """Initialize camera system with existing components"""
        self.video_source = video_source
        self.is_video_file = is_video_file
        self.browser_mode = browser_mode          # NEW: browser sends frames
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
            
            # ReID system with Recognition - USE ENHANCED VERSION
            # NOTE: ReID will use persistent identity database if it exists
            if config.ENABLE_REID:
                logger.info("🔄 Initializing Face Re-Identification with Recognition...")
                self.reid_system = FaceReIDWithRecognition(
                    student_registry=student_registry,
                    similarity_threshold=config.REID_SIMILARITY_THRESHOLD,
                    embedding_size=512
                )
                # Reset ReID session state (but keep persistent identity DB)
                self.reid_system.reset_session()
                logger.info(f"✅ Recognition enabled with {len(student_registry.students)} registered students")
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
                # ── BROWSER MODE: no local VideoCapture needed ──────────────
                if self.browser_mode:
                    logger.info("[CAMERA] Browser-webcam mode: skipping cv2.VideoCapture")
                    self.running = True
                    self.frame_number = 0
                    self.active_tracks = set()
                    self.processing_complete = False
                    self.processing_error = None
                    logger.info("[CAMERA] ✅ Browser-webcam session ready")
                    logger.info("=" * 60)
                    return True
                # ────────────────────────────────────────────────────────────

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

    def process_browser_frame(self, frame):
        """
        Process a single frame received from the browser webcam.
        Runs the EXACT SAME AI pipeline as _process_frame().
        Returns the annotated frame as a JPEG bytes object.
        Called from the /api/process_frame endpoint.
        """
        if not self.running:
            return None
        
        # Run existing AI pipeline (unchanged)
        processed = self._process_frame(frame)
        
        # Store as latest frame so MJPEG stream also shows it
        with self.frame_lock:
            self.latest_frame = processed.copy()
        
        self._update_fps()
        
        # Encode to JPEG and return
        ret, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            return None
        return buffer.tobytes()
    
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
            
            # Apply ReID with Recognition
            if self.reid_system and config.ENABLE_REID:
                # FaceReIDWithRecognition returns: (identity_id, identity_name, is_new, confidence, is_registered)
                identity_id, identity_name, is_new, confidence_score, is_registered = self.reid_system.register_or_match_face(
                    track_id, face_crop, current_time, force_extract=False
                )
                
                # Handle recognition result
                if is_registered:
                    # Registered student recognized
                    student_id = identity_id
                    student_name = identity_name
                    
                    if is_new:
                        logger.info(f"✅ Recognized: {student_name} (Track {track_id})")
                    
                    # ONLY save face image for registered students
                    self.image_manager.save_face_image(frame, bbox, student_id, confidence)
                    
                    # Store for display with name
                    reid_detections.append((x1, y1, x2, y2, student_id, confidence, student_name, True))
                    
                else:
                    # Unknown person - SKIP analytics and storage
                    student_name = identity_name  # "Unknown X"
                    
                    # Show in video feed but DON'T save or analyze
                    # No call to save_face_image() for unknown persons
                    reid_detections.append((x1, y1, x2, y2, track_id, confidence, student_name, False))
                    
            else:
                # No ReID - use track ID
                student_id = track_id
                self.image_manager.save_face_image(frame, bbox, student_id, confidence)
                reid_detections.append((x1, y1, x2, y2, student_id, confidence, f"Student {student_id}", True))
        
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
        """Draw bounding boxes and labels with student names"""
        for detection in detections:
            if len(detection) == 8:
                # New format with recognition
                x1, y1, x2, y2, student_id, confidence, student_name, is_registered = detection
            else:
                # Old format fallback
                x1, y1, x2, y2, student_id, confidence = detection
                student_name = f"Student {student_id}"
                is_registered = True
            
            # Color coding: Green for registered, Orange for unknown
            if is_registered:
                color = (0, 255, 0)  # Green
                label = student_name
            else:
                color = (0, 165, 255)  # Orange
                label = "Unknown Person"
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw label
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


def get_analytics_engine(user_id: str):
    """Create a fresh analytics engine scoped to the calling user's session."""
    state = get_user_state(user_id)
    camera_sys = state['camera_system']
    current_sid = state['current_session_id']

    # Use the user's active session directory if available
    if camera_sys and current_sid:
        students_dir = get_session_dir(current_sid)
    else:
        students_dir = 'students'

    logger.info(f"📊 Creating analytics engine for user {user_id[:8]}… → {students_dir}")
    return HybridAttentivenessAnalyzer(
        students_dir=students_dir,
        focus_threshold=0.65,
        yaw_threshold=30.0,
        pitch_threshold=30.0,
        roll_threshold=35.0,
        consecutive_distraction_threshold=2
    )


def generate_frames(user_id: str):
    """Generate MJPEG frames for a specific user's camera session."""
    frame_count = 0
    logger.info(f"[MJPEG] Stream started for user {user_id[:8]}…")

    while True:
        state = get_user_state(user_id)
        cam_sys = state['camera_system']

        frame = None
        if cam_sys is not None:
            frame = cam_sys.get_frame()

        if frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera Not Started", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            frame = placeholder
        else:
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"[MJPEG] user {user_id[:8]}… frame {frame_count}: {frame.shape}")

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

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
    """Video streaming route — per-user stream"""
    user_id = get_user_id()
    return Response(generate_frames(user_id),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """
    Receive a single frame from the browser webcam, run it through the
    existing AI pipeline, and return the annotated frame as a JPEG.

    Expects JSON: { "frame": "<base64-encoded JPEG>" }
    Returns JSON: { "success": true, "frame": "<base64-encoded JPEG>" }
                  or { "success": false, "error": "..." }
    """
    import base64

    try:
        user_id = get_user_id()
        state = get_user_state(user_id)
        camera_system = state['camera_system']

        if camera_system is None or not camera_system.running:
            return jsonify({'success': False, 'error': 'Camera session not started'}), 400

        data = request.get_json(silent=True)
        if not data or 'frame' not in data:
            return jsonify({'success': False, 'error': 'No frame data provided'}), 400

        # Decode base64 → OpenCV frame
        b64_data = data['frame']
        if ',' in b64_data:                        # strip data-URL prefix if present
            b64_data = b64_data.split(',', 1)[1]

        img_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'success': False, 'error': 'Failed to decode frame'}), 400

        # Run existing AI pipeline (unchanged)
        jpeg_bytes = camera_system.process_browser_frame(frame)

        if jpeg_bytes is None:
            return jsonify({'success': False, 'error': 'Frame processing failed'}), 500

        # Return processed frame as base64
        result_b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
        return jsonify({'success': True, 'frame': result_b64})

    except Exception as e:
        logger.error(f"[process_frame] ❌ {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera processing with NEW SESSION — isolated per user"""
    try:
        user_id = get_user_id()
        state = get_user_state(user_id)
        cleanup_idle_sessions()

        logger.info("=" * 60)
        logger.info(f"🎥 Starting new camera session for user {user_id[:8]}…")
        logger.info("=" * 60)

        # Stop existing camera for this user if running
        if state['camera_system'] and state['camera_system'].running:
            logger.info("🛑 Stopping previous session for this user...")
            state['camera_system'].stop()
            time.sleep(0.5)

        # Create NEW camera system with fresh session for this user
        new_session_id = generate_session_id("webcam")
        state['current_session_id'] = new_session_id
        state['video_processing_active'] = False

        logger.info(f"🆕 Creating new camera system for session: {new_session_id}")
        cam_sys = CameraSystem(
            video_source=0,
            session_id=new_session_id,
            is_video_file=False,
            browser_mode=True
        )
        state['camera_system'] = cam_sys

        success = cam_sys.start()

        if success:
            logger.info(f"✅ New session started: {cam_sys.session_id}")
            return jsonify({
                'success': True,
                'message': 'Camera started successfully',
                'session_id': cam_sys.session_id,
                'session_dir': cam_sys.session_dir,
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
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/video/upload', methods=['POST'])
def upload_video():
    """Upload video file for processing — stored in per-user directory"""
    try:
        user_id = get_user_id()

        logger.info("=" * 60)
        logger.info(f"📤 Video upload from user {user_id[:8]}…")
        logger.info("=" * 60)

        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400

        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        # Secure filename with timestamp
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"

        # Save into per-user upload directory
        user_upload_dir = get_user_upload_dir(user_id)
        filepath = os.path.abspath(os.path.join(user_upload_dir, filename))

        # Security: ensure path stays inside the user's upload folder
        if not filepath.startswith(os.path.abspath(user_upload_dir)):
            return jsonify({'success': False, 'error': 'Invalid file path'}), 400

        file.save(filepath)

        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': 'Failed to save file'}), 500

        file_size = os.path.getsize(filepath)
        file_size_mb = file_size / (1024 * 1024)

        if file_size > MAX_UPLOAD_SIZE:
            os.remove(filepath)
            return jsonify({
                'success': False,
                'error': f'File too large: {file_size_mb:.2f} MB'
            }), 400

        # Validate it's actually a video
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                cap.release()
                os.remove(filepath)
                return jsonify({'success': False, 'error': 'File is not a valid video'}), 400
            cap.release()
        except Exception as e:
            os.remove(filepath)
            return jsonify({'success': False, 'error': 'Invalid video file'}), 400

        logger.info(f"✅ Video uploaded: {filename} ({file_size_mb:.2f} MB) → {filepath}")

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
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/video/process', methods=['POST'])
def process_video():
    """Start processing uploaded video — isolated per user"""
    try:
        user_id = get_user_id()
        state = get_user_state(user_id)

        logger.info("=" * 60)
        logger.info(f"🎬 VIDEO PROCESSING REQUEST from user {user_id[:8]}…")
        logger.info("=" * 60)

        data = request.get_json()
        if not data or 'filepath' not in data:
            return jsonify({'success': False, 'error': 'No filepath provided'}), 400

        filepath = data['filepath']

        if not filepath or not isinstance(filepath, str) or filepath.strip() == '':
            return jsonify({'success': False, 'error': 'Invalid filepath'}), 400

        if not os.path.exists(filepath):
            return jsonify({'success': False, 'error': f'Video file not found: {filepath}'}), 404

        if not os.path.isfile(filepath):
            return jsonify({'success': False, 'error': f'Path is not a file: {filepath}'}), 400

        # Security: uploaded file must live inside this user's upload directory
        user_upload_dir = os.path.abspath(get_user_upload_dir(user_id))
        if not os.path.abspath(filepath).startswith(user_upload_dir):
            return jsonify({'success': False, 'error': 'Access denied: file not in your upload directory'}), 403

        # Stop existing processing for this user
        if state['camera_system'] and state['camera_system'].running:
            logger.info("🛑 Stopping previous session for this user...")
            state['camera_system'].stop()
            time.sleep(0.5)

        new_session_id = generate_session_id("video")
        state['current_session_id'] = new_session_id
        state['video_processing_active'] = True

        logger.info(f"🆕 Video session {new_session_id} for user {user_id[:8]}…")
        cam_sys = CameraSystem(video_source=filepath, session_id=new_session_id, is_video_file=True)
        state['camera_system'] = cam_sys

        success = cam_sys.start()

        if success:
            return jsonify({
                'success': True,
                'message': 'Video processing started',
                'session_id': cam_sys.session_id,
                'session_dir': cam_sys.session_dir,
                'total_frames': cam_sys.total_frames,
                'source_type': 'video',
                'video_path': filepath
            })
        else:
            error_msg = cam_sys.processing_error or 'Failed to start video processing'
            return jsonify({'success': False, 'error': error_msg}), 500

    except Exception as e:
        logger.error(f"❌ Video processing error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/video/stop', methods=['POST'])
def stop_video():
    """Stop video processing for this user"""
    try:
        user_id = get_user_id()
        state = get_user_state(user_id)

        if state['camera_system'] is None:
            return jsonify({'success': True, 'message': 'No video processing active'})

        state['camera_system'].stop()
        state['video_processing_active'] = False

        return jsonify({'success': True, 'message': 'Video processing stopped'})

    except Exception as e:
        logger.error(f"Stop video error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera processing for this user"""
    try:
        user_id = get_user_id()
        state = get_user_state(user_id)

        if state['camera_system'] is None:
            return jsonify({'success': True, 'message': 'Camera not running'})

        state['camera_system'].stop()

        return jsonify({'success': True, 'message': 'Camera stopped'})

    except Exception as e:
        logger.error(f"Stop camera error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    """Get camera status for this user"""
    try:
        user_id = get_user_id()
        state = get_user_state(user_id)
        cam_sys = state['camera_system']

        if cam_sys is None:
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

        return jsonify({'success': True, 'status': cam_sys.get_stats()})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List sessions belonging to the current user"""
    try:
        user_id = get_user_id()
        state = get_user_state(user_id)
        current_sid = state['current_session_id']
        sessions_dir = 'sessions'

        if not os.path.exists(sessions_dir):
            return jsonify({'success': True, 'sessions': [], 'current_session': current_sid})

        sessions = []
        for session_folder in sorted(os.listdir(sessions_dir)):
            session_path = os.path.join(sessions_dir, session_folder)
            if not os.path.isdir(session_path):
                continue

            student_folders = [
                d for d in os.listdir(session_path)
                if os.path.isdir(os.path.join(session_path, d)) and d.startswith('student_')
            ]
            total_images = sum(
                len([f for f in os.listdir(os.path.join(session_path, sf))
                     if f.endswith(('.jpg', '.jpeg', '.png'))])
                for sf in student_folders
            )
            sessions.append({
                'session_id': session_folder,
                'students': len(student_folders),
                'images': total_images,
                'is_current': session_folder == current_sid
            })

        return jsonify({'success': True, 'sessions': sessions, 'current_session': current_sid})

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_classroom():
    """Analyze all students — scoped to the calling user's session"""
    try:
        user_id = get_user_id()
        logger.info(f"🔍 Analysis requested by user {user_id[:8]}…")

        engine = get_analytics_engine(user_id)

        all_analytics = engine.analyze_all_students()
        logger.info(f"✅ Analyzed {len(all_analytics)} students")

        summary = engine.compute_classroom_summary(all_analytics)

        return jsonify({
            'success': True,
            'summary': summary,
            'students': all_analytics,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/student/<student_id>', methods=['GET'])
def get_student_details(student_id):
    """Get specific student details — scoped to calling user's session"""
    try:
        user_id = get_user_id()
        engine = get_analytics_engine(user_id)
        student_folder = os.path.join(engine.students_dir, student_id)

        if not os.path.exists(student_folder):
            return jsonify({'success': False, 'error': f'Student {student_id} not found'}), 404

        analytics = engine.analyze_student_folder(student_folder)
        if analytics is None:
            return jsonify({'success': False, 'error': f'No valid data for {student_id}'}), 404

        return jsonify({'success': True, 'student': analytics})

    except Exception as e:
        logger.error(f"Failed to get student details: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


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
    """Get statistics from the calling user's current session"""
    try:
        user_id = get_user_id()
        state = get_user_state(user_id)
        current_sid = state['current_session_id']
        cam_sys = state['camera_system']

        if cam_sys and current_sid:
            students_dir = get_session_dir(current_sid)
        else:
            students_dir = 'students'

        if not os.path.exists(students_dir):
            return jsonify({
                'success': True,
                'total_students': 0,
                'total_images': 0,
                'session_id': current_sid
            })

        student_folders = [
            d for d in os.listdir(students_dir)
            if os.path.isdir(os.path.join(students_dir, d)) and d.startswith('student_')
        ]
        total_images = sum(
            len([f for f in os.listdir(os.path.join(students_dir, sf))
                 if f.endswith(('.jpg', '.jpeg', '.png'))])
            for sf in student_folders
        )

        return jsonify({
            'success': True,
            'total_students': len(student_folders),
            'total_images': total_images,
            'session_id': current_sid
        })

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500




# ============================================================================
# STUDENT REGISTRATION ENDPOINTS
# ============================================================================

@app.route('/api/students/register', methods=['POST'])
def register_student():
    """Register a new student with face images"""
    try:
        data = request.get_json()
        
        student_name = data.get('student_name')
        student_id = data.get('student_id')
        face_images_b64 = data.get('face_images', [])
        
        logger.info(f"📝 Registration request: {student_name} ({student_id})")
        logger.info(f"   Images provided: {len(face_images_b64)}")
        
        # Decode base64 images
        face_images = []
        for idx, img_b64 in enumerate(face_images_b64):
            try:
                import base64
                # Remove data:image/jpeg;base64, prefix if present
                if ',' in img_b64:
                    img_b64 = img_b64.split(',')[1]
                
                img_data = base64.b64decode(img_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    face_images.append(img)
                    logger.info(f"   ✅ Image {idx+1} decoded: {img.shape}")
                else:
                    logger.warning(f"   ⚠️  Image {idx+1} decode failed")
            except Exception as e:
                logger.error(f"   ❌ Image {idx+1} error: {e}")
        
        if len(face_images) == 0:
            return jsonify({
                'success': False,
                'error': 'No valid images provided'
            }), 400
        
        # Register student
        result = student_registry.register_student(student_name, student_id, face_images)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Registration error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/students/list', methods=['GET'])
def list_students():
    """Get list of all registered students"""
    try:
        students = student_registry.get_all_students()
        return jsonify({
            'success': True,
            'students': students,
            'count': len(students)
        })
    except Exception as e:
        logger.error(f"❌ List students error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/students/delete/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    """Delete a registered student"""
    try:
        result = student_registry.delete_student(student_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"❌ Delete student error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/students/capture', methods=['POST'])
def capture_faces_from_camera():
    """Capture face images from the calling user's current camera feed"""
    try:
        user_id = get_user_id()
        state = get_user_state(user_id)
        camera_system = state['camera_system']

        if not camera_system or not camera_system.running:
            return jsonify({
                'success': False,
                'error': 'Camera not running. Please start camera first.'
            }), 400

        num_captures = request.json.get('num_captures', 10) if request.json else 10
        captured_images = []

        logger.info(f"📸 Capturing {num_captures} face images for user {user_id[:8]}…")

        for i in range(num_captures):
            frame = camera_system.get_frame()
            if frame is not None:
                detections = camera_system.detector.detect_faces(frame)
                if len(detections) > 0:
                    x1, y1, x2, y2, conf = detections[0]
                    face_crop = camera_system.image_manager.crop_face(frame, (x1, y1, x2, y2))
                    if face_crop is not None:
                        import base64
                        _, buffer = cv2.imencode('.jpg', face_crop)
                        img_b64 = base64.b64encode(buffer).decode('utf-8')
                        captured_images.append(f"data:image/jpeg;base64,{img_b64}")
            time.sleep(0.5)

        logger.info(f"✅ Captured {len(captured_images)} images")
        return jsonify({'success': True, 'images': captured_images, 'count': len(captured_images)})

    except Exception as e:
        logger.error(f"❌ Capture error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


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
