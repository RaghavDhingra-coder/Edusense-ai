"""
Enhanced Face ReID with Student Recognition
Combines tracking-based ReID with registered student recognition
"""

import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging

logger = logging.getLogger(__name__)


class FaceReIDWithRecognition:
    """
    Enhanced ReID system that recognizes registered students
    Falls back to tracking-based IDs for unknown persons
    """
    
    def __init__(self, student_registry, similarity_threshold=0.6, embedding_size=512):
        """
        Initialize ReID with recognition
        
        Args:
            student_registry: StudentRegistry instance
            similarity_threshold: Threshold for tracking-based matching
            embedding_size: Size of face embedding vector
        """
        self.registry = student_registry
        self.similarity_threshold = similarity_threshold
        self.embedding_size = embedding_size
        
        # Track ID to recognized identity mapping
        self.track_to_identity = {}  # {track_id: {'type': 'registered'/'unknown', 'id': ..., 'name': ...}}
        
        # Recognition cache (avoid re-recognition every frame)
        self.recognition_cache = {}  # {track_id: {'identity': ..., 'timestamp': ..., 'confidence': ...}}
        self.cache_duration = 10.0  # Cache recognition for 10 seconds (increased for stability)
        
        # Recognition frequency control - OPTIMIZED FOR SPEED
        self.recognition_interval = 3  # Run recognition every 3 frames for unknown persons (reduced from 10)
        self.track_frame_count = {}  # {track_id: frame_count}
        
        # Aggressive first-frame recognition for new tracks
        self.aggressive_frames = [0, 1, 2, 3, 5, 7, 10]  # Run recognition on these frames for new tracks
        
        # Unknown person counter
        self.unknown_counter = 0
        
        # Tracking statistics
        self.embedding_count = 0
        self.recognition_count = 0
        self.cache_hits = 0
        
        # InsightFace model (shared with registry)
        self.face_app = None
        
        logger.info("🎯 ReID with Recognition initialized")
    
    def set_face_app(self, face_app):
        """Set InsightFace app"""
        self.face_app = face_app
        self.registry.set_face_app(face_app)
        logger.info("✅ InsightFace app connected")
    
    def register_or_match_face(self, track_id, face_crop, frame_time, force_extract=False):
        """
        Register or match face with recognition support
        OPTIMIZED: Aggressive first-frame recognition for instant identification
        
        Args:
            track_id: Current tracking ID
            face_crop: Face image crop
            frame_time: Current frame timestamp
            force_extract: Force recognition even if cached
            
        Returns:
            Tuple (identity_id, identity_name, is_new, confidence, is_registered)
        """
        start_time = time.time()
        
        # Check if track already has recognized identity
        if track_id in self.track_to_identity and not force_extract:
            identity = self.track_to_identity[track_id]
            
            # Check if cache is still valid
            if track_id in self.recognition_cache:
                cache_entry = self.recognition_cache[track_id]
                if frame_time - cache_entry['timestamp'] < self.cache_duration:
                    self.cache_hits += 1
                    elapsed = (time.time() - start_time) * 1000
                    logger.debug(f"⚡ Cache hit for {identity['name']} ({elapsed:.1f}ms)")
                    return (
                        identity['id'],
                        identity['name'],
                        False,
                        cache_entry['confidence'],
                        identity['type'] == 'registered'
                    )
        
        # Determine if we should run recognition
        should_recognize = self._should_run_recognition(track_id, force_extract)
        
        if should_recognize:
            # Run recognition with timing
            recog_start = time.time()
            result = self._recognize_face(track_id, face_crop, frame_time)
            recog_time = (time.time() - recog_start) * 1000
            
            if result:
                total_time = (time.time() - start_time) * 1000
                logger.info(f"⚡ Recognition: {result[1]} in {total_time:.1f}ms (recog: {recog_time:.1f}ms)")
                return result
        
        # Fallback: use tracking-based ID for unknown persons
        if track_id not in self.track_to_identity:
            self.unknown_counter += 1
            identity = {
                'type': 'unknown',
                'id': f'unknown_{self.unknown_counter}',
                'name': f'Unknown {self.unknown_counter}'
            }
            self.track_to_identity[track_id] = identity
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"❓ Unknown person: {identity['name']} (Track {track_id}, {elapsed:.1f}ms)")
            
            return (identity['id'], identity['name'], True, 0.0, False)
        
        # Return existing identity
        identity = self.track_to_identity[track_id]
        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"⚡ Existing identity: {identity['name']} ({elapsed:.1f}ms)")
        return (identity['id'], identity['name'], False, 0.0, identity['type'] == 'registered')
    
    def _should_run_recognition(self, track_id, force=False):
        """
        Determine if recognition should run for this track
        OPTIMIZED: Aggressive recognition for new tracks, periodic for unknown
        """
        if force:
            return True
        
        # New track - initialize frame count
        if track_id not in self.track_frame_count:
            self.track_frame_count[track_id] = 0
            return True  # Always recognize on first frame
        
        # Get current frame count
        frame_count = self.track_frame_count[track_id]
        
        # Increment frame count
        self.track_frame_count[track_id] += 1
        
        # Check if track is already recognized as registered student
        if track_id in self.track_to_identity:
            identity = self.track_to_identity[track_id]
            if identity['type'] == 'registered':
                # Already recognized - use cache, no need to re-recognize frequently
                return False
        
        # For unknown persons or unrecognized tracks:
        # Aggressive recognition on early frames
        if frame_count in self.aggressive_frames:
            return True
        
        # After aggressive phase, run recognition every N frames
        if frame_count > max(self.aggressive_frames):
            if frame_count % self.recognition_interval == 0:
                return True
        
        return False
    
    def _recognize_face(self, track_id, face_crop, frame_time):
        """
        Run face recognition against registered students
        OPTIMIZED: Added detailed timing logs
        
        Returns:
            Tuple (identity_id, identity_name, is_new, confidence, is_registered) or None
        """
        try:
            self.recognition_count += 1
            start_time = time.time()
            
            # Recognize face
            recognition_result = self.registry.recognize_face(face_crop)
            recog_time = (time.time() - start_time) * 1000
            
            if recognition_result:
                # Recognized as registered student
                student_id = recognition_result['student_id']
                student_name = recognition_result['student_name']
                confidence = recognition_result['confidence']
                
                # Check if this is a new recognition for this track
                is_new = track_id not in self.track_to_identity
                
                # Update identity mapping
                identity = {
                    'type': 'registered',
                    'id': student_id,
                    'name': student_name
                }
                self.track_to_identity[track_id] = identity
                
                # Update cache
                self.recognition_cache[track_id] = {
                    'identity': identity,
                    'timestamp': frame_time,
                    'confidence': confidence
                }
                
                if is_new:
                    logger.info(f"✅ Recognized: {student_name} (Track {track_id}, {confidence:.3f}, {recog_time:.1f}ms)")
                else:
                    logger.debug(f"✅ Re-confirmed: {student_name} (Track {track_id}, {confidence:.3f}, {recog_time:.1f}ms)")
                
                return (student_id, student_name, is_new, confidence, True)
            else:
                logger.debug(f"❓ No match found (Track {track_id}, {recog_time:.1f}ms)")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Recognition error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def handle_track_lost(self, track_id):
        """Handle when a track is lost"""
        if track_id in self.track_to_identity:
            identity = self.track_to_identity[track_id]
            logger.debug(f"⚠️  Track lost: {identity['name']} (Track {track_id})")
            
            # Keep identity mapping for potential recovery
            # Don't delete immediately
        
        # Clear frame count
        if track_id in self.track_frame_count:
            del self.track_frame_count[track_id]
    
    def cleanup_old_tracks(self, current_time, timeout=30.0):
        """Clean up old track data"""
        tracks_to_remove = []
        
        for track_id, cache_entry in self.recognition_cache.items():
            if current_time - cache_entry['timestamp'] > timeout:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if track_id in self.recognition_cache:
                del self.recognition_cache[track_id]
            if track_id in self.track_to_identity:
                del self.track_to_identity[track_id]
            if track_id in self.track_frame_count:
                del self.track_frame_count[track_id]
    
    def reset_session(self):
        """Reset session state while preserving registry"""
        logger.info("🔄 Resetting ReID session state...")
        
        self.track_to_identity.clear()
        self.recognition_cache.clear()
        self.track_frame_count.clear()
        self.unknown_counter = 0
        
        # Reset statistics
        self.embedding_count = 0
        self.recognition_count = 0
        self.cache_hits = 0
        
        logger.info("✅ ReID session state reset")
        logger.info(f"   Registered students preserved: {len(self.registry.students)}")
    
    def get_statistics(self):
        """Get ReID statistics"""
        return {
            'registered_students': len(self.registry.students),
            'active_tracks': len(self.track_to_identity),
            'recognized_tracks': sum(1 for i in self.track_to_identity.values() if i['type'] == 'registered'),
            'unknown_tracks': sum(1 for i in self.track_to_identity.values() if i['type'] == 'unknown'),
            'recognition_count': self.recognition_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': f"{(self.cache_hits / max(1, self.recognition_count + self.cache_hits) * 100):.1f}%"
        }
