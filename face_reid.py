"""
Face Re-Identification (ReID) module using InsightFace
Maintains stable student IDs across the entire session
"""

import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import time
import os


class FaceReID:
    """
    Face Re-Identification system for maintaining stable student IDs
    Uses InsightFace embeddings and cosine similarity matching
    """
    
    def __init__(self, similarity_threshold=0.6, embedding_size=512):
        """
        Initialize Face ReID system
        
        Args:
            similarity_threshold: Minimum cosine similarity to match faces (0.6 = 60%)
            embedding_size: Size of face embedding vector
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_size = embedding_size
        
        # Storage for student embeddings
        self.student_embeddings = {}  # {student_id: [embedding1, embedding2, ...]}
        self.student_last_seen = {}   # {student_id: timestamp}
        
        # Track ID to Student ID mapping
        self.track_to_student = {}    # {track_id: student_id}
        
        # Tracking statistics
        self.embedding_count = 0
        self.match_count = 0
        self.new_student_count = 0
        
        # Performance optimization
        self.embedding_cache = {}     # {track_id: (embedding, timestamp)}
        self.cache_duration = 5.0     # Cache embeddings for 5 seconds
        
        # Initialize InsightFace model
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize InsightFace model for face recognition
        Uses lightweight model for real-time performance
        """
        try:
            print("🔄 Loading InsightFace model for ReID...")
            
            import insightface
            from insightface.app import FaceAnalysis
            
            # Use lightweight model for speed
            # 'buffalo_l' is accurate but slower
            # 'buffalo_s' is faster and good for classroom
            self.app = FaceAnalysis(
                name='buffalo_s',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Prepare model with target size
            self.app.prepare(ctx_id=0, det_size=(160, 160))
            
            print("✅ InsightFace model loaded successfully")
            print(f"   Similarity threshold: {self.similarity_threshold}")
            print(f"   Using lightweight 'buffalo_s' model for speed")
            
        except Exception as e:
            print(f"⚠️  Failed to load InsightFace: {str(e)}")
            print("   ReID will be disabled. Install with: pip install insightface onnxruntime")
            self.app = None
    
    def reset_session(self):
        """
        Reset session-specific data while preserving persistent identity database
        
        CLEARS (temporary session data):
        - Track ID to Student ID mapping
        - Embedding cache
        - Active track states
        
        PRESERVES (persistent identity data):
        - Student embeddings (for cross-session recognition)
        - Student last seen timestamps
        - Model instance
        
        This allows starting a fresh session while maintaining identity memory
        """
        print("🔄 Resetting ReID session state...")
        
        # Clear temporary session mappings
        self.track_to_student.clear()
        self.embedding_cache.clear()
        
        # Reset session statistics
        self.embedding_count = 0
        self.match_count = 0
        # Note: new_student_count is NOT reset (persistent across sessions)
        
        print("✅ ReID session state reset")
        print(f"   Preserved {len(self.student_embeddings)} student identities")
    
    def _initialize_model(self):
        """
        Initialize InsightFace model for face recognition
        Uses lightweight model for real-time performance
        """
        try:
            print("🔄 Loading InsightFace model for ReID...")
            
            import insightface
            from insightface.app import FaceAnalysis
            
            # Use lightweight model for speed
            # 'buffalo_l' is accurate but slower
            # 'buffalo_s' is faster and good for classroom
            self.app = FaceAnalysis(
                name='buffalo_s',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Prepare model with target size
            self.app.prepare(ctx_id=0, det_size=(160, 160))
            
            print("✅ InsightFace model loaded successfully")
            print(f"   Similarity threshold: {self.similarity_threshold}")
            print(f"   Using lightweight 'buffalo_s' model for speed")
            
        except Exception as e:
            print(f"⚠️  Failed to load InsightFace: {str(e)}")
            print("   ReID will be disabled. Install with: pip install insightface onnxruntime")
            self.app = None
    
    def extract_embedding(self, face_image):
        """
        Extract face embedding from image
        
        Args:
            face_image: Face crop image (BGR format)
            
        Returns:
            Face embedding vector or None if extraction fails
        """
        if self.app is None:
            return None
        
        try:
            # Resize for faster processing (optimization)
            if face_image.shape[0] > 160 or face_image.shape[1] > 160:
                face_image = cv2.resize(face_image, (160, 160))
            
            # Extract faces and embeddings
            faces = self.app.get(face_image)
            
            if len(faces) > 0:
                # Get embedding from first detected face
                embedding = faces[0].embedding
                self.embedding_count += 1
                return embedding
            
            return None
            
        except Exception as e:
            print(f"⚠️  Embedding extraction error: {str(e)}")
            return None
    
    def find_matching_student(self, embedding, exclude_track_id=None):
        """
        Find matching student ID for given embedding
        
        Args:
            embedding: Face embedding to match
            exclude_track_id: Track ID to exclude from matching
            
        Returns:
            Tuple (student_id, similarity_score) or (None, 0.0) if no match
        """
        if embedding is None or len(self.student_embeddings) == 0:
            return None, 0.0
        
        best_match_id = None
        best_similarity = 0.0
        
        # Compare with all stored student embeddings
        for student_id, stored_embeddings in self.student_embeddings.items():
            # Skip if this student is currently tracked by excluded track
            if exclude_track_id and self.track_to_student.get(exclude_track_id) == student_id:
                continue
            
            # Calculate similarity with all embeddings for this student
            for stored_emb in stored_embeddings:
                similarity = cosine_similarity(
                    embedding.reshape(1, -1),
                    stored_emb.reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = student_id
        
        # Return match only if above threshold
        if best_similarity >= self.similarity_threshold:
            self.match_count += 1
            return best_match_id, best_similarity
        
        return None, best_similarity
    
    def register_or_match_face(self, track_id, face_crop, frame_time, force_extract=False):
        """
        Register new face or match with existing student
        
        Args:
            track_id: Current tracking ID
            face_crop: Face image crop
            frame_time: Current frame timestamp
            force_extract: Force embedding extraction even if cached
            
        Returns:
            Tuple (student_id, is_new, similarity_score)
        """
        # Check if track already has assigned student ID
        if track_id in self.track_to_student:
            student_id = self.track_to_student[track_id]
            self.student_last_seen[student_id] = frame_time
            return student_id, False, 1.0
        
        # Check embedding cache (optimization)
        if not force_extract and track_id in self.embedding_cache:
            cached_emb, cached_time = self.embedding_cache[track_id]
            if frame_time - cached_time < self.cache_duration:
                # Use cached embedding
                embedding = cached_emb
            else:
                # Cache expired, extract new
                embedding = self.extract_embedding(face_crop)
                if embedding is not None:
                    self.embedding_cache[track_id] = (embedding, frame_time)
        else:
            # Extract new embedding
            embedding = self.extract_embedding(face_crop)
            if embedding is not None:
                self.embedding_cache[track_id] = (embedding, frame_time)
        
        if embedding is None:
            # Failed to extract embedding, assign temporary ID
            return track_id, True, 0.0
        
        # Try to match with existing students
        matched_student_id, similarity = self.find_matching_student(embedding, exclude_track_id=track_id)
        
        if matched_student_id is not None:
            # Matched with existing student
            print(f"🔄 ReID: Track {track_id} matched to Student {matched_student_id} (similarity: {similarity:.3f})")
            
            # Assign student ID to track
            self.track_to_student[track_id] = matched_student_id
            self.student_last_seen[matched_student_id] = frame_time
            
            # Add embedding to student's collection (for better future matching)
            if len(self.student_embeddings[matched_student_id]) < 5:  # Keep max 5 embeddings
                self.student_embeddings[matched_student_id].append(embedding)
            
            return matched_student_id, False, similarity
        
        else:
            # New student detected
            self.new_student_count += 1
            new_student_id = self.new_student_count
            
            print(f"✨ ReID: New student detected - Student {new_student_id} (Track {track_id})")
            
            # Register new student
            self.student_embeddings[new_student_id] = [embedding]
            self.track_to_student[track_id] = new_student_id
            self.student_last_seen[new_student_id] = frame_time
            
            return new_student_id, True, 0.0
    
    def handle_track_lost(self, track_id):
        """
        Handle when a track is lost
        
        Args:
            track_id: Lost tracking ID
        """
        if track_id in self.track_to_student:
            student_id = self.track_to_student[track_id]
            print(f"⚠️  ReID: Track {track_id} lost (Student {student_id})")
            
            # Remove from active tracking but keep embeddings
            del self.track_to_student[track_id]
        
        # Clear cache for this track
        if track_id in self.embedding_cache:
            del self.embedding_cache[track_id]
    
    def should_extract_embedding(self, track_id, track_age, confidence):
        """
        Determine if embedding should be extracted for this track
        Optimization: Don't extract every frame
        
        Args:
            track_id: Tracking ID
            track_age: Number of frames this track has existed
            confidence: Detection confidence
            
        Returns:
            True if embedding should be extracted
        """
        # Always extract for new tracks
        if track_id not in self.track_to_student:
            return True
        
        # Extract if confidence is low (might be different person)
        if confidence < 0.7:
            return True
        
        # Extract periodically for long-running tracks (every 30 frames)
        if track_age % 30 == 0:
            return True
        
        # Check cache
        if track_id in self.embedding_cache:
            cached_emb, cached_time = self.embedding_cache[track_id]
            current_time = time.time()
            if current_time - cached_time > self.cache_duration:
                return True
        
        return False
    
    def get_statistics(self):
        """
        Get ReID statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_students': len(self.student_embeddings),
            'active_tracks': len(self.track_to_student),
            'embeddings_extracted': self.embedding_count,
            'successful_matches': self.match_count,
            'new_students_detected': self.new_student_count,
            'cached_embeddings': len(self.embedding_cache)
        }
    
    def cleanup_old_tracks(self, current_time, timeout=30.0):
        """
        Clean up tracks that haven't been seen recently
        
        Args:
            current_time: Current timestamp
            timeout: Seconds before considering a student gone
        """
        tracks_to_remove = []
        
        for track_id, student_id in self.track_to_student.items():
            last_seen = self.student_last_seen.get(student_id, 0)
            if current_time - last_seen > timeout:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self.handle_track_lost(track_id)
