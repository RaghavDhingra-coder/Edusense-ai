"""
Robust Face Re-Identification (ReID) module
Addresses real-world issues:
- Same student getting multiple IDs
- Different students merged into same ID
- Temporal consistency
- Embedding smoothing
- Quality filtering
"""

import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from collections import deque, defaultdict
import time
import os


class RobustFaceReID:
    """
    Production-grade Face Re-Identification system with:
    - Temporal consistency
    - Embedding smoothing
    - Quality filtering
    - Spatial tracking
    - Hybrid matching
    """
    
    def __init__(self, 
                 similarity_threshold=0.55,  # Lowered for better matching
                 spatial_weight=0.3,         # Weight for spatial distance
                 temporal_weight=0.2,        # Weight for temporal continuity
                 embedding_weight=0.5,       # Weight for embedding similarity
                 max_embeddings_per_student=10,
                 cooldown_period=3.0,        # Seconds before creating new ID
                 quality_threshold=30.0):    # Minimum quality score
        """
        Initialize Robust ReID system
        
        Args:
            similarity_threshold: Minimum similarity for matching
            spatial_weight: Weight for spatial distance in hybrid matching
            temporal_weight: Weight for temporal continuity
            embedding_weight: Weight for embedding similarity
            max_embeddings_per_student: Max embeddings to store per student
            cooldown_period: Seconds to wait before creating new student ID
            quality_threshold: Minimum quality score for face crops
        """
        self.similarity_threshold = similarity_threshold
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.embedding_weight = embedding_weight
        self.max_embeddings_per_student = max_embeddings_per_student
        self.cooldown_period = cooldown_period
        self.quality_threshold = quality_threshold
        
        # Student data storage
        self.student_embeddings = {}      # {student_id: deque([emb1, emb2, ...])}
        self.student_avg_embeddings = {}  # {student_id: avg_embedding}
        self.student_last_seen = {}       # {student_id: timestamp}
        self.student_last_bbox = {}       # {student_id: (x1, y1, x2, y2)}
        self.student_quality_scores = {}  # {student_id: [quality_scores]}
        
        # Track management
        self.track_to_student = {}        # {track_id: student_id}
        self.track_history = {}           # {track_id: deque([(bbox, time), ...])}
        self.track_first_seen = {}        # {track_id: timestamp}
        
        # Recently lost tracks (grace period)
        self.lost_tracks = {}             # {track_id: (student_id, lost_time)}
        self.lost_track_timeout = 5.0     # Keep lost tracks for 5 seconds
        
        # Statistics
        self.embedding_count = 0
        self.match_count = 0
        self.new_student_count = 0
        self.quality_rejected = 0
        self.cooldown_prevented = 0
        
        # Initialize InsightFace
        self.app = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize InsightFace model"""
        try:
            print("🔄 Loading InsightFace model for Robust ReID...")
            
            import insightface
            from insightface.app import FaceAnalysis
            
            self.app = FaceAnalysis(
                name='buffalo_s',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_size=(160, 160))
            
            print("✅ Robust ReID system initialized")
            print(f"   Similarity threshold: {self.similarity_threshold}")
            print(f"   Cooldown period: {self.cooldown_period}s")
            print(f"   Quality threshold: {self.quality_threshold}")
            print(f"   Hybrid matching: embedding({self.embedding_weight}) + "
                  f"spatial({self.spatial_weight}) + temporal({self.temporal_weight})")
            
        except Exception as e:
            print(f"⚠️  Failed to load InsightFace: {str(e)}")
            self.app = None
    
    def assess_face_quality(self, face_crop, bbox, confidence):
        """
        Assess quality of face crop
        Rejects: blurry, tiny, extreme angles, low confidence
        
        Args:
            face_crop: Face image
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Detection confidence
            
        Returns:
            Quality score (0-100), higher is better
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0
        
        quality_score = 0.0
        
        # 1. Size quality (0-30 points)
        height, width = face_crop.shape[:2]
        size_score = min(30, (min(width, height) / 160.0) * 30)
        quality_score += size_score
        
        # 2. Confidence quality (0-30 points)
        conf_score = confidence * 30
        quality_score += conf_score
        
        # 3. Sharpness quality (0-25 points) - Laplacian variance
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(25, (laplacian_var / 500.0) * 25)
            quality_score += sharpness_score
        except:
            pass
        
        # 4. Aspect ratio quality (0-15 points)
        aspect_ratio = width / height if height > 0 else 0
        if 0.7 <= aspect_ratio <= 1.3:  # Near square is good for faces
            quality_score += 15
        elif 0.5 <= aspect_ratio <= 1.5:
            quality_score += 10
        else:
            quality_score += 5
        
        return quality_score
    
    def extract_embedding(self, face_crop):
        """Extract face embedding with quality check"""
        if self.app is None:
            return None
        
        try:
            # Resize for consistency
            if face_crop.shape[0] > 160 or face_crop.shape[1] > 160:
                face_crop = cv2.resize(face_crop, (160, 160))
            
            faces = self.app.get(face_crop)
            
            if len(faces) > 0:
                embedding = faces[0].embedding
                self.embedding_count += 1
                return embedding
            
            return None
            
        except Exception as e:
            return None
    
    def calculate_spatial_distance(self, bbox1, bbox2):
        """
        Calculate normalized spatial distance between two bboxes
        
        Returns:
            Distance score (0-1), 0 = same location, 1 = far apart
        """
        if bbox1 is None or bbox2 is None:
            return 1.0
        
        # Calculate centers
        x1_center = (bbox1[0] + bbox1[2]) / 2
        y1_center = (bbox1[1] + bbox1[3]) / 2
        x2_center = (bbox2[0] + bbox2[2]) / 2
        y2_center = (bbox2[1] + bbox2[3]) / 2
        
        # Euclidean distance
        distance = np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
        
        # Normalize by frame diagonal (assume 1920x1080)
        frame_diagonal = np.sqrt(1920**2 + 1080**2)
        normalized_distance = min(1.0, distance / frame_diagonal)
        
        return normalized_distance
    
    def calculate_temporal_score(self, student_id, current_time):
        """
        Calculate temporal continuity score
        Recently seen students get higher scores
        
        Returns:
            Score (0-1), 1 = just seen, 0 = not seen recently
        """
        if student_id not in self.student_last_seen:
            return 0.0
        
        time_since_seen = current_time - self.student_last_seen[student_id]
        
        # Exponential decay: score = e^(-time/tau)
        tau = 5.0  # Time constant (seconds)
        score = np.exp(-time_since_seen / tau)
        
        return score
    
    def hybrid_matching_score(self, embedding, bbox, current_time, student_id):
        """
        Calculate hybrid matching score combining:
        - Embedding similarity
        - Spatial distance
        - Temporal continuity
        
        Returns:
            Combined score (0-1), higher is better match
        """
        # 1. Embedding similarity
        if student_id in self.student_avg_embeddings:
            avg_emb = self.student_avg_embeddings[student_id]
            emb_similarity = cosine_similarity(
                embedding.reshape(1, -1),
                avg_emb.reshape(1, -1)
            )[0][0]
        else:
            emb_similarity = 0.0
        
        # 2. Spatial proximity (convert distance to similarity)
        if student_id in self.student_last_bbox:
            spatial_distance = self.calculate_spatial_distance(bbox, self.student_last_bbox[student_id])
            spatial_similarity = 1.0 - spatial_distance
        else:
            spatial_similarity = 0.5  # Neutral if no previous location
        
        # 3. Temporal continuity
        temporal_score = self.calculate_temporal_score(student_id, current_time)
        
        # Weighted combination
        combined_score = (
            self.embedding_weight * emb_similarity +
            self.spatial_weight * spatial_similarity +
            self.temporal_weight * temporal_score
        )
        
        return combined_score, emb_similarity, spatial_similarity, temporal_score
    
    def find_best_match(self, embedding, bbox, current_time, exclude_track_id=None):
        """
        Find best matching student using hybrid approach
        
        Returns:
            Tuple (student_id, combined_score, details_dict) or (None, 0.0, {})
        """
        if embedding is None or len(self.student_embeddings) == 0:
            return None, 0.0, {}
        
        best_match_id = None
        best_score = 0.0
        best_details = {}
        
        for student_id in self.student_embeddings.keys():
            # Skip if this student is currently tracked by excluded track
            if exclude_track_id and self.track_to_student.get(exclude_track_id) == student_id:
                continue
            
            # Calculate hybrid score
            combined_score, emb_sim, spatial_sim, temporal_score = self.hybrid_matching_score(
                embedding, bbox, current_time, student_id
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_match_id = student_id
                best_details = {
                    'embedding_similarity': emb_sim,
                    'spatial_similarity': spatial_sim,
                    'temporal_score': temporal_score,
                    'combined_score': combined_score
                }
        
        # Return match only if above threshold
        if best_score >= self.similarity_threshold:
            self.match_count += 1
            return best_match_id, best_score, best_details
        
        return None, best_score, best_details
    
    def update_student_embeddings(self, student_id, embedding):
        """
        Update student embeddings with outlier rejection
        Uses rolling average for smoothing
        """
        if student_id not in self.student_embeddings:
            self.student_embeddings[student_id] = deque(maxlen=self.max_embeddings_per_student)
            self.student_quality_scores[student_id] = deque(maxlen=self.max_embeddings_per_student)
        
        # Check if embedding is an outlier (if we have existing embeddings)
        if len(self.student_embeddings[student_id]) >= 3:
            avg_emb = self.student_avg_embeddings[student_id]
            similarity = cosine_similarity(
                embedding.reshape(1, -1),
                avg_emb.reshape(1, -1)
            )[0][0]
            
            # Reject outliers (too different from average)
            if similarity < 0.4:
                print(f"   ⚠️  Outlier embedding rejected for Student {student_id} (sim: {similarity:.3f})")
                return
        
        # Add embedding
        self.student_embeddings[student_id].append(embedding)
        
        # Update average embedding
        embeddings_array = np.array(list(self.student_embeddings[student_id]))
        self.student_avg_embeddings[student_id] = np.mean(embeddings_array, axis=0)
    
    def check_cooldown(self, current_time):
        """
        Check if enough time has passed since last new student
        Prevents rapid ID creation
        
        Returns:
            True if can create new ID, False if in cooldown
        """
        if len(self.student_last_seen) == 0:
            return True
        
        # Find most recent student creation
        most_recent = max(self.student_last_seen.values())
        time_since_last = current_time - most_recent
        
        if time_since_last < self.cooldown_period:
            self.cooldown_prevented += 1
            return False
        
        return True
    
    def register_or_match_face(self, track_id, face_crop, bbox, confidence, frame_time):
        """
        Main ReID function with all robust features
        
        Returns:
            Tuple (student_id, is_new, details_dict)
        """
        # Quality check
        quality_score = self.assess_face_quality(face_crop, bbox, confidence)
        
        if quality_score < self.quality_threshold:
            self.quality_rejected += 1
            # Return existing assignment or temporary ID
            if track_id in self.track_to_student:
                return self.track_to_student[track_id], False, {'quality': quality_score, 'rejected': True}
            else:
                return track_id, True, {'quality': quality_score, 'rejected': True}
        
        # Check if track already assigned
        if track_id in self.track_to_student:
            student_id = self.track_to_student[track_id]
            self.student_last_seen[student_id] = frame_time
            self.student_last_bbox[student_id] = bbox
            
            # Update track history
            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=30)
            self.track_history[track_id].append((bbox, frame_time))
            
            return student_id, False, {'quality': quality_score, 'existing': True}
        
        # Check if this is a recently lost track
        if track_id in self.lost_tracks:
            student_id, lost_time = self.lost_tracks[track_id]
            if frame_time - lost_time < self.lost_track_timeout:
                # Recover the track
                print(f"🔄 Track {track_id} recovered → Student {student_id}")
                self.track_to_student[track_id] = student_id
                self.student_last_seen[student_id] = frame_time
                self.student_last_bbox[student_id] = bbox
                del self.lost_tracks[track_id]
                return student_id, False, {'quality': quality_score, 'recovered': True}
        
        # Extract embedding
        embedding = self.extract_embedding(face_crop)
        
        if embedding is None:
            # Failed to extract, use track ID temporarily
            return track_id, True, {'quality': quality_score, 'no_embedding': True}
        
        # Try to match with existing students (hybrid matching)
        matched_id, match_score, match_details = self.find_best_match(
            embedding, bbox, frame_time, exclude_track_id=track_id
        )
        
        if matched_id is not None:
            # Matched with existing student
            print(f"🔄 ReID: Track {track_id} → Student {matched_id} "
                  f"(score: {match_score:.3f}, emb: {match_details['embedding_similarity']:.3f}, "
                  f"spatial: {match_details['spatial_similarity']:.3f}, "
                  f"temporal: {match_details['temporal_score']:.3f})")
            
            self.track_to_student[track_id] = matched_id
            self.student_last_seen[matched_id] = frame_time
            self.student_last_bbox[matched_id] = bbox
            
            # Update embeddings with smoothing
            self.update_student_embeddings(matched_id, embedding)
            
            match_details['quality'] = quality_score
            return matched_id, False, match_details
        
        else:
            # Check cooldown before creating new student
            if not self.check_cooldown(frame_time):
                print(f"⏳ Cooldown active, delaying new student creation for Track {track_id}")
                # Use track ID temporarily
                return track_id, True, {'quality': quality_score, 'cooldown': True}
            
            # Create new student
            self.new_student_count += 1
            new_student_id = self.new_student_count
            
            print(f"✨ New student: Student {new_student_id} (Track {track_id}, quality: {quality_score:.1f})")
            
            # Register new student
            self.student_embeddings[new_student_id] = deque(maxlen=self.max_embeddings_per_student)
            self.student_embeddings[new_student_id].append(embedding)
            self.student_avg_embeddings[new_student_id] = embedding
            self.student_quality_scores[new_student_id] = deque(maxlen=self.max_embeddings_per_student)
            self.student_quality_scores[new_student_id].append(quality_score)
            
            self.track_to_student[track_id] = new_student_id
            self.student_last_seen[new_student_id] = frame_time
            self.student_last_bbox[new_student_id] = bbox
            self.track_first_seen[track_id] = frame_time
            
            return new_student_id, True, {'quality': quality_score, 'new': True}
    
    def handle_track_lost(self, track_id):
        """Handle lost track with grace period"""
        if track_id in self.track_to_student:
            student_id = self.track_to_student[track_id]
            lost_time = time.time()
            
            # Move to lost tracks (grace period)
            self.lost_tracks[track_id] = (student_id, lost_time)
            
            print(f"⚠️  Track {track_id} lost (Student {student_id}) - grace period active")
            
            # Remove from active tracking
            del self.track_to_student[track_id]
    
    def cleanup_old_tracks(self, current_time):
        """Clean up old lost tracks"""
        tracks_to_remove = []
        
        for track_id, (student_id, lost_time) in self.lost_tracks.items():
            if current_time - lost_time > self.lost_track_timeout:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.lost_tracks[track_id]
    
    def get_statistics(self):
        """Get comprehensive statistics"""
        return {
            'total_students': len(self.student_embeddings),
            'active_tracks': len(self.track_to_student),
            'lost_tracks': len(self.lost_tracks),
            'embeddings_extracted': self.embedding_count,
            'successful_matches': self.match_count,
            'new_students_detected': self.new_student_count,
            'quality_rejected': self.quality_rejected,
            'cooldown_prevented': self.cooldown_prevented
        }
