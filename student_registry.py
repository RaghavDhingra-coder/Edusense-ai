"""
Student Registration System
Manages registered students with face embeddings for recognition
"""

import os
import json
import pickle
import numpy as np
import time
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import logging

logger = logging.getLogger(__name__)


class StudentRegistry:
    """
    Manages registered students with face embeddings
    Provides fast recognition using cached embeddings
    """
    
    def __init__(self, registry_dir='registered_students'):
        """
        Initialize student registry
        
        Args:
            registry_dir: Directory to store registered student data
        """
        self.registry_dir = registry_dir
        self.students = {}  # {student_id: StudentProfile}
        self.embeddings_matrix = None  # Numpy array of all embeddings
        self.student_ids_list = []  # Ordered list of student IDs
        
        # Recognition settings
        self.recognition_threshold = 0.60  # Cosine similarity threshold (lowered from 0.65 for faster recognition)
        self.min_embeddings = 3  # Minimum embeddings per student
        
        # InsightFace model (will be set externally)
        self.face_app = None
        
        # Create registry directory
        os.makedirs(registry_dir, exist_ok=True)
        
        # Load existing students
        self._load_registry()
        
        logger.info(f"📚 Student Registry initialized: {len(self.students)} students")
    
    def set_face_app(self, face_app):
        """Set InsightFace app for embedding extraction"""
        self.face_app = face_app
        logger.info("✅ InsightFace app connected to registry")
    
    def register_student(self, student_name, student_id, face_images):
        """
        Register a new student with face images
        
        Args:
            student_name: Student's full name
            student_id: Unique student ID (USN)
            face_images: List of face crop images (numpy arrays)
            
        Returns:
            dict with success status and message
        """
        try:
            logger.info("=" * 60)
            logger.info(f"📝 Registering student: {student_name} ({student_id})")
            logger.info("=" * 60)
            
            # Validate inputs
            if not student_name or not student_id:
                return {'success': False, 'error': 'Name and ID required'}
            
            if not face_images or len(face_images) == 0:
                return {'success': False, 'error': 'At least one face image required'}
            
            # Check if already registered
            if student_id in self.students:
                return {'success': False, 'error': f'Student {student_id} already registered'}
            
            # Extract embeddings from face images
            embeddings = []
            valid_images = []
            
            for idx, face_img in enumerate(face_images):
                embedding = self._extract_embedding(face_img)
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_images.append(face_img)
                    logger.info(f"   ✅ Embedding {idx+1}/{len(face_images)} extracted")
                else:
                    logger.warning(f"   ⚠️  Embedding {idx+1}/{len(face_images)} failed")
            
            if len(embeddings) < self.min_embeddings:
                return {
                    'success': False,
                    'error': f'Need at least {self.min_embeddings} valid face images, got {len(embeddings)}'
                }
            
            # Create student profile
            profile = {
                'student_id': student_id,
                'student_name': student_name,
                'embeddings': embeddings,
                'num_embeddings': len(embeddings),
                'registered_date': datetime.now().isoformat(),
                'last_seen': None
            }
            
            # Save student data
            self._save_student(student_id, profile, valid_images)
            
            # Add to registry
            self.students[student_id] = profile
            
            # Rebuild embeddings matrix
            self._rebuild_embeddings_matrix()
            
            logger.info(f"✅ Student registered: {student_name}")
            logger.info(f"   ID: {student_id}")
            logger.info(f"   Embeddings: {len(embeddings)}")
            logger.info("=" * 60)
            
            return {
                'success': True,
                'message': f'Student {student_name} registered successfully',
                'student_id': student_id,
                'num_embeddings': len(embeddings)
            }
            
        except Exception as e:
            logger.error(f"❌ Registration failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def recognize_face(self, face_image):
        """
        Recognize a face against registered students
        OPTIMIZED: Added detailed timing logs
        
        Args:
            face_image: Face crop image (numpy array)
            
        Returns:
            dict with student_id, student_name, confidence, or None if unknown
        """
        try:
            start_time = time.time()
            
            # Extract embedding
            embedding = self._extract_embedding(face_image)
            embedding_time = (time.time() - start_time) * 1000
            
            if embedding is None:
                logger.debug(f"⚠️  Embedding extraction failed ({embedding_time:.1f}ms)")
                return None
            
            # No registered students
            if len(self.students) == 0 or self.embeddings_matrix is None:
                return None
            
            # Compare with all registered embeddings
            compare_start = time.time()
            embedding_2d = embedding.reshape(1, -1)
            similarities = cosine_similarity(embedding_2d, self.embeddings_matrix)[0]
            compare_time = (time.time() - compare_start) * 1000
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            total_time = (time.time() - start_time) * 1000
            
            # Check threshold
            if best_similarity >= self.recognition_threshold:
                student_id = self.student_ids_list[best_idx]
                student_name = self.students[student_id]['student_name']
                
                # Update last seen
                self.students[student_id]['last_seen'] = datetime.now().isoformat()
                
                logger.debug(f"✅ Match: {student_name} ({best_similarity:.3f}) - embed:{embedding_time:.1f}ms, compare:{compare_time:.1f}ms, total:{total_time:.1f}ms")
                
                return {
                    'student_id': student_id,
                    'student_name': student_name,
                    'confidence': float(best_similarity),
                    'is_registered': True
                }
            else:
                logger.debug(f"❓ No match (best: {best_similarity:.3f}) - total:{total_time:.1f}ms")
                return None
                
        except Exception as e:
            logger.error(f"❌ Recognition error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_all_students(self):
        """Get list of all registered students"""
        return [
            {
                'student_id': sid,
                'student_name': profile['student_name'],
                'num_embeddings': profile['num_embeddings'],
                'registered_date': profile['registered_date'],
                'last_seen': profile.get('last_seen')
            }
            for sid, profile in self.students.items()
        ]
    
    def delete_student(self, student_id):
        """Delete a registered student"""
        try:
            if student_id not in self.students:
                return {'success': False, 'error': 'Student not found'}
            
            # Remove from registry
            student_name = self.students[student_id]['student_name']
            del self.students[student_id]
            
            # Delete files
            student_dir = os.path.join(self.registry_dir, student_id)
            if os.path.exists(student_dir):
                import shutil
                shutil.rmtree(student_dir)
            
            # Rebuild embeddings matrix
            self._rebuild_embeddings_matrix()
            
            logger.info(f"🗑️  Deleted student: {student_name} ({student_id})")
            
            return {
                'success': True,
                'message': f'Student {student_name} deleted'
            }
            
        except Exception as e:
            logger.error(f"❌ Delete failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_embedding(self, face_image):
        """
        Extract face embedding using InsightFace
        OPTIMIZED: Reduced image preprocessing overhead
        """
        if self.face_app is None:
            logger.error("❌ InsightFace app not set")
            return None
        
        try:
            # Resize to optimal size for InsightFace (112x112 is native for recognition model)
            # Avoid unnecessary resizing if already close to target size
            h, w = face_image.shape[:2]
            
            if h > 160 or w > 160 or h < 80 or w < 80:
                # Only resize if significantly different from optimal range
                face_image = cv2.resize(face_image, (112, 112))
            
            # Extract faces
            faces = self.face_app.get(face_image)
            
            if len(faces) > 0:
                return faces[0].embedding
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Embedding extraction error: {e}")
            return None
    
    def _save_student(self, student_id, profile, images):
        """Save student data to disk"""
        student_dir = os.path.join(self.registry_dir, student_id)
        os.makedirs(student_dir, exist_ok=True)
        
        # Save metadata (without embeddings - too large for JSON)
        metadata = {
            'student_id': profile['student_id'],
            'student_name': profile['student_name'],
            'num_embeddings': profile['num_embeddings'],
            'registered_date': profile['registered_date']
        }
        
        metadata_path = os.path.join(student_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save embeddings as pickle
        embeddings_path = os.path.join(student_dir, 'embeddings.pkl')
        with open(embeddings_path, 'wb') as f:
            pickle.dump(profile['embeddings'], f)
        
        # Save sample images
        images_dir = os.path.join(student_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        for idx, img in enumerate(images[:5]):  # Save first 5 images
            img_path = os.path.join(images_dir, f'sample_{idx+1}.jpg')
            cv2.imwrite(img_path, img)
    
    def _load_registry(self):
        """Load all registered students from disk"""
        if not os.path.exists(self.registry_dir):
            return
        
        for student_id in os.listdir(self.registry_dir):
            student_dir = os.path.join(self.registry_dir, student_id)
            
            if not os.path.isdir(student_dir):
                continue
            
            try:
                # Load metadata
                metadata_path = os.path.join(student_dir, 'metadata.json')
                if not os.path.exists(metadata_path):
                    continue
                
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Load embeddings
                embeddings_path = os.path.join(student_dir, 'embeddings.pkl')
                if not os.path.exists(embeddings_path):
                    continue
                
                with open(embeddings_path, 'rb') as f:
                    embeddings = pickle.load(f)
                
                # Create profile
                profile = {
                    'student_id': metadata['student_id'],
                    'student_name': metadata['student_name'],
                    'embeddings': embeddings,
                    'num_embeddings': len(embeddings),
                    'registered_date': metadata['registered_date'],
                    'last_seen': None
                }
                
                self.students[student_id] = profile
                logger.info(f"   Loaded: {metadata['student_name']} ({len(embeddings)} embeddings)")
                
            except Exception as e:
                logger.error(f"   Failed to load {student_id}: {e}")
        
        # Build embeddings matrix
        self._rebuild_embeddings_matrix()
    
    def _rebuild_embeddings_matrix(self):
        """Rebuild the embeddings matrix for fast comparison"""
        if len(self.students) == 0:
            self.embeddings_matrix = None
            self.student_ids_list = []
            return
        
        all_embeddings = []
        self.student_ids_list = []
        
        for student_id, profile in self.students.items():
            for embedding in profile['embeddings']:
                all_embeddings.append(embedding)
                self.student_ids_list.append(student_id)
        
        self.embeddings_matrix = np.array(all_embeddings)
        logger.info(f"📊 Embeddings matrix rebuilt: {self.embeddings_matrix.shape}")
