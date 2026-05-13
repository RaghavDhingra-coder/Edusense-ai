"""
Identity Merger - Post-processing module for merging duplicate student IDs
Runs after attentiveness analysis to consolidate duplicate identities
"""

import os
import shutil
import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available, using fallback similarity method")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IdentityMerger:
    """
    Automatic identity merger for duplicate student IDs
    Uses face embeddings to detect and merge duplicate identities
    """
    
    def __init__(self, 
                 students_dir='students',
                 similarity_threshold=0.75,
                 max_samples_per_student=10):
        """
        Initialize identity merger
        
        Args:
            students_dir: Directory containing student folders
            similarity_threshold: Cosine similarity threshold for merging (0-1)
            max_samples_per_student: Max images to sample per student for embedding
        """
        self.students_dir = students_dir
        self.similarity_threshold = similarity_threshold
        self.max_samples_per_student = max_samples_per_student
        
        # Initialize face recognition model
        if INSIGHTFACE_AVAILABLE:
            try:
                self.face_app = FaceAnalysis(
                    name='buffalo_l',
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                self.use_insightface = True
                logger.info("✅ Using InsightFace for embeddings")
            except Exception as e:
                logger.warning(f"InsightFace initialization failed: {e}")
                self.use_insightface = False
        else:
            self.use_insightface = False
        
        if not self.use_insightface:
            logger.info("⚠️  Using fallback histogram-based similarity")
        
        logger.info("=" * 60)
        logger.info("🔧 Identity Merger Initialized")
        logger.info("=" * 60)
        logger.info(f"Students dir: {students_dir}")
        logger.info(f"Similarity threshold: {similarity_threshold}")
        logger.info(f"Max samples: {max_samples_per_student}")
        logger.info("=" * 60)
    
    def extract_embedding_insightface(self, image_path: str) -> Optional[np.ndarray]:
        """Extract face embedding using InsightFace"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Resize if too small
            h, w = img.shape[:2]
            if h < 112 or w < 112:
                scale = max(112 / h, 112 / w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            
            faces = self.face_app.get(img)
            
            if len(faces) == 0:
                # If no face detected, assume entire image is a face (pre-cropped)
                # Use recognition model directly with get_feat
                try:
                    # Get recognition model
                    rec_model = self.face_app.models.get('recognition')
                    
                    if rec_model is None:
                        return None
                    
                    # Prepare face image - resize to 112x112
                    face_img = cv2.resize(img, (112, 112))
                    
                    # Get embedding using model's get_feat method
                    embedding = rec_model.get_feat(face_img)
                    
                    # Normalize
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-7)
                    
                    return embedding
                except Exception as e:
                    logger.debug(f"Direct recognition failed: {e}")
                    return None
            
            # Use the first (largest) face
            face = faces[0]
            embedding = face.embedding
            
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-7)
            
            return embedding
        
        except Exception as e:
            logger.debug(f"Failed to extract embedding from {image_path}: {e}")
            return None
    
    def extract_embedding_fallback(self, image_path: str) -> Optional[np.ndarray]:
        """Fallback: Extract histogram-based feature vector"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Resize to standard size
            img = cv2.resize(img, (128, 128))
            
            # Convert to multiple color spaces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Compute histograms
            hist_gray = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            
            # Concatenate and normalize
            feature = np.concatenate([
                hist_gray.flatten(),
                hist_h.flatten(),
                hist_s.flatten()
            ])
            
            feature = feature / (np.linalg.norm(feature) + 1e-7)
            
            return feature
        
        except Exception as e:
            logger.debug(f"Failed to extract fallback features from {image_path}: {e}")
            return None
    
    def extract_student_embedding(self, student_folder: str) -> Optional[np.ndarray]:
        """
        Extract representative embedding for a student
        Samples multiple images and averages embeddings
        
        Args:
            student_folder: Path to student folder
            
        Returns:
            Average embedding vector or None
        """
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = [
            os.path.join(student_folder, f)
            for f in os.listdir(student_folder)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        if len(image_files) == 0:
            return None
        
        # Sample images evenly
        if len(image_files) > self.max_samples_per_student:
            step = len(image_files) // self.max_samples_per_student
            sampled_files = image_files[::step][:self.max_samples_per_student]
        else:
            sampled_files = image_files
        
        # Extract embeddings
        embeddings = []
        
        for img_file in sampled_files:
            if self.use_insightface:
                emb = self.extract_embedding_insightface(img_file)
            else:
                emb = self.extract_embedding_fallback(img_file)
            
            if emb is not None:
                embeddings.append(emb)
        
        if len(embeddings) == 0:
            return None
        
        # Average embeddings
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Normalize
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-7)
        
        return avg_embedding
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        # Flatten if needed
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        
        similarity = np.dot(emb1, emb2)
        return float(similarity)
    
    def get_image_timestamps(self, student_folder: str) -> List[float]:
        """Get timestamps of images in student folder"""
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = [
            os.path.join(student_folder, f)
            for f in os.listdir(student_folder)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
        
        timestamps = []
        for img_file in image_files:
            try:
                # Get file modification time
                timestamp = os.path.getmtime(img_file)
                timestamps.append(timestamp)
            except:
                pass
        
        return timestamps
    
    def check_temporal_overlap(self, timestamps1: List[float], timestamps2: List[float]) -> bool:
        """
        Check if two students have temporal overlap
        If they appear simultaneously, they are different people
        
        Returns:
            True if NO overlap (can be same person)
            False if overlap exists (different people)
        """
        if not timestamps1 or not timestamps2:
            return True  # No data, assume no overlap
        
        min1, max1 = min(timestamps1), max(timestamps1)
        min2, max2 = min(timestamps2), max(timestamps2)
        
        # Check for overlap
        overlap = not (max1 < min2 or max2 < min1)
        
        # Return True if NO overlap (can merge)
        return not overlap
    
    def find_duplicate_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Find duplicate student ID pairs
        
        Returns:
            List of (student_id1, student_id2, similarity) tuples
        """
        logger.info("\n" + "=" * 60)
        logger.info("🔍 Scanning for duplicate identities...")
        logger.info("=" * 60)
        
        if not os.path.exists(self.students_dir):
            logger.warning(f"Students directory not found: {self.students_dir}")
            return []
        
        # Get all student folders
        student_folders = sorted([
            d for d in os.listdir(self.students_dir)
            if os.path.isdir(os.path.join(self.students_dir, d))
            and d.startswith('student_')
        ])
        
        if len(student_folders) < 2:
            logger.info("Less than 2 students, no merging needed")
            return []
        
        logger.info(f"Found {len(student_folders)} student folders")
        
        # Extract embeddings for all students
        logger.info("Extracting face embeddings...")
        student_embeddings = {}
        student_timestamps = {}
        
        for student_id in student_folders:
            folder_path = os.path.join(self.students_dir, student_id)
            
            embedding = self.extract_student_embedding(folder_path)
            timestamps = self.get_image_timestamps(folder_path)
            
            if embedding is not None:
                student_embeddings[student_id] = embedding
                student_timestamps[student_id] = timestamps
                logger.info(f"  ✅ {student_id}: embedding extracted")
            else:
                logger.warning(f"  ⚠️  {student_id}: failed to extract embedding")
        
        # Find duplicate pairs
        logger.info("\nComparing student identities...")
        duplicate_pairs = []
        
        student_ids = list(student_embeddings.keys())
        
        for i in range(len(student_ids)):
            for j in range(i + 1, len(student_ids)):
                id1 = student_ids[i]
                id2 = student_ids[j]
                
                emb1 = student_embeddings[id1]
                emb2 = student_embeddings[id2]
                
                similarity = self.compute_similarity(emb1, emb2)
                
                logger.info(f"  {id1} vs {id2}: similarity = {similarity:.3f}")
                
                # Check if similarity exceeds threshold
                if similarity >= self.similarity_threshold:
                    # Check temporal overlap
                    no_overlap = self.check_temporal_overlap(
                        student_timestamps[id1],
                        student_timestamps[id2]
                    )
                    
                    if no_overlap:
                        duplicate_pairs.append((id1, id2, similarity))
                        logger.info(f"    ✅ DUPLICATE DETECTED (no temporal overlap)")
                    else:
                        logger.info(f"    ⚠️  High similarity but temporal overlap - different people")
        
        logger.info(f"\n✅ Found {len(duplicate_pairs)} duplicate pairs")
        
        return duplicate_pairs
    
    def merge_students(self, source_id: str, target_id: str) -> bool:
        """
        Merge source student into target student
        
        Args:
            source_id: Student ID to merge (will be removed)
            target_id: Student ID to merge into (will be kept)
            
        Returns:
            True if merge successful
        """
        source_folder = os.path.join(self.students_dir, source_id)
        target_folder = os.path.join(self.students_dir, target_id)
        
        if not os.path.exists(source_folder):
            logger.error(f"Source folder not found: {source_folder}")
            return False
        
        if not os.path.exists(target_folder):
            logger.error(f"Target folder not found: {target_folder}")
            return False
        
        try:
            # Move all images from source to target
            image_extensions = {'.jpg', '.jpeg', '.png'}
            source_images = [
                f for f in os.listdir(source_folder)
                if os.path.splitext(f)[1].lower() in image_extensions
            ]
            
            moved_count = 0
            for img_file in source_images:
                source_path = os.path.join(source_folder, img_file)
                target_path = os.path.join(target_folder, img_file)
                
                # Handle filename conflicts
                if os.path.exists(target_path):
                    base, ext = os.path.splitext(img_file)
                    counter = 1
                    while os.path.exists(target_path):
                        target_path = os.path.join(target_folder, f"{base}_merged{counter}{ext}")
                        counter += 1
                
                shutil.move(source_path, target_path)
                moved_count += 1
            
            # Remove source folder
            shutil.rmtree(source_folder)
            
            logger.info(f"  ✅ Moved {moved_count} images from {source_id} to {target_id}")
            logger.info(f"  ✅ Removed folder: {source_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to merge {source_id} into {target_id}: {e}")
            return False
    
    def merge_duplicates(self) -> Dict:
        """
        Find and merge all duplicate student identities
        
        Returns:
            Dict with merge statistics
        """
        logger.info("\n" + "=" * 60)
        logger.info("🔀 Starting Identity Merge Process")
        logger.info("=" * 60)
        
        # Find duplicate pairs
        duplicate_pairs = self.find_duplicate_pairs()
        
        if len(duplicate_pairs) == 0:
            logger.info("\n✅ No duplicates found - all identities are unique")
            return {
                'duplicates_found': 0,
                'merges_performed': 0,
                'merge_details': []
            }
        
        # Build merge groups
        # Use union-find to group all related duplicates
        merge_groups = self._build_merge_groups(duplicate_pairs)
        
        logger.info("\n" + "=" * 60)
        logger.info("🔀 Performing Merges...")
        logger.info("=" * 60)
        
        merge_details = []
        merges_performed = 0
        
        for group in merge_groups:
            if len(group) < 2:
                continue
            
            # Keep the first ID, merge others into it
            target_id = group[0]
            
            for source_id in group[1:]:
                logger.info(f"\n[MERGE] {source_id} → {target_id}")
                
                success = self.merge_students(source_id, target_id)
                
                if success:
                    merges_performed += 1
                    merge_details.append({
                        'source': source_id,
                        'target': target_id,
                        'success': True
                    })
                else:
                    merge_details.append({
                        'source': source_id,
                        'target': target_id,
                        'success': False
                    })
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Identity Merge Complete")
        logger.info("=" * 60)
        logger.info(f"Duplicate pairs found: {len(duplicate_pairs)}")
        logger.info(f"Merges performed: {merges_performed}")
        logger.info(f"Duplicate IDs removed: {merges_performed}")
        logger.info("=" * 60)
        
        return {
            'duplicates_found': len(duplicate_pairs),
            'merges_performed': merges_performed,
            'merge_details': merge_details
        }
    
    def _build_merge_groups(self, duplicate_pairs: List[Tuple[str, str, float]]) -> List[List[str]]:
        """
        Build merge groups using union-find
        Groups all transitively related duplicates together
        
        Example:
            If A=B and B=C, then group = [A, B, C]
        """
        # Union-Find data structure
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Build unions
        for id1, id2, _ in duplicate_pairs:
            union(id1, id2)
        
        # Group by root
        groups = defaultdict(list)
        for student_id in parent.keys():
            root = find(student_id)
            groups[root].append(student_id)
        
        # Sort each group to have consistent target
        result = []
        for group in groups.values():
            sorted_group = sorted(group, key=lambda x: int(x.split('_')[1]))
            result.append(sorted_group)
        
        return result
    
    def save_merge_history(self, merge_stats: Dict, output_file='merge_history.json'):
        """Save merge history to JSON file"""
        try:
            history = {
                'timestamp': str(np.datetime64('now')),
                'statistics': merge_stats
            }
            
            with open(output_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"Merge history saved to {output_file}")
        
        except Exception as e:
            logger.error(f"Failed to save merge history: {e}")
