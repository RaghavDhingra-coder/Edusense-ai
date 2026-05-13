"""
Image management module for saving and organizing student face crops
"""

import os
import cv2
import time
from datetime import datetime
from pathlib import Path
import config


class ImageManager:
    """
    Manages saving and organizing student face images
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the image manager
        
        Args:
            output_dir: Base directory for saving images
        """
        self.output_dir = output_dir or config.OUTPUT_DIR
        self.save_interval = config.SAVE_INTERVAL
        self.crop_padding = config.CROP_PADDING
        self.image_format = config.IMAGE_FORMAT
        self.image_quality = config.IMAGE_QUALITY
        
        # Track last save time and frame counter for each student
        self.last_save_time = {}
        self.save_counters = {}  # sequential per-student counter for unique filenames
        
        # Create base output directory
        self._create_base_directory()
    
    def _create_base_directory(self):
        """
        Create the base output directory if it doesn't exist
        """
        try:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            print(f"📁 Output directory: {self.output_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to create output directory: {str(e)}")
    
    def create_student_folder(self, student_id):
        """
        Create a folder for a specific student
        
        Args:
            student_id: Unique student ID
            
        Returns:
            Path to student folder
        """
        student_folder = os.path.join(self.output_dir, f"student_{student_id}")
        
        try:
            Path(student_folder).mkdir(parents=True, exist_ok=True)
            return student_folder
        except Exception as e:
            print(f"⚠️  Failed to create folder for student {student_id}: {str(e)}")
            return None
    
    def should_save_image(self, track_id, current_time=None):
        """
        Check if enough time has passed to save a new image for this student.

        Args:
            track_id: Student tracking ID
            current_time: Timestamp to compare against. Pass video-clock seconds for
                          video files so the interval is measured in video-time rather
                          than wall-clock time (avoids too-few images when the video is
                          processed faster than real-time).

        Returns:
            True if image should be saved, False otherwise
        """
        if current_time is None:
            current_time = time.time()

        if track_id not in self.last_save_time:
            return True

        return (current_time - self.last_save_time[track_id]) >= self.save_interval
    
    def crop_face(self, frame, bbox):
        """
        Crop ONLY face region from frame with minimal padding
        Ensures tight face crops without body/background
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Cropped face image or None if invalid
        """
        if frame is None or frame.size == 0:
            return None
        
        x1, y1, x2, y2 = bbox
        height, width = frame.shape[:2]
        
        # Calculate face dimensions
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Validate face dimensions before cropping
        if face_width < config.MIN_CROP_SIZE or face_height < config.MIN_CROP_SIZE:
            return None
        
        # Add small padding (reduced for tighter face crops)
        padding = self.crop_padding
        x1_padded = max(0, x1 - padding)
        y1_padded = max(0, y1 - padding)
        x2_padded = min(width, x2 + padding)
        y2_padded = min(height, y2 + padding)
        
        # Ensure we don't go out of bounds
        if x1_padded >= x2_padded or y1_padded >= y2_padded:
            return None
        
        # Crop face region
        try:
            face_crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]
            
            # Strict validation for proper face crops
            if face_crop.size == 0:
                return None
            
            crop_height, crop_width = face_crop.shape[:2]
            
            # Ensure minimum crop dimensions
            if crop_width < config.MIN_CROP_SIZE or crop_height < config.MIN_CROP_SIZE:
                return None
            
            # Validate aspect ratio (faces should be roughly square)
            aspect_ratio = crop_width / crop_height if crop_height > 0 else 0
            if aspect_ratio < 0.4 or aspect_ratio > 2.0:
                return None
            
            return face_crop
            
        except Exception as e:
            print(f"⚠️  Crop error: {str(e)}")
            return None
    
    def validate_face_crop(self, crop):
        """
        Validate that a crop is a proper face image
        
        Args:
            crop: Cropped image
            
        Returns:
            True if valid, False otherwise
        """
        if crop is None or crop.size == 0:
            return False
        
        height, width = crop.shape[:2]
        
        # Check minimum dimensions
        if width < config.MIN_CROP_SIZE or height < config.MIN_CROP_SIZE:
            return False
        
        # Check aspect ratio
        aspect_ratio = width / height if height > 0 else 0
        if aspect_ratio < 0.4 or aspect_ratio > 2.0:
            return False
        
        return True
    
    def save_face_image(self, frame, bbox, track_id, confidence, current_time=None):
        """
        Save face crop with strict validation.
        Only saves if: confidence > threshold, size > minimum, crop is valid.

        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            track_id: Student tracking ID
            confidence: Detection confidence
            current_time: Optional timestamp override (pass video-clock seconds for
                          video files so save interval is measured in video-time, not
                          wall-clock time — avoids too-few images when video is
                          processed faster than real-time).

        Returns:
            True if saved successfully, False otherwise
        """
        # Check if we should save based on time interval
        if not self.should_save_image(track_id, current_time):
            return False

        # Validate confidence threshold
        if confidence < config.CONFIDENCE_THRESHOLD:
            return False

        # Validate bbox dimensions
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        if width < config.MIN_FACE_WIDTH or height < config.MIN_FACE_HEIGHT:
            return False

        # Crop face with tight bounds
        face_crop = self.crop_face(frame, bbox)
        if face_crop is None:
            return False

        # Validate the crop is a proper face
        if not self.validate_face_crop(face_crop):
            return False

        # Create student folder
        student_folder = self.create_student_folder(track_id)
        if student_folder is None:
            return False

        # Unique sequential filename — avoids collisions when the video is
        # processed faster than real-time (multiple frames share the same
        # wall-clock second and would otherwise overwrite each other).
        count = self.save_counters.get(track_id, 0)
        self.save_counters[track_id] = count + 1
        filename = f"frame_{count:06d}.{self.image_format}"
        filepath = os.path.join(student_folder, filename)

        # Save image
        try:
            if self.image_format.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(filepath, face_crop, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
            else:
                cv2.imwrite(filepath, face_crop)

            # Record the time of this save (use provided timestamp for video-time accounting)
            self.last_save_time[track_id] = current_time if current_time is not None else time.time()

            return True
        except Exception as e:
            print(f"⚠️  Failed to save image for student {track_id}: {str(e)}")
            return False
    
    def get_student_count(self):
        """
        Get the number of unique students tracked
        
        Returns:
            Number of unique students
        """
        return len(self.last_save_time)
    
    def get_statistics(self):
        """
        Get statistics about saved images
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_students': 0,
            'total_images': 0,
            'students': {}
        }
        
        try:
            if not os.path.exists(self.output_dir):
                return stats
            
            for student_folder in os.listdir(self.output_dir):
                folder_path = os.path.join(self.output_dir, student_folder)
                
                if os.path.isdir(folder_path) and student_folder.startswith('student_'):
                    student_id = student_folder.replace('student_', '')
                    image_count = len([f for f in os.listdir(folder_path) 
                                     if f.endswith(f'.{self.image_format}')])
                    
                    stats['students'][student_id] = image_count
                    stats['total_images'] += image_count
                    stats['total_students'] += 1
            
            return stats
        except Exception as e:
            print(f"⚠️  Error getting statistics: {str(e)}")
            return stats
