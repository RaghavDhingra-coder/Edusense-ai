"""
Hybrid Attentiveness Analyzer
Prioritizes head pose detection with image quality fallback
"""

import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
import json
import logging
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import mediapipe as mp
# NEW: Import MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridAttentivenessAnalyzer:
    """
    Hybrid attentiveness analyzer:
    - Primary: Head pose estimation (yaw/pitch/roll)
    - Fallback: Image quality when pose fails
    - Temporal smoothing for stable results
    """

    def __init__(self,
                 students_dir='students',
                 focus_threshold=0.65,
                 yaw_threshold=30.0,
                 pitch_threshold=25.0,
                 roll_threshold=35.0,
                 consecutive_distraction_threshold=3,
                 model_path='face_landmarker.task'): # Added model_path
        """
        Initialize hybrid analyzer

        Args:
            students_dir: Directory containing student folders
            focus_threshold: Overall focus threshold (0-1)
            yaw_threshold: Max yaw angle for focused (degrees)
            pitch_threshold: Max pitch angle for focused (degrees)
            roll_threshold: Max roll angle for focused (degrees)
            consecutive_distraction_threshold: Frames before marking distracted
        """
        self.students_dir = students_dir
        self.focus_threshold = focus_threshold
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        self.consecutive_distraction_threshold = consecutive_distraction_threshold
        self.use_head_pose = True  # Can be disabled externally

        # NEW: Initialize MediaPipe Face Landmarker with GPU acceleration
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.GPU # Enable GPU
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # 3D model points for solvePnP
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

        logger.info("=" * 60)
        logger.info("🔧 Initializing Hybrid Attentiveness Analyzer (GPU Landmarker)")
        logger.info("=" * 60)
        logger.info(f"Students dir: {students_dir}")
        logger.info(f"Focus threshold: {focus_threshold * 100}%")
        logger.info(f"Yaw threshold: ±{yaw_threshold}°")
        logger.info(f"Pitch threshold: ±{pitch_threshold}°")
        logger.info(f"Roll threshold: ±{roll_threshold}°")
        logger.info(f"Consecutive distraction: {consecutive_distraction_threshold} frames")
        logger.info("=" * 60)

    def estimate_head_pose(self, image: np.ndarray) -> Optional[Dict]:
        """
        Estimate head pose from image

        Returns:
            Dict with yaw, pitch, roll, confidence, method
            None if estimation fails
        """
        try:
            if image is None or image.size == 0:
                return None

            h, w = image.shape[:2]

            # Convert to RGB and MediaPipe Image format
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            # Process with MediaPipe Face Landmarker
            results = self.detector.detect(mp_image)

            if not results.face_landmarks:
                return None

            # Access the first face's landmarks
            face_landmarks = results.face_landmarks[0]

            # Extract key landmarks for pose estimation
            landmark_indices = [1, 152, 33, 263, 61, 291]  # Nose, chin, eyes, mouth
            landmarks_2d = []

            for idx in landmark_indices:
                landmark = face_landmarks[idx] # Changed from .landmark[idx]
                x = landmark.x * w
                y = landmark.y * h
                landmarks_2d.append([x, y])

            landmarks_2d = np.array(landmarks_2d, dtype=np.float64)

            # Camera matrix
            focal_length = w
            camera_center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, camera_center[0]],
                [0, focal_length, camera_center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

            dist_coeffs = np.zeros((4, 1))

            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.model_points,
                landmarks_2d,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return None

            # Convert to rotation matrix
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)

            # Calculate Euler angles
            yaw, pitch, roll = self._rotation_matrix_to_euler(rotation_mat)

            # Calculate confidence based on landmark visibility
            confidence = self._calculate_pose_confidence(face_landmarks, yaw, pitch, roll)

            return {
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll,
                'confidence': confidence,
                'method': 'head_pose'
            }

        except Exception as e:
            logger.debug(f"Head pose estimation failed: {e}")
            return None

    def _rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        # Convert to degrees
        pitch = np.degrees(x)
        yaw = np.degrees(y)
        roll = np.degrees(z)

        # Normalize pitch to -180 to 180 range
        # If pitch is > 90, it means face is inverted, so adjust
        if pitch > 90:
            pitch = 180 - pitch
        elif pitch < -90:
            pitch = -180 - pitch

        return yaw, pitch, roll  # yaw, pitch, roll

    def _calculate_pose_confidence(self, face_landmarks, yaw: float, pitch: float, roll: float) -> float:
        """
        Calculate confidence in pose estimation

        Higher confidence when:
        - Face is frontal
        - Landmarks are well-distributed
        - Angles are reasonable
        """
        # Angle-based confidence (frontal = higher confidence)
        angle_distance = np.sqrt(yaw**2 + pitch**2 + roll**2)
        angle_confidence = max(0, 1 - (angle_distance / 90.0))

        # Landmark spread confidence (Landmarks are now in a list directly)
        xs = [lm.x for lm in face_landmarks[:10]]
        ys = [lm.y for lm in face_landmarks[:10]]
        spread = (max(xs) - min(xs)) * (max(ys) - min(ys))
        spread_confidence = min(1.0, spread * 10)

        # Combined confidence
        confidence = (angle_confidence * 0.6 + spread_confidence * 0.4)

        return confidence

    def calculate_image_quality(self, image: np.ndarray) -> Dict:
        """
        Calculate image quality metrics (fallback method)

        Returns:
            Dict with quality score and components
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(1.0, laplacian_var / 500.0)

            # Brightness
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2

            # Contrast
            contrast = gray.std() / 128.0
            contrast_score = min(1.0, contrast)

            # Overall quality
            quality = (sharpness * 0.5 + brightness_score * 0.3 + contrast_score * 0.2)

            return {
                'quality': quality,
                'sharpness': sharpness,
                'brightness': brightness,
                'contrast': contrast_score,
                'method': 'image_quality'
            }

        except Exception as e:
            logger.debug(f"Quality calculation failed: {e}")
            return {'quality': 0.0, 'method': 'image_quality'}

    def analyze_single_frame(self, image_path: str) -> Dict:
        """
        Analyze single frame - SIMPLIFIED POSE-ONLY LOGIC

        RULE: If pose detected → USE ONLY POSE (no overrides, no averaging)
        RULE: If pose fails → Use quality fallback

        Returns:
            Dict with is_focused, confidence, method, and metrics
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'is_focused': False,
                    'confidence': 0.0,
                    'method': 'failed',
                    'reason': 'image_load_failed',
                    'fallback_used': False
                }

            # Try head pose estimation
            pose_data = self.estimate_head_pose(image)

            # If pose detected, USE ONLY POSE - no quality override
            if pose_data is not None:
                yaw = pose_data['yaw']
                pitch = pose_data['pitch']
                roll = pose_data['roll']

                # SIMPLE RULE: Check thresholds directly
                # If abs(yaw) > 30 OR abs(pitch) > 30 → NOT_FOCUSED
                yaw_ok = abs(yaw) <= self.yaw_threshold
                pitch_ok = abs(pitch) <= self.pitch_threshold
                roll_ok = abs(roll) <= self.roll_threshold

                # POSE DECISION IS FINAL - no overrides
                is_focused = yaw_ok and pitch_ok and roll_ok

                # Debug logging
                decision_str = "FOCUSED" if is_focused else "NOT_FOCUSED"
                logger.info(f"[POSE] yaw={yaw:.1f}° pitch={pitch:.1f}° roll={roll:.1f}° "
                           f"→ {decision_str} (yaw_ok={yaw_ok}, pitch_ok={pitch_ok}, roll_ok={roll_ok})")

                return {
                    'is_focused': is_focused,
                    'confidence': pose_data['confidence'],
                    'method': 'head_pose',
                    'yaw': yaw,
                    'pitch': pitch,
                    'roll': roll,
                    'yaw_ok': yaw_ok,
                    'pitch_ok': pitch_ok,
                    'roll_ok': roll_ok,
                    'fallback_used': False
                }

            # FALLBACK ONLY: Pose completely failed
            logger.info(f"[FALLBACK] Pose detection failed, using quality")
            quality_data = self.calculate_image_quality(image)

            # Conservative: prefer unfocused when uncertain
            is_focused = quality_data['quality'] > 0.6

            decision_str = "FOCUSED" if is_focused else "NOT_FOCUSED"
            logger.info(f"[FALLBACK] quality={quality_data['quality']:.2f} → {decision_str}")

            return {
                'is_focused': is_focused,
                'confidence': quality_data['quality'] * 0.3,
                'method': 'image_quality_fallback',
                'quality': quality_data['quality'],
                'sharpness': quality_data.get('sharpness', 0),
                'brightness': quality_data.get('brightness', 0),
                'fallback_used': True,
                'yaw': None,
                'pitch': None,
                'roll': None
            }

        except Exception as e:
            logger.error(f"Frame analysis failed for {image_path}: {e}")
            return {
                'is_focused': False,
                'confidence': 0.0,
                'method': 'failed',
                'reason': str(e),
                'fallback_used': False
            }

    def analyze_student_folder(self, student_folder: str) -> Optional[Dict]:
        """
        Analyze all images in student folder - DIRECT COUNTING (minimal smoothing)

        Args:
            student_folder: Path to student folder

        Returns:
            Analytics dictionary with detailed pose information
        """
        student_id = os.path.basename(student_folder)

        logger.info(f"📁 Scanning folder: {student_folder}")

        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = sorted([
            os.path.join(student_folder, f)
            for f in os.listdir(student_folder)
            if os.path.splitext(f)[1].lower() in image_extensions
        ])

        if len(image_files) == 0:
            logger.warning(f"⚠️  No images in {student_folder}")
            return None

        logger.info(f"✅ Found {len(image_files)} images")

        # Analyze each frame - DIRECT COUNTING
        frame_results = []
        focused_count = 0
        unfocused_count = 0

        method_counts = defaultdict(int)
        pose_angles = {'yaw': [], 'pitch': [], 'roll': []}
        fallback_count = 0

        for img_file in image_files:
            result = self.analyze_single_frame(img_file)
            frame_results.append(result)

            method_counts[result['method']] += 1

            # Track pose angles if available
            if result.get('yaw') is not None:
                pose_angles['yaw'].append(result['yaw'])
                pose_angles['pitch'].append(result['pitch'])
                pose_angles['roll'].append(result['roll'])

            # Track fallback usage
            if result.get('fallback_used', False):
                fallback_count += 1

            # DIRECT COUNTING - no smoothing
            if result['is_focused']:
                focused_count += 1
            else:
                unfocused_count += 1

        total_frames = len(image_files)

        if total_frames == 0:
            logger.warning(f"⚠️  No valid frames for {student_id}")
            return None

        # Calculate attentiveness percentage - DIRECT from counts
        attentiveness_percentage = (focused_count / total_frames) * 100

        # Determine status
        if attentiveness_percentage >= (self.focus_threshold * 100):
            status = "Focused"
            status_color = "success"
        else:
            status = "Not Focused"
            status_color = "danger"

        # Calculate statistics
        pose_based_count = method_counts['head_pose']
        quality_fallback_count = method_counts['image_quality_fallback']
        failed_count = method_counts['failed']

        pose_reliability = pose_based_count / total_frames if total_frames > 0 else 0

        avg_yaw = np.mean(pose_angles['yaw']) if pose_angles['yaw'] else None
        avg_pitch = np.mean(pose_angles['pitch']) if pose_angles['pitch'] else None
        avg_roll = np.mean(pose_angles['roll']) if pose_angles['roll'] else None

        std_yaw = np.std(pose_angles['yaw']) if pose_angles['yaw'] else None
        std_pitch = np.std(pose_angles['pitch']) if pose_angles['pitch'] else None

        # Logging
        logger.info(f"   Focused: {focused_count}/{total_frames} ({attentiveness_percentage:.1f}%)")
        logger.info(f"   Status: {status}")
        logger.info(f"   Method: {pose_based_count} pose, {quality_fallback_count} fallback, {failed_count} failed")
        logger.info(f"   Pose reliability: {pose_reliability:.1%}")
        if avg_yaw is not None:
            logger.info(f"   Avg pose: yaw={avg_yaw:.1f}°, pitch={avg_pitch:.1f}°, roll={avg_roll:.1f}°")

        sample_image = image_files[len(image_files) // 2] if image_files else None

        return {
            'student_id': student_id,
            'status': status,
            'status_color': status_color,
            'attentiveness_percentage': round(attentiveness_percentage, 1),
            'engagement_percentage': round(attentiveness_percentage, 1),
            'engagement_score': round(attentiveness_percentage, 1),
            'images_analyzed': total_frames,
            'valid_frames': total_frames,
            'focused_frames': focused_count,
            'unfocused_frames': unfocused_count,
            'focused_count': focused_count,
            'unfocused_count': unfocused_count,
            'method_breakdown': {
                'head_pose_count': pose_based_count,
                'fallback_count': quality_fallback_count,
                'failed_count': failed_count,
                'pose_reliability': round(pose_reliability, 3)
            },
            'head_pose_stats': {
                'avg_yaw': round(avg_yaw, 2) if avg_yaw is not None else None,
                'avg_pitch': round(avg_pitch, 2) if avg_pitch is not None else None,
                'avg_roll': round(avg_roll, 2) if avg_roll is not None else None,
                'std_yaw': round(std_yaw, 2) if std_yaw is not None else None,
                'std_pitch': round(std_pitch, 2) if std_pitch is not None else None
            },
            'sample_image': sample_image,
            'most_common_state': status,
            'prediction_breakdown': {
                'Focused': {
                    'count': focused_count,
                    'percentage': round((focused_count / total_frames) * 100, 1)
                },
                'Not Focused': {
                    'count': unfocused_count,
                    'percentage': round((unfocused_count / total_frames) * 100, 1)
                }
            }
        }

    def analyze_all_students(self, progress_callback=None) -> List[Dict]:
        """Analyze all student folders"""
        logger.info("=" * 60)
        logger.info("📂 Checking students directory...")

        if not os.path.exists(self.students_dir):
            logger.error(f"❌ Students directory not found: {self.students_dir}")
            return []

        student_folders = [
            os.path.join(self.students_dir, d)
            for d in os.listdir(self.students_dir)
            if os.path.isdir(os.path.join(self.students_dir, d))
            and d.startswith('student_')
        ]

        if len(student_folders) == 0:
            logger.warning("⚠️  No student folders found")
            return []

        logger.info(f"✅ Found {len(student_folders)} student folders")

        all_analytics = []

        for idx, folder in enumerate(tqdm(student_folders, desc="Analyzing students")):
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 Analyzing {os.path.basename(folder)} ({idx+1}/{len(student_folders)})")
            logger.info(f"{'='*60}")

            try:
                analytics = self.analyze_student_folder(folder)
                if analytics:
                    all_analytics.append(analytics)
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

        logger.info(f"\n✅ Analyzed {len(all_analytics)}/{len(student_folders)} students\n")
        return all_analytics

    def compute_classroom_summary(self, all_analytics: List[Dict]) -> Dict:
        """Compute classroom summary"""
        if not all_analytics:
            return {
                'total_students': 0,
                'focused_students': 0,
                'moderately_focused_students': 0,
                'unfocused_students': 0,
                'average_attentiveness': 0,
                'average_engagement': 0,
                'total_images_analyzed': 0,
                'avg_pose_reliability': 0
            }

        status_counts = defaultdict(int)
        for a in all_analytics:
            status_counts[a['status']] += 1

        avg_attentiveness = sum(a['attentiveness_percentage'] for a in all_analytics) / len(all_analytics)
        total_images = sum(a['images_analyzed'] for a in all_analytics)

        # Calculate average pose reliability
        avg_pose_reliability = np.mean([
            a['method_breakdown']['pose_reliability']
            for a in all_analytics
        ])

        return {
            'total_students': len(all_analytics),
            'focused_students': status_counts['Focused'],
            'moderately_focused_students': 0,
            'unfocused_students': status_counts['Not Focused'],
            'average_attentiveness': round(avg_attentiveness, 1),
            'average_engagement': round(avg_attentiveness, 1),
            'total_images_analyzed': total_images,
            'focused_percentage': round((status_counts['Focused'] / len(all_analytics)) * 100, 1),
            'unfocused_percentage': round((status_counts['Not Focused'] / len(all_analytics)) * 100, 1),
            'avg_pose_reliability': round(avg_pose_reliability, 3)
        }

    def generate_report(self, output_file='attentiveness_report.json') -> Dict:
        """Generate report"""
        logger.info("Generating attentiveness report...")

        all_analytics = self.analyze_all_students()
        summary = self.compute_classroom_summary(all_analytics)

        report = {
            'summary': summary,
            'students': all_analytics,
            'analysis_type': 'hybrid_head_pose_attentiveness',
            'configuration': {
                'yaw_threshold': self.yaw_threshold,
                'pitch_threshold': self.pitch_threshold,
                'roll_threshold': self.roll_threshold,
                'focus_threshold': self.focus_threshold,
                'consecutive_distraction_threshold': self.consecutive_distraction_threshold
            }
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {output_file}")
        return report

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'detector'):
            self.detector.close() # Updated reference
