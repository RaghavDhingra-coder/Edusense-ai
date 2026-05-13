"""
Post-processing utility for student folder cleanup
Detects and fixes:
1. Duplicate folders (same student with multiple IDs)
2. Mixed-identity folders (multiple students in one folder)
3. Low-quality folders
"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import shutil
import json


class FolderCleanup:
    """
    Post-processing system for cleaning up student folders
    """
    
    def __init__(self, 
                 students_dir="students",
                 merge_threshold=0.65,      # Similarity for merging folders
                 split_threshold=0.45,      # Similarity for splitting folders
                 min_images_per_student=3): # Minimum images to keep folder
        """
        Initialize folder cleanup system
        
        Args:
            students_dir: Directory containing student folders
            merge_threshold: Cosine similarity threshold for merging
            split_threshold: Threshold for detecting mixed identities
            min_images_per_student: Minimum images to keep a folder
        """
        self.students_dir = students_dir
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold
        self.min_images_per_student = min_images_per_student
        
        # Initialize InsightFace
        self.app = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize InsightFace for embedding extraction"""
        try:
            print("🔄 Loading InsightFace for folder cleanup...")
            
            import insightface
            from insightface.app import FaceAnalysis
            
            self.app = FaceAnalysis(
                name='buffalo_s',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_size=(160, 160))
            
            print("✅ InsightFace loaded for cleanup")
            
        except Exception as e:
            print(f"❌ Failed to load InsightFace: {str(e)}")
            self.app = None
    
    def extract_embedding(self, image_path):
        """Extract embedding from image file"""
        if self.app is None:
            return None
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Resize if needed
            if img.shape[0] > 160 or img.shape[1] > 160:
                img = cv2.resize(img, (160, 160))
            
            faces = self.app.get(img)
            
            if len(faces) > 0:
                return faces[0].embedding
            
            return None
            
        except Exception as e:
            print(f"⚠️  Error extracting embedding from {image_path}: {str(e)}")
            return None
    
    def get_folder_embeddings(self, folder_path, max_images=20):
        """
        Extract embeddings from all images in a folder
        
        Args:
            folder_path: Path to student folder
            max_images: Maximum images to process (for speed)
            
        Returns:
            List of embeddings
        """
        embeddings = []
        image_files = [f for f in os.listdir(folder_path) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sample images if too many
        if len(image_files) > max_images:
            import random
            image_files = random.sample(image_files, max_images)
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            embedding = self.extract_embedding(img_path)
            
            if embedding is not None:
                embeddings.append(embedding)
        
        return embeddings
    
    def calculate_folder_similarity(self, folder1_embeddings, folder2_embeddings):
        """
        Calculate similarity between two folders
        Uses average of pairwise similarities
        
        Returns:
            Average similarity score (0-1)
        """
        if len(folder1_embeddings) == 0 or len(folder2_embeddings) == 0:
            return 0.0
        
        # Calculate average embeddings
        avg_emb1 = np.mean(folder1_embeddings, axis=0)
        avg_emb2 = np.mean(folder2_embeddings, axis=0)
        
        # Cosine similarity
        similarity = cosine_similarity(
            avg_emb1.reshape(1, -1),
            avg_emb2.reshape(1, -1)
        )[0][0]
        
        return similarity
    
    def detect_duplicate_folders(self):
        """
        Detect folders that belong to the same student
        
        Returns:
            List of duplicate groups: [[folder1, folder2], [folder3, folder4], ...]
        """
        print("\n" + "="*60)
        print("🔍 DETECTING DUPLICATE FOLDERS")
        print("="*60)
        
        if not os.path.exists(self.students_dir):
            print("❌ Students directory not found")
            return []
        
        # Get all student folders
        folders = [f for f in os.listdir(self.students_dir) 
                  if os.path.isdir(os.path.join(self.students_dir, f)) 
                  and f.startswith('student_')]
        
        if len(folders) < 2:
            print("ℹ️  Less than 2 folders, no duplicates possible")
            return []
        
        print(f"📁 Found {len(folders)} student folders")
        print("🔄 Extracting embeddings...")
        
        # Extract embeddings for each folder
        folder_embeddings = {}
        for folder in folders:
            folder_path = os.path.join(self.students_dir, folder)
            embeddings = self.get_folder_embeddings(folder_path)
            
            if len(embeddings) > 0:
                folder_embeddings[folder] = embeddings
                print(f"   {folder}: {len(embeddings)} embeddings")
        
        # Compare all pairs
        print("\n🔄 Comparing folders...")
        duplicate_groups = []
        processed = set()
        
        folder_list = list(folder_embeddings.keys())
        
        for i, folder1 in enumerate(folder_list):
            if folder1 in processed:
                continue
            
            group = [folder1]
            
            for folder2 in folder_list[i+1:]:
                if folder2 in processed:
                    continue
                
                similarity = self.calculate_folder_similarity(
                    folder_embeddings[folder1],
                    folder_embeddings[folder2]
                )
                
                print(f"   {folder1} vs {folder2}: {similarity:.3f}")
                
                if similarity >= self.merge_threshold:
                    group.append(folder2)
                    processed.add(folder2)
            
            if len(group) > 1:
                duplicate_groups.append(group)
                processed.add(folder1)
        
        print(f"\n✅ Found {len(duplicate_groups)} duplicate groups")
        for i, group in enumerate(duplicate_groups):
            print(f"   Group {i+1}: {group}")
        
        return duplicate_groups
    
    def detect_mixed_identity_folders(self):
        """
        Detect folders containing multiple different students
        Uses DBSCAN clustering on embeddings
        
        Returns:
            Dict {folder_name: num_clusters}
        """
        print("\n" + "="*60)
        print("🔍 DETECTING MIXED-IDENTITY FOLDERS")
        print("="*60)
        
        if not os.path.exists(self.students_dir):
            print("❌ Students directory not found")
            return {}
        
        folders = [f for f in os.listdir(self.students_dir) 
                  if os.path.isdir(os.path.join(self.students_dir, f)) 
                  and f.startswith('student_')]
        
        mixed_folders = {}
        
        for folder in folders:
            folder_path = os.path.join(self.students_dir, folder)
            embeddings = self.get_folder_embeddings(folder_path, max_images=50)
            
            if len(embeddings) < 5:  # Need enough samples for clustering
                continue
            
            # Cluster embeddings using DBSCAN
            embeddings_array = np.array(embeddings)
            
            # DBSCAN parameters
            # eps: maximum distance between samples
            # min_samples: minimum samples in a cluster
            clustering = DBSCAN(
                eps=0.6,  # Adjust based on embedding space
                min_samples=2,
                metric='cosine'
            ).fit(embeddings_array)
            
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            print(f"   {folder}: {n_clusters} clusters, {n_noise} noise points")
            
            # If multiple clusters found, it's likely mixed identity
            if n_clusters > 1:
                mixed_folders[folder] = n_clusters
        
        print(f"\n✅ Found {len(mixed_folders)} mixed-identity folders")
        for folder, n_clusters in mixed_folders.items():
            print(f"   {folder}: {n_clusters} identities detected")
        
        return mixed_folders
    
    def merge_duplicate_folders(self, duplicate_groups, dry_run=True):
        """
        Merge duplicate folders into primary folder
        
        Args:
            duplicate_groups: List of duplicate groups from detect_duplicate_folders()
            dry_run: If True, only simulate (don't actually merge)
        """
        print("\n" + "="*60)
        print("🔄 MERGING DUPLICATE FOLDERS")
        print("="*60)
        
        if dry_run:
            print("⚠️  DRY RUN MODE - No actual changes will be made")
        
        for i, group in enumerate(duplicate_groups):
            print(f"\n📦 Group {i+1}: {group}")
            
            # Choose primary folder (one with most images)
            folder_sizes = {}
            for folder in group:
                folder_path = os.path.join(self.students_dir, folder)
                num_images = len([f for f in os.listdir(folder_path) 
                                 if f.endswith(('.jpg', '.jpeg', '.png'))])
                folder_sizes[folder] = num_images
            
            primary_folder = max(folder_sizes, key=folder_sizes.get)
            print(f"   Primary: {primary_folder} ({folder_sizes[primary_folder]} images)")
            
            # Merge others into primary
            for folder in group:
                if folder == primary_folder:
                    continue
                
                print(f"   Merging {folder} → {primary_folder}")
                
                if not dry_run:
                    src_path = os.path.join(self.students_dir, folder)
                    dst_path = os.path.join(self.students_dir, primary_folder)
                    
                    # Copy all images
                    for img_file in os.listdir(src_path):
                        if img_file.endswith(('.jpg', '.jpeg', '.png')):
                            src_file = os.path.join(src_path, img_file)
                            dst_file = os.path.join(dst_path, f"{folder}_{img_file}")
                            shutil.copy2(src_file, dst_file)
                    
                    # Remove old folder
                    shutil.rmtree(src_path)
                    print(f"   ✅ Merged and removed {folder}")
        
        if dry_run:
            print("\n⚠️  This was a DRY RUN. Run with dry_run=False to apply changes.")
    
    def split_mixed_folder(self, folder_name, dry_run=True):
        """
        Split a mixed-identity folder into separate folders
        
        Args:
            folder_name: Name of folder to split
            dry_run: If True, only simulate
        """
        print(f"\n🔄 Splitting mixed folder: {folder_name}")
        
        if dry_run:
            print("⚠️  DRY RUN MODE")
        
        folder_path = os.path.join(self.students_dir, folder_name)
        
        # Get all embeddings with filenames
        image_files = [f for f in os.listdir(folder_path) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        embeddings = []
        valid_files = []
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            embedding = self.extract_embedding(img_path)
            
            if embedding is not None:
                embeddings.append(embedding)
                valid_files.append(img_file)
        
        if len(embeddings) < 5:
            print("   ⚠️  Not enough valid embeddings for splitting")
            return
        
        # Cluster
        embeddings_array = np.array(embeddings)
        clustering = DBSCAN(eps=0.6, min_samples=2, metric='cosine').fit(embeddings_array)
        labels = clustering.labels_
        
        # Group files by cluster
        clusters = defaultdict(list)
        for img_file, label in zip(valid_files, labels):
            if label != -1:  # Ignore noise
                clusters[label].append(img_file)
        
        print(f"   Found {len(clusters)} clusters")
        
        if len(clusters) <= 1:
            print("   ℹ️  Only one cluster found, no split needed")
            return
        
        # Create new folders for each cluster
        if not dry_run:
            for cluster_id, files in clusters.items():
                new_folder = f"{folder_name}_split_{cluster_id}"
                new_folder_path = os.path.join(self.students_dir, new_folder)
                os.makedirs(new_folder_path, exist_ok=True)
                
                for img_file in files:
                    src = os.path.join(folder_path, img_file)
                    dst = os.path.join(new_folder_path, img_file)
                    shutil.copy2(src, dst)
                
                print(f"   ✅ Created {new_folder} with {len(files)} images")
            
            # Remove original folder
            shutil.rmtree(folder_path)
            print(f"   ✅ Removed original folder {folder_name}")
        else:
            for cluster_id, files in clusters.items():
                print(f"   Would create {folder_name}_split_{cluster_id} with {len(files)} images")
    
    def remove_low_quality_folders(self, dry_run=True):
        """Remove folders with too few images"""
        print("\n" + "="*60)
        print("🔄 REMOVING LOW-QUALITY FOLDERS")
        print("="*60)
        
        if dry_run:
            print("⚠️  DRY RUN MODE")
        
        folders = [f for f in os.listdir(self.students_dir) 
                  if os.path.isdir(os.path.join(self.students_dir, f)) 
                  and f.startswith('student_')]
        
        removed_count = 0
        
        for folder in folders:
            folder_path = os.path.join(self.students_dir, folder)
            num_images = len([f for f in os.listdir(folder_path) 
                             if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            if num_images < self.min_images_per_student:
                print(f"   {folder}: {num_images} images (below threshold)")
                
                if not dry_run:
                    shutil.rmtree(folder_path)
                    print(f"   ✅ Removed {folder}")
                else:
                    print(f"   Would remove {folder}")
                
                removed_count += 1
        
        print(f"\n✅ {removed_count} folders marked for removal")
    
    def generate_cleanup_report(self, output_file="cleanup_report.json"):
        """Generate comprehensive cleanup report"""
        print("\n" + "="*60)
        print("📊 GENERATING CLEANUP REPORT")
        print("="*60)
        
        report = {
            'timestamp': str(Path(self.students_dir).stat().st_mtime),
            'total_folders': 0,
            'duplicate_groups': [],
            'mixed_folders': {},
            'low_quality_folders': [],
            'recommendations': []
        }
        
        # Count folders
        folders = [f for f in os.listdir(self.students_dir) 
                  if os.path.isdir(os.path.join(self.students_dir, f)) 
                  and f.startswith('student_')]
        report['total_folders'] = len(folders)
        
        # Detect duplicates
        duplicate_groups = self.detect_duplicate_folders()
        report['duplicate_groups'] = duplicate_groups
        
        # Detect mixed identities
        mixed_folders = self.detect_mixed_identity_folders()
        report['mixed_folders'] = mixed_folders
        
        # Find low quality
        for folder in folders:
            folder_path = os.path.join(self.students_dir, folder)
            num_images = len([f for f in os.listdir(folder_path) 
                             if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            if num_images < self.min_images_per_student:
                report['low_quality_folders'].append({
                    'folder': folder,
                    'num_images': num_images
                })
        
        # Generate recommendations
        if len(duplicate_groups) > 0:
            report['recommendations'].append(
                f"Merge {len(duplicate_groups)} duplicate groups to reduce redundancy"
            )
        
        if len(mixed_folders) > 0:
            report['recommendations'].append(
                f"Split {len(mixed_folders)} mixed-identity folders for accuracy"
            )
        
        if len(report['low_quality_folders']) > 0:
            report['recommendations'].append(
                f"Remove {len(report['low_quality_folders'])} low-quality folders"
            )
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Report saved to {output_file}")
        print("\n📊 SUMMARY:")
        print(f"   Total folders: {report['total_folders']}")
        print(f"   Duplicate groups: {len(duplicate_groups)}")
        print(f"   Mixed-identity folders: {len(mixed_folders)}")
        print(f"   Low-quality folders: {len(report['low_quality_folders'])}")
        
        return report


def main():
    """Main cleanup utility"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Student Folder Cleanup Utility")
    parser.add_argument('--dir', default='students', help='Students directory')
    parser.add_argument('--action', choices=['report', 'merge', 'split', 'clean'], 
                       default='report', help='Action to perform')
    parser.add_argument('--dry-run', action='store_true', help='Simulate without making changes')
    parser.add_argument('--folder', help='Specific folder to split (for split action)')
    
    args = parser.parse_args()
    
    cleanup = FolderCleanup(students_dir=args.dir)
    
    if args.action == 'report':
        cleanup.generate_cleanup_report()
    
    elif args.action == 'merge':
        duplicates = cleanup.detect_duplicate_folders()
        if len(duplicates) > 0:
            cleanup.merge_duplicate_folders(duplicates, dry_run=args.dry_run)
        else:
            print("No duplicates found")
    
    elif args.action == 'split':
        if args.folder:
            cleanup.split_mixed_folder(args.folder, dry_run=args.dry_run)
        else:
            mixed = cleanup.detect_mixed_identity_folders()
            for folder in mixed.keys():
                cleanup.split_mixed_folder(folder, dry_run=args.dry_run)
    
    elif args.action == 'clean':
        cleanup.remove_low_quality_folders(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
