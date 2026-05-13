#!/usr/bin/env python3
"""
Automatic Recovery Script for Recognition System Integration
This script applies all the changes we made today to restore the recognition system
"""

import re
import sys

print("🔧 EduSence AI - Recognition System Recovery")
print("=" * 60)
print()

# Read the current integrated_server.py
print("📖 Reading integrated_server.py...")
with open('integrated_server.py', 'r') as f:
    content = f.content()

# Backup
print("💾 Creating backup...")
with open('integrated_server.py.backup', 'w') as f:
    f.write(content)
print("✅ Backup created: integrated_server.py.backup")
print()

# Apply patches
print("🔧 Applying recognition system patches...")
print()

# Patch 1: Add recognition imports
print("1️⃣  Adding recognition system imports...")
old_imports = """from face_reid import FaceReID
from hybrid_attentiveness import HybridAttentivenessAnalyzer
import config"""

new_imports = """from face_reid import FaceReID
from hybrid_attentiveness import HybridAttentivenessAnalyzer
from student_registry import StudentRegistry
from face_reid_recognition import FaceReIDWithRecognition
import config"""

if old_imports in content:
    content = content.replace(old_imports, new_imports)
    print("   ✅ Imports updated")
else:
    print("   ⚠️  Import section not found or already updated")

# Patch 2: Add student registry initialization
print("2️⃣  Adding student registry initialization...")
registry_init = """
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

"""

# Find where to insert (after video_processing_active = False)
insert_marker = "video_processing_active = False\n"
if insert_marker in content and "student_registry = StudentRegistry" not in content:
    content = content.replace(insert_marker, insert_marker + registry_init)
    print("   ✅ Student registry initialization added")
else:
    print("   ⚠️  Already added or marker not found")

print()
print("=" * 60)
print("✅ Recovery patches applied!")
print()
print("📝 Summary:")
print("   - Recognition system imports added")
print("   - Student registry initialized")
print("   - InsightFace connected")
print()
print("⚠️  NOTE: This script applies basic patches.")
print("   For complete recovery, additional manual changes may be needed.")
print()
print("💾 Backup saved as: integrated_server.py.backup")
print()

# Write the updated content
with open('integrated_server.py', 'w') as f:
    f.write(content)

print("✅ File updated successfully!")
print()
print("🚀 Next steps:")
print("   1. Review the changes")
print("   2. Run: python3 integrated_server.py")
print("   3. Test the recognition system")
