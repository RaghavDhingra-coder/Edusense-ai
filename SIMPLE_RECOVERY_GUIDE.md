# Simple Recovery Guide

## ✅ What's Already Safe

These files still have all our recognition code:
- ✅ `student_registry.py` - Complete
- ✅ `face_reid_recognition.py` - Complete  
- ✅ `hybrid_attentiveness.py` - Has the unknown person filter

## 🔧 What Needs to Be Fixed

Only `integrated_server.py` needs to be updated with recognition integration.

## 📝 Quick Fix - Add These Lines

### Step 1: Update Imports (Line ~27)

Add after `from hybrid_attentiveness import HybridAttentivenessAnalyzer`:
```python
from student_registry import StudentRegistry
from face_reid_recognition import FaceReIDWithRecognition
```

### Step 2: Initialize Student Registry (Line ~50, after `UPLOAD_FOLDER` setup)

Add:
```python
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
```

## 💡 Alternative: Use Kiro to Reapply Changes

Since you're using Kiro AI assistant, you can ask me to:

1. Read the current `integrated_server.py`
2. Apply all the recognition system changes
3. Save the updated file

This is the fastest and safest method!

## 🚀 Recommended Action

**Ask Kiro**: "Please update integrated_server.py to add the complete recognition system integration that we implemented earlier today"

I have all the changes in my context and can reapply them accurately.
