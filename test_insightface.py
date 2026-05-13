"""
Test InsightFace initialization
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("=" * 60)
    logger.info("Testing InsightFace initialization...")
    logger.info("=" * 60)
    
    import insightface
    from insightface.app import FaceAnalysis
    
    logger.info(f"✅ InsightFace imported: version {insightface.__version__}")
    
    logger.info("🔄 Creating FaceAnalysis app...")
    face_app = FaceAnalysis(
        name='buffalo_s',
        providers=['CPUExecutionProvider']
    )
    
    logger.info("🔄 Preparing model...")
    face_app.prepare(ctx_id=0, det_size=(160, 160))
    
    logger.info("✅ InsightFace initialized successfully!")
    logger.info(f"   Models loaded: {len(face_app.models)}")
    
    # Test with a dummy image
    import numpy as np
    import cv2
    
    logger.info("🔄 Testing with dummy image...")
    dummy_img = np.zeros((160, 160, 3), dtype=np.uint8)
    faces = face_app.get(dummy_img)
    logger.info(f"✅ Test complete: {len(faces)} faces detected (expected 0)")
    
    logger.info("=" * 60)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("=" * 60)
    
except Exception as e:
    logger.error(f"❌ InsightFace initialization FAILED: {e}")
    import traceback
    traceback.print_exc()
