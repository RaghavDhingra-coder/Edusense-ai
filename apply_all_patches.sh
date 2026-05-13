#!/bin/bash

echo "🔧 Applying ALL recognition system patches..."
echo "=============================================="
echo ""

# Apply all patches using the Python scripts we created earlier
python3 << 'EOF'
import re

with open('integrated_server.py', 'r') as f:
    content = f.read()

patches_applied = 0

# Patch: _initialize_components
if 'FaceReIDWithRecognition' not in content:
    old = """            # ReID system - SAME AS main.py
            # NOTE: ReID will use persistent identity database if it exists
            if config.ENABLE_REID:
                logger.info("🔄 Initializing Face Re-Identification...")
                self.reid_system = FaceReID(
                    similarity_threshold=config.REID_SIMILARITY_THRESHOLD,
                    embedding_size=512
                )
                # Reset ReID session state (but keep persistent identity DB)
                self.reid_system.reset_session()
            else:
                logger.info("⚠️  Face ReID disabled")
                self.reid_system = None"""
    
    new = """            # ReID system with recognition support
            if config.ENABLE_REID:
                logger.info("🔄 Initializing Face Recognition System...")
                
                # Use recognition-enhanced ReID
                self.reid_system = FaceReIDWithRecognition(
                    student_registry=student_registry,
                    similarity_threshold=config.REID_SIMILARITY_THRESHOLD,
                    embedding_size=512
                )
                
                # Initialize InsightFace for recognition
                try:
                    import insightface
                    from insightface.app import FaceAnalysis
                    
                    face_app = FaceAnalysis(
                        name='buffalo_s',
                        providers=['CPUExecutionProvider']
                    )
                    face_app.prepare(ctx_id=0, det_size=(160, 160))
                    
                    # Connect to ReID system
                    self.reid_system.set_face_app(face_app)
                    logger.info("✅ Face recognition enabled")
                    
                except Exception as e:
                    logger.error(f"⚠️  InsightFace initialization failed: {e}")
                    logger.info("   Falling back to tracking-only mode")
                    # Fall back to basic ReID
                    self.reid_system = FaceReID(
                        similarity_threshold=config.REID_SIMILARITY_THRESHOLD,
                        embedding_size=512
                    )
                
                # Reset ReID session state
                self.reid_system.reset_session()
            else:
                logger.info("⚠️  Face ReID disabled")
                self.reid_system = None"""
    
    if old in content:
        content = content.replace(old, new)
        patches_applied += 1
        print("✅ Patch 1: _initialize_components")

with open('integrated_server.py', 'w') as f:
    f.write(content)

print(f"\n✅ Applied {patches_applied} patches")
print("⚠️  Note: Some patches require manual review")
print("\nRun: python3 integrated_server.py")
EOF

echo ""
echo "=============================================="
echo "✅ Patch script complete!"
echo ""
echo "To complete recovery, you need to:"
echo "1. Check if server starts: python3 integrated_server.py"
echo "2. If errors, let me know and I'll fix them"
echo ""
