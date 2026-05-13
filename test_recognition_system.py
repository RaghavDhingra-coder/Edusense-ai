#!/usr/bin/env python3
"""
Test script to verify the recognition system integration
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported"""
    logger.info("=" * 60)
    logger.info("TEST 1: Module Imports")
    logger.info("=" * 60)
    
    try:
        from student_registry import StudentRegistry
        logger.info("✅ StudentRegistry imported")
        
        from face_reid_recognition import FaceReIDWithRecognition
        logger.info("✅ FaceReIDWithRecognition imported")
        
        import integrated_server
        logger.info("✅ integrated_server imported")
        
        logger.info("✅ All imports successful\n")
        return True
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_registry_creation():
    """Test creating a student registry"""
    logger.info("=" * 60)
    logger.info("TEST 2: Student Registry Creation")
    logger.info("=" * 60)
    
    try:
        from student_registry import StudentRegistry
        
        registry = StudentRegistry(registry_dir='test_registry')
        logger.info(f"✅ Registry created: {len(registry.students)} students")
        
        # Clean up
        import shutil
        import os
        if os.path.exists('test_registry'):
            shutil.rmtree('test_registry')
            logger.info("✅ Test registry cleaned up")
        
        logger.info("✅ Registry creation test passed\n")
        return True
    except Exception as e:
        logger.error(f"❌ Registry creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reid_creation():
    """Test creating ReID with recognition"""
    logger.info("=" * 60)
    logger.info("TEST 3: ReID with Recognition Creation")
    logger.info("=" * 60)
    
    try:
        from student_registry import StudentRegistry
        from face_reid_recognition import FaceReIDWithRecognition
        
        registry = StudentRegistry(registry_dir='test_registry')
        reid = FaceReIDWithRecognition(
            student_registry=registry,
            similarity_threshold=0.6,
            embedding_size=512
        )
        
        logger.info("✅ ReID system created")
        logger.info(f"   Registered students: {len(registry.students)}")
        logger.info(f"   Cache duration: {reid.cache_duration}s")
        logger.info(f"   Recognition interval: {reid.recognition_interval} frames")
        
        # Clean up
        import shutil
        import os
        if os.path.exists('test_registry'):
            shutil.rmtree('test_registry')
        
        logger.info("✅ ReID creation test passed\n")
        return True
    except Exception as e:
        logger.error(f"❌ ReID creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoints():
    """Test that all API endpoints are defined"""
    logger.info("=" * 60)
    logger.info("TEST 4: API Endpoints")
    logger.info("=" * 60)
    
    try:
        import integrated_server
        
        required_endpoints = [
            '/api/students/register',
            '/api/students/list',
            '/api/students/delete/<student_id>',
            '/api/students/capture',
            '/api/camera/start',
            '/api/camera/stop',
            '/api/video/upload',
            '/api/video/process',
            '/api/analyze'
        ]
        
        # Get all registered routes
        app = integrated_server.app
        routes = [str(rule) for rule in app.url_map.iter_rules()]
        
        logger.info(f"Total routes registered: {len(routes)}")
        
        missing = []
        for endpoint in required_endpoints:
            # Check if endpoint exists (handle dynamic routes)
            endpoint_base = endpoint.split('<')[0]
            found = any(endpoint_base in route for route in routes)
            
            if found:
                logger.info(f"   ✅ {endpoint}")
            else:
                logger.info(f"   ❌ {endpoint}")
                missing.append(endpoint)
        
        if missing:
            logger.error(f"❌ Missing endpoints: {missing}")
            return False
        
        logger.info("✅ All API endpoints registered\n")
        return True
    except Exception as e:
        logger.error(f"❌ API endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frontend_files():
    """Test that frontend files exist"""
    logger.info("=" * 60)
    logger.info("TEST 5: Frontend Files")
    logger.info("=" * 60)
    
    try:
        import os
        
        required_files = [
            'frontend/index_integrated.html',
            'frontend/app_integrated.js',
            'frontend/register.html',
            'frontend/register.js',
            'frontend/styles.css'
        ]
        
        missing = []
        for file in required_files:
            if os.path.exists(file):
                logger.info(f"   ✅ {file}")
            else:
                logger.info(f"   ❌ {file}")
                missing.append(file)
        
        if missing:
            logger.error(f"❌ Missing files: {missing}")
            return False
        
        logger.info("✅ All frontend files exist\n")
        return True
    except Exception as e:
        logger.error(f"❌ Frontend files test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n")
    logger.info("🧪 RECOGNITION SYSTEM INTEGRATION TEST")
    logger.info("=" * 60)
    logger.info("\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Registry Creation", test_registry_creation),
        ("ReID Creation", test_reid_creation),
        ("API Endpoints", test_api_endpoints),
        ("Frontend Files", test_frontend_files)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {name}")
    
    logger.info("=" * 60)
    logger.info(f"TOTAL: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("\nNext steps:")
        logger.info("1. Start server: python3 integrated_server.py")
        logger.info("2. Open dashboard: http://localhost:8080")
        logger.info("3. Register students: http://localhost:8080/register.html")
        logger.info("4. Start webcam and verify recognition works")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
