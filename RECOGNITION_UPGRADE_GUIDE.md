# 🎯 Student Recognition System - Implementation Guide

## Overview

This guide explains how to upgrade the EduSence AI system from tracking-based IDs to registered student face recognition.

## ✅ What's Been Created

### 1. Student Registry System (`student_registry.py`)
- Manages registered students with face embeddings
- Stores student profiles with name, ID, and multiple face embeddings
- Fast recognition using cosine similarity
- Persistent storage in `registered_students/` directory

### 2. Enhanced ReID (`face_reid_recognition.py`)
- Combines tracking with recognition
- Recognizes registered students automatically
- Falls back to "Unknown" for unregistered persons
- Optimized with caching to maintain performance

## 🚀 Implementation Steps

### Step 1: Update CameraSystem to Use Recognition

In `integrated_server.py`, modify `_initialize_components()`:

```python
def _initialize_components(self):
    """Initialize all components with recognition support"""
    try:
        logger.info("📦 Initializing components...")
        
        # Face detector
        self.detector = FaceDetector()
        
        # Image manager
        self.image_manager = ImageManager(self.session_dir)
        
        # ReID system with recognition
        if config.ENABLE_REID:
            logger.info("🔄 Initializing Face Recognition...")
            self.reid_system = FaceReIDWithRecognition(
                student_registry=student_registry,
                similarity_threshold=config.REID_SIMILARITY_THRESHOLD,
                embedding_size=512
            )
            # Set InsightFace app
            import insightface
            from insightface.app import FaceAnalysis
            face_app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=0, det_size=(160, 160))
            self.reid_system.set_face_app(face_app)
            
            # Reset session state
            self.reid_system.reset_session()
        else:
            logger.info("⚠️  Face ReID disabled")
            self.reid_system = None
        
        logger.info("✅ Components initialized")
```

### Step 2: Update Frame Processing

In `_process_frame()`, change the ReID call:

```python
# OLD:
student_id, is_new, similarity = self.reid_system.register_or_match_face(
    track_id, face_crop, current_time, force_extract=should_extract
)

# NEW:
identity_id, identity_name, is_new, confidence, is_registered = self.reid_system.register_or_match_face(
    track_id, face_crop, current_time, force_extract=should_extract
)

# Use identity_id instead of student_id
# Use identity_name for display
```

### Step 3: Update Drawing Function

In `_draw_detections()`, show student names:

```python
def _draw_detections(self, frame, detections):
    """Draw bounding boxes with student names"""
    for x1, y1, x2, y2, identity_id, identity_name, confidence, is_registered in detections:
        # Color based on registration status
        color = (0, 255, 0) if is_registered else (255, 165, 0)  # Green for registered, Orange for unknown
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label with name
        label = identity_name
        if is_registered:
            label += f" ({confidence:.0%})"
        
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Background for label
        cv2.rectangle(frame, 
                     (int(x1), int(y1) - label_size[1] - 10),
                     (int(x1) + label_size[0], int(y1)),
                     color, -1)
        
        # Label text
        cv2.putText(frame, label, (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame
```

### Step 4: Add Registration API Endpoints

Add these endpoints to `integrated_server.py`:

```python
@app.route('/api/students/register', methods=['POST'])
def register_student():
    """Register a new student with face images"""
    try:
        data = request.get_json()
        
        student_name = data.get('student_name')
        student_id = data.get('student_id')
        
        # Get face images (base64 encoded)
        face_images_b64 = data.get('face_images', [])
        
        # Decode images
        face_images = []
        for img_b64 in face_images_b64:
            import base64
            img_data = base64.b64decode(img_b64.split(',')[1])  # Remove data:image/jpeg;base64,
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            face_images.append(img)
        
        # Register student
        result = student_registry.register_student(student_name, student_id, face_images)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/students/list', methods=['GET'])
def list_students():
    """Get list of all registered students"""
    try:
        students = student_registry.get_all_students()
        return jsonify({
            'success': True,
            'students': students,
            'count': len(students)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/students/delete/<student_id>', methods=['DELETE'])
def delete_student(student_id):
    """Delete a registered student"""
    try:
        result = student_registry.delete_student(student_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/students/capture', methods=['POST'])
def capture_faces_from_camera():
    """Capture face images from current camera feed for registration"""
    try:
        global camera_system
        
        if not camera_system or not camera_system.running:
            return jsonify({'success': False, 'error': 'Camera not running'}), 400
        
        # Capture N frames
        num_captures = request.json.get('num_captures', 10)
        captured_images = []
        
        for i in range(num_captures):
            frame = camera_system.get_frame()
            if frame is not None:
                # Detect face
                detections = camera_system.detector.detect_faces(frame)
                if len(detections) > 0:
                    # Get first face
                    x1, y1, x2, y2, conf = detections[0]
                    face_crop = camera_system.image_manager.crop_face(frame, (x1, y1, x2, y2))
                    if face_crop is not None:
                        # Encode to base64
                        import base64
                        _, buffer = cv2.imencode('.jpg', face_crop)
                        img_b64 = base64.b64encode(buffer).decode('utf-8')
                        captured_images.append(f"data:image/jpeg;base64,{img_b64}")
            
            time.sleep(0.5)  # Wait between captures
        
        return jsonify({
            'success': True,
            'images': captured_images,
            'count': len(captured_images)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
```

### Step 5: Update Frontend

Create `frontend/register.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Student Registration</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>Register Student</h1>
        
        <form id="registrationForm">
            <div class="form-group">
                <label>Student Name:</label>
                <input type="text" id="studentName" required>
            </div>
            
            <div class="form-group">
                <label>Student ID/USN:</label>
                <input type="text" id="studentId" required>
            </div>
            
            <div class="form-group">
                <label>Capture Method:</label>
                <select id="captureMethod">
                    <option value="camera">From Camera</option>
                    <option value="upload">Upload Images</option>
                </select>
            </div>
            
            <div id="cameraCapture" style="display:none;">
                <button type="button" id="captureBtn">Capture Faces</button>
                <div id="capturedImages"></div>
            </div>
            
            <div id="uploadCapture" style="display:none;">
                <input type="file" id="fileInput" multiple accept="image/*">
            </div>
            
            <button type="submit">Register Student</button>
        </form>
        
        <h2>Registered Students</h2>
        <div id="studentsList"></div>
    </div>
    
    <script src="register.js"></script>
</body>
</html>
```

## 📊 Performance Optimization

The system is optimized to maintain real-time performance:

1. **Recognition Caching**: Results cached for 5 seconds per track
2. **Selective Recognition**: Only runs every 10 frames per track
3. **Fast Matching**: Uses numpy matrix operations for speed
4. **Lazy Extraction**: Embeddings only extracted when needed

## 🎯 Expected Behavior

### Registered Students
- Bounding box: **Green**
- Label: **"Raghav Dhingra (95%)"**
- Stable identity across frames
- Analytics use real name

### Unknown Persons
- Bounding box: **Orange**
- Label: **"Unknown 1"**
- Temporary tracking-based ID
- Analytics show as "Unknown"

## 📈 Benefits

1. **Stable Identity**: No more duplicate IDs for same student
2. **Real Names**: Analytics show actual student names
3. **Cross-Session**: Students recognized across different sessions
4. **Attendance**: Automatic attendance tracking
5. **Better Analytics**: Engagement reports per actual student

## 🔧 Configuration

In `student_registry.py`:
- `recognition_threshold = 0.65` - Similarity threshold for recognition
- `min_embeddings = 3` - Minimum face images required for registration

In `face_reid_recognition.py`:
- `cache_duration = 5.0` - How long to cache recognition results
- `recognition_interval = 10` - Run recognition every N frames

## 🚀 Quick Start

1. Start server: `python3 integrated_server.py`
2. Open dashboard: `http://localhost:8080`
3. Register students: `http://localhost:8080/register.html`
4. Start camera: System will recognize registered students automatically

## 📝 Notes

- Registration data stored in `registered_students/` directory
- Each student needs 3-20 face images for best accuracy
- System falls back to "Unknown" for unregistered persons
- Recognition runs optimally without impacting FPS
- All existing features (analytics, video upload, etc.) work unchanged

