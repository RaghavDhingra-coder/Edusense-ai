// API Configuration — use dynamic origin so this works both locally and on deployed VMs
const API_BASE_URL = `${window.location.origin}/api`;

// DOM Elements
const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const uploadVideoBtn = document.getElementById('uploadVideoBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const videoFeed = document.getElementById('videoFeed');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const videoStats = document.getElementById('videoStats');
const videoProgress = document.getElementById('videoProgress');
const videoSectionTitle = document.getElementById('videoSectionTitle');
const loadingState = document.getElementById('loadingState');
const emptyState = document.getElementById('emptyState');
const dashboardContent = document.getElementById('dashboardContent');
const studentGrid = document.getElementById('studentGrid');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');

// Upload Modal Elements
const uploadModal = document.getElementById('uploadModal');
const closeModalBtn = document.getElementById('closeModalBtn');
const uploadArea = document.getElementById('uploadArea');
const videoFileInput = document.getElementById('videoFileInput');
const browseBtn = document.getElementById('browseBtn');
const uploadProgress = document.getElementById('uploadProgress');
const uploadSuccess = document.getElementById('uploadSuccess');
const startProcessingBtn = document.getElementById('startProcessingBtn');
const uploadProgressFill = document.getElementById('uploadProgressFill');
const uploadProgressText = document.getElementById('uploadProgressText');
const uploadFileName = document.getElementById('uploadFileName');
const uploadFileSize = document.getElementById('uploadFileSize');

// State
let cameraRunning = false;
let videoProcessing = false;
let statsInterval = null;
let isAnalyzing = false;
let currentSessionId = null;
let uploadedVideoPath = null;
let sourceType = null; // 'webcam' or 'video'

// Browser webcam state (cloud deployment — no VM webcam)
let browserStream = null;       // MediaStream from getUserMedia
let frameInterval = null;       // setInterval handle for frame sending
let captureCanvas = null;       // hidden <canvas> for frame capture
let captureCtx = null;          // 2D context of captureCanvas
let localVideo = null;          // temporary <video> element to hold the stream

// Debug function - can be called from browser console
window.debugUploadState = function() {
    console.log('═══════════════════════════════════════════════════════');
    console.log('DEBUG: UPLOAD STATE');
    console.log('═══════════════════════════════════════════════════════');
    console.log('uploadedVideoPath:', uploadedVideoPath);
    console.log('Type:', typeof uploadedVideoPath);
    console.log('Value:', uploadedVideoPath);
    console.log('Is null?', uploadedVideoPath === null);
    console.log('Is undefined?', uploadedVideoPath === undefined);
    console.log('Truthy?', !!uploadedVideoPath);
    console.log('═══════════════════════════════════════════════════════');
    return uploadedVideoPath;
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Set up event listeners
    startCameraBtn.addEventListener('click', startCamera);
    stopCameraBtn.addEventListener('click', stopCamera);
    uploadVideoBtn.addEventListener('click', openUploadModal);
    analyzeBtn.addEventListener('click', analyzeClassroom);
    
    // Upload modal listeners
    closeModalBtn.addEventListener('click', closeUploadModal);
    browseBtn.addEventListener('click', () => videoFileInput.click());
    videoFileInput.addEventListener('change', handleFileSelect);
    startProcessingBtn.addEventListener('click', startVideoProcessing);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => videoFileInput.click());
    
    // Close modal on outside click
    uploadModal.addEventListener('click', (e) => {
        if (e.target === uploadModal) {
            closeUploadModal();
        }
    });
    
    // Check initial camera status
    checkCameraStatus();
    
    // Show empty state initially
    emptyState.style.display = 'block';
});

/**
 * Reset dashboard for new session
 */
function resetDashboard() {
    console.log('🔄 Resetting dashboard for new session');
    
    // Hide analytics dashboard
    dashboardContent.style.display = 'none';
    
    // Show empty state
    emptyState.style.display = 'block';
    
    // Clear student grid
    studentGrid.innerHTML = '';
    
    // Reset summary cards
    document.getElementById('totalStudents').textContent = '0';
    document.getElementById('focusedStudents').textContent = '0';
    document.getElementById('distractedStudents').textContent = '0';
    document.getElementById('avgEngagement').textContent = '0%';
    
    // Reset distribution bars
    document.getElementById('focusedBar').style.width = '0%';
    document.getElementById('focusedPercent').textContent = '0%';
    document.getElementById('neutralBar').style.width = '0%';
    document.getElementById('neutralPercent').textContent = '0%';
    document.getElementById('distractedBar').style.width = '0%';
    document.getElementById('distractedPercent').textContent = '0%';
    
    console.log('✅ Dashboard reset complete');
}

/**
 * Start camera — uses browser getUserMedia and sends frames to backend.
 * This works on both local machines and Google Cloud VMs (no VM webcam needed).
 */
async function startCamera() {
    try {
        console.log('🎥 Starting new camera session (browser webcam)...');
        startCameraBtn.disabled = true;

        // Reset dashboard for new session
        resetDashboard();

        // 1. Ask backend to initialise a session (no cv2.VideoCapture on server)
        const response = await fetch(`${API_BASE_URL}/camera/start`, {
            method: 'POST'
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Failed to start camera session');
        }

        console.log('✅ Backend session started:', data.session_id);
        cameraRunning = true;
        currentSessionId = data.session_id;

        // 2. Open browser webcam
        browserStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false
        });

        // 3. Set up hidden video element to feed the canvas
        localVideo = document.createElement('video');
        localVideo.srcObject = browserStream;
        localVideo.setAttribute('playsinline', '');
        localVideo.muted = true;
        await localVideo.play();

        // 4. Set up capture canvas (matches video dimensions)
        captureCanvas = document.getElementById('captureCanvas');
        captureCanvas.width  = localVideo.videoWidth  || 640;
        captureCanvas.height = localVideo.videoHeight || 480;
        captureCtx = captureCanvas.getContext('2d');

        // 5. Update UI
        videoSectionTitle.textContent = 'Live Classroom Feed';
        startCameraBtn.style.display = 'none';
        uploadVideoBtn.style.display = 'none';
        stopCameraBtn.style.display = 'inline-flex';
        stopCameraBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Camera';
        videoPlaceholder.style.display = 'none';
        videoFeed.style.display = 'block';
        videoStats.style.display = 'flex';
        videoProgress.style.display = 'none';

        // 6. Start sending frames to backend every 250 ms
        frameInterval = setInterval(sendFrameToBackend, 250);

        // 7. Start stats polling
        startStatsPolling();

        showNotification(`New session started: ${data.session_id}`, 'success');

    } catch (error) {
        console.error('❌ Start camera error:', error);
        // Clean up any partial state
        _stopBrowserStream();
        alert(`Failed to start camera: ${error.message}`);
    } finally {
        startCameraBtn.disabled = false;
    }
}

/**
 * Capture one frame from the browser webcam, POST it to /api/process_frame,
 * and display the returned annotated frame in the <img> element.
 */
async function sendFrameToBackend() {
    if (!cameraRunning || !localVideo || !captureCtx) return;

    try {
        // Draw current video frame onto canvas
        captureCanvas.width  = localVideo.videoWidth  || 640;
        captureCanvas.height = localVideo.videoHeight || 480;
        captureCtx.drawImage(localVideo, 0, 0, captureCanvas.width, captureCanvas.height);

        // Get JPEG as base64 (quality 0.7 keeps payload small)
        const b64 = captureCanvas.toDataURL('image/jpeg', 0.7);

        const res = await fetch(`${API_BASE_URL}/process_frame`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame: b64 })
        });

        if (!res.ok) return;

        const result = await res.json();
        if (result.success && result.frame) {
            // Display the AI-annotated frame returned by the server
            videoFeed.src = `data:image/jpeg;base64,${result.frame}`;
        }
    } catch (err) {
        // Silently ignore individual frame errors (network hiccup etc.)
        console.debug('Frame send error:', err);
    }
}

/**
 * Internal helper — stop browser media stream and clear frame interval.
 */
function _stopBrowserStream() {
    if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
    }
    if (browserStream) {
        browserStream.getTracks().forEach(t => t.stop());
        browserStream = null;
    }
    if (localVideo) {
        localVideo.srcObject = null;
        localVideo = null;
    }
}

/**
 * Stop camera — stops browser stream and notifies backend.
 */
async function stopCamera() {
    try {
        console.log('🛑 Stopping camera...');
        stopCameraBtn.disabled = true;

        // Stop browser webcam stream and frame sending
        _stopBrowserStream();

        const response = await fetch(`${API_BASE_URL}/camera/stop`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            console.log('✅ Camera stopped');
            cameraRunning = false;
            currentSessionId = null;

            // Update UI
            videoSectionTitle.textContent = 'Live Classroom Feed';
            startCameraBtn.style.display = 'inline-flex';
            uploadVideoBtn.style.display = 'inline-flex';
            stopCameraBtn.style.display = 'none';
            videoPlaceholder.style.display = 'flex';
            videoFeed.style.display = 'none';
            videoStats.style.display = 'none';
            videoFeed.src = '';

            // Stop stats polling
            stopStatsPolling();

            // Reset stats display
            document.getElementById('statFps').textContent = '0';
            document.getElementById('statFaces').textContent = '0';
            document.getElementById('statStudents').textContent = '0';
            document.getElementById('statImages').textContent = '0';

            showNotification('Camera stopped - session data preserved', 'info');
        }

    } catch (error) {
        console.error('❌ Stop camera error:', error);
        alert(`Failed to stop camera: ${error.message}`);
    } finally {
        stopCameraBtn.disabled = false;
    }
}

/**
 * Check camera status on page load.
 * For browser-webcam mode we cannot auto-resume a stream without a user gesture,
 * so we just reflect the server state in the UI without starting getUserMedia.
 */
async function checkCameraStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/camera/status`);
        const data = await response.json();

        if (data.success && data.status.running) {
            // A session is active server-side (e.g. video file processing)
            // Only auto-resume the MJPEG stream for video-file sessions
            if (data.status.is_video_file) {
                cameraRunning = true;
                startCameraBtn.style.display = 'none';
                stopCameraBtn.style.display = 'inline-flex';
                videoPlaceholder.style.display = 'none';
                videoFeed.style.display = 'block';
                videoStats.style.display = 'flex';
                videoFeed.src = `${API_BASE_URL}/video_feed?t=${Date.now()}`;
                startStatsPolling();
            }
            // For webcam sessions: leave UI in default state — user must click Start Camera
        }
    } catch (error) {
        console.log('Camera status check failed:', error);
    }
}

/**
 * Start polling camera stats
 */
function startStatsPolling() {
    if (statsInterval) {
        clearInterval(statsInterval);
    }
    
    statsInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/camera/status`);
            const data = await response.json();
            
            if (data.success && data.status) {
                const stats = data.status;
                document.getElementById('statFps').textContent = stats.fps.toFixed(1);
                document.getElementById('statFaces').textContent = stats.active_tracks || 0;
                document.getElementById('statStudents').textContent = stats.total_students || 0;
                document.getElementById('statImages').textContent = stats.total_images || 0;
                
                // Update video progress for video files
                if (stats.is_video_file && videoProgress) {
                    videoProgress.style.display = 'block';
                    document.getElementById('progressFrames').textContent = 
                        `Frame ${stats.frame_number} / ${stats.total_frames}`;
                    document.getElementById('progressPercent').textContent = 
                        `${Math.round(stats.progress_percent)}%`;
                    document.getElementById('videoProgressFill').style.width = 
                        `${stats.progress_percent}%`;
                    
                    // Check if processing is complete
                    if (stats.processing_complete) {
                        showNotification('Video processing complete!', 'success');
                        stopStatsPolling();
                        videoProcessing = false;
                        stopCameraBtn.style.display = 'none';
                        analyzeBtn.disabled = false;
                    }
                }
            }
        } catch (error) {
            console.error('Stats polling error:', error);
        }
    }, 1000);
}

/**
 * Stop polling camera stats
 */
function stopStatsPolling() {
    if (statsInterval) {
        clearInterval(statsInterval);
        statsInterval = null;
    }
}

/**
 * Analyze classroom engagement
 */
async function analyzeClassroom() {
    if (isAnalyzing) return;
    
    isAnalyzing = true;
    analyzeBtn.disabled = true;
    
    // Show loading state
    emptyState.style.display = 'none';
    dashboardContent.style.display = 'none';
    loadingState.style.display = 'block';
    
    // Simulate progress
    simulateProgress();
    
    try {
        console.log('🔍 Starting analysis...');
        
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const errorData = await response.json();
            console.error('API Error:', errorData);
            throw new Error(errorData.error || 'Analysis failed');
        }
        
        const data = await response.json();
        console.log('Analysis result:', data);
        
        if (data.success) {
            // Complete progress
            updateProgress(100);
            
            // Wait a bit for visual effect
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Display results
            displayResults(data);
            
            showNotification('Analysis complete!', 'success');
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
        
    } catch (error) {
        console.error('Analysis error:', error);
        
        let errorMessage = 'Failed to analyze classroom. ';
        
        if (error.message.includes('Failed to fetch')) {
            errorMessage += 'Cannot connect to server.';
        } else if (error.message.includes('No student folders')) {
            errorMessage += 'No student data found. Start the camera first to collect data.';
        } else {
            errorMessage += error.message;
        }
        
        alert(errorMessage);
        
        // Show empty state again
        loadingState.style.display = 'none';
        emptyState.style.display = 'block';
    } finally {
        isAnalyzing = false;
        analyzeBtn.disabled = false;
    }
}

/**
 * Simulate progress bar
 */
function simulateProgress() {
    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress >= 90) {
            progress = 90;
            clearInterval(interval);
        }
        updateProgress(progress);
    }, 500);
}

/**
 * Update progress bar
 */
function updateProgress(percent) {
    progressFill.style.width = `${percent}%`;
    progressText.textContent = `${Math.round(percent)}%`;
}

/**
 * Display analysis results
 */
function displayResults(data) {
    const { summary, students } = data;
    
    // Hide loading, show dashboard
    loadingState.style.display = 'none';
    dashboardContent.style.display = 'block';
    
    // Update summary cards
    updateSummaryCards(summary);
    
    // Update distribution chart
    updateDistributionChart(summary);
    
    // Display student cards
    displayStudentCards(students);
}

/**
 * Update summary cards
 */
function updateSummaryCards(summary) {
    document.getElementById('totalStudents').textContent = summary.total_students;
    document.getElementById('focusedStudents').textContent = summary.focused_students;
    document.getElementById('distractedStudents').textContent = summary.unfocused_students;
    document.getElementById('avgEngagement').textContent = `${summary.average_engagement}%`;
}

/**
 * Update distribution chart
 */
function updateDistributionChart(summary) {
    const total = summary.total_students;
    
    if (total === 0) return;
    
    const focusedPercent = (summary.focused_students / total) * 100;
    const moderatePercent = (summary.moderately_focused_students / total) * 100;
    const unfocusedPercent = (summary.unfocused_students / total) * 100;
    
    // Animate bars
    setTimeout(() => {
        document.getElementById('focusedBar').style.width = `${focusedPercent}%`;
        document.getElementById('focusedPercent').textContent = `${Math.round(focusedPercent)}%`;
        
        document.getElementById('neutralBar').style.width = `${moderatePercent}%`;
        document.getElementById('neutralPercent').textContent = `${Math.round(moderatePercent)}%`;
        
        document.getElementById('distractedBar').style.width = `${unfocusedPercent}%`;
        document.getElementById('distractedPercent').textContent = `${Math.round(unfocusedPercent)}%`;
    }, 100);
}

/**
 * Display student cards
 */
function displayStudentCards(students) {
    studentGrid.innerHTML = '';
    
    // Sort by engagement percentage (descending)
    students.sort((a, b) => {
        const scoreA = a.engagement_percentage || a.engagement_score || 0;
        const scoreB = b.engagement_percentage || b.engagement_score || 0;
        return scoreB - scoreA;
    });
    
    students.forEach((student, index) => {
        const card = createStudentCard(student);
        studentGrid.appendChild(card);
        
        // Stagger animation
        card.style.animationDelay = `${index * 0.05}s`;
    });
}

/**
 * Create student card element
 */
function createStudentCard(student) {
    const card = document.createElement('div');
    card.className = 'student-card';
    
    const engagementScore = student.engagement_percentage || student.engagement_score || 0;
    
    let scoreClass = 'low';
    if (engagementScore >= 75) scoreClass = 'high';
    else if (engagementScore >= 40) scoreClass = 'medium';
    
    let statusClass = student.status.toLowerCase().replace(/\s+/g, '-');
    
    // Handle both session-based and legacy image paths
    let imageUrl = 'placeholder.jpg';
    if (student.sample_image) {
        // Check if it's already a session path
        if (student.sample_image.startsWith('sessions/')) {
            imageUrl = `${API_BASE_URL}/images/${student.sample_image}`;
        } else {
            // Legacy path - remove 'students/' prefix
            imageUrl = `${API_BASE_URL}/images/${student.sample_image.replace('students/', '')}`;
        }
    }
    
    const breakdown = Object.entries(student.prediction_breakdown)
        .sort((a, b) => b[1].count - a[1].count)
        .slice(0, 3);
    
    // Build pose debug info if available
    let poseDebugHtml = '';
    if (student.head_pose_stats && student.head_pose_stats.avg_yaw !== null) {
        const pose = student.head_pose_stats;
        const method = student.method_breakdown;
        const fallbackUsed = method.fallback_count > 0;
        const poseReliability = (method.pose_reliability * 100).toFixed(0);
        
        poseDebugHtml = `
            <div class="pose-debug">
                <div class="pose-debug-title">🎯 Head Pose Detection</div>
                <div class="pose-angles">
                    <div class="pose-angle">
                        <span class="pose-label">Yaw:</span>
                        <span class="pose-value ${Math.abs(pose.avg_yaw) > 30 ? 'pose-bad' : 'pose-good'}">${pose.avg_yaw}°</span>
                    </div>
                    <div class="pose-angle">
                        <span class="pose-label">Pitch:</span>
                        <span class="pose-value ${Math.abs(pose.avg_pitch) > 30 ? 'pose-bad' : 'pose-good'}">${pose.avg_pitch}°</span>
                    </div>
                    <div class="pose-angle">
                        <span class="pose-label">Roll:</span>
                        <span class="pose-value ${Math.abs(pose.avg_roll) > 35 ? 'pose-bad' : 'pose-good'}">${pose.avg_roll}°</span>
                    </div>
                </div>
                <div class="pose-meta">
                    <span class="pose-reliability">Pose Reliability: ${poseReliability}%</span>
                    <span class="pose-fallback ${fallbackUsed ? 'fallback-yes' : 'fallback-no'}">
                        Fallback: ${fallbackUsed ? 'Yes (' + method.fallback_count + ' frames)' : 'No'}
                    </span>
                </div>
            </div>
        `;
    }
    
    card.innerHTML = `
        <img src="${imageUrl}" alt="${student.student_id}" class="student-image" 
             onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22320%22 height=%22200%22%3E%3Crect fill=%22%23e5e7eb%22 width=%22320%22 height=%22200%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 dominant-baseline=%22middle%22 text-anchor=%22middle%22 font-family=%22sans-serif%22 font-size=%2218%22 fill=%22%239ca3af%22%3ENo Image%3C/text%3E%3C/svg%3E'">
        
        <div class="student-info">
            <div class="student-header">
                <div class="student-id">${formatStudentId(student.student_id)}</div>
                <span class="status-badge ${statusClass}">${student.status}</span>
            </div>
            
            <div class="engagement-score">
                <div class="score-label">Engagement Score</div>
                <div class="score-bar">
                    <div class="score-fill ${scoreClass}" style="width: ${engagementScore}%">
                        ${Math.round(engagementScore)}%
                    </div>
                </div>
            </div>
            
            ${poseDebugHtml}
            
            <div class="student-stats">
                <div class="stat-item">
                    <div class="stat-value">${student.images_analyzed}</div>
                    <div class="stat-label">Images</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${student.most_common_state}</div>
                    <div class="stat-label">Most Common</div>
                </div>
            </div>
            
            <ul class="breakdown-list">
                ${breakdown.map(([state, data]) => `
                    <li class="breakdown-item">
                        <span class="breakdown-label">${state}</span>
                        <span class="breakdown-value">${data.count} (${data.percentage}%)</span>
                    </li>
                `).join('')}
            </ul>
        </div>
    `;
    
    return card;
}

/**
 * Format student ID for display
 */
function formatStudentId(id) {
    return id.replace('student_', 'Student ');
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    // Could add toast notifications here
}


// ============================================================================
// VIDEO UPLOAD FUNCTIONS
// ============================================================================

/**
 * Open upload modal
 */
function openUploadModal() {
    uploadModal.style.display = 'flex';
    resetUploadModal();
}

/**
 * Close upload modal
 */
function closeUploadModal() {
    uploadModal.style.display = 'none';
    resetUploadModal();
}

/**
 * Reset upload modal to initial state
 */
function resetUploadModal() {
    console.log('🔄 Resetting upload modal UI');
    uploadArea.style.display = 'block';
    uploadProgress.style.display = 'none';
    uploadSuccess.style.display = 'none';
    videoFileInput.value = '';
    // DO NOT reset uploadedVideoPath here - it needs to persist!
    console.log('📁 uploadedVideoPath preserved:', uploadedVideoPath);
}

/**
 * Handle drag over
 */
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('drag-over');
}

/**
 * Handle drag leave
 */
function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');
}

/**
 * Handle drop
 */
async function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        console.log('📁 File dropped:', files[0].name);
        await handleFile(files[0]);
    }
}

/**
 * Handle file select
 */
async function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        console.log('📁 File selected from input:', files[0].name);
        await handleFile(files[0]);
    }
}

/**
 * Handle file upload
 */
async function handleFile(file) {
    console.log('═══════════════════════════════════════════════════════');
    console.log('📤 STARTING FILE UPLOAD');
    console.log('═══════════════════════════════════════════════════════');
    console.log('📁 File name:', file.name);
    console.log('📊 File size:', (file.size / (1024 * 1024)).toFixed(2), 'MB');
    console.log('📝 File type:', file.type);
    console.log('📁 uploadedVideoPath BEFORE upload:', uploadedVideoPath);
    
    // Validate file type
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 
                       'video/x-flv', 'video/x-ms-wmv', 'video/webm'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv|flv|wmv|webm)$/i)) {
        alert('Invalid file type. Please upload a video file (MP4, AVI, MOV, MKV, FLV, WMV, WEBM)');
        return;
    }
    
    // Validate file size (500MB max)
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
        alert('File too large. Maximum size is 500MB');
        return;
    }
    
    // Show upload progress
    uploadArea.style.display = 'none';
    uploadProgress.style.display = 'block';
    uploadFileName.textContent = file.name;
    uploadFileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    
    // Create form data
    const formData = new FormData();
    formData.append('video', file);
    
    console.log('📤 Sending upload request...');
    
    // Upload with progress tracking using Promise wrapper
    try {
        const result = await new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percent = (e.loaded / e.total) * 100;
                    uploadProgressFill.style.width = `${percent}%`;
                    uploadProgressText.textContent = `Uploading... ${Math.round(percent)}%`;
                    if (percent % 25 === 0) {
                        console.log(`📤 Upload progress: ${Math.round(percent)}%`);
                    }
                }
            });
            
            xhr.addEventListener('load', () => {
                console.log('📥 Upload response received');
                console.log('   Status:', xhr.status);
                console.log('   Response text:', xhr.responseText);
                
                if (xhr.status === 200) {
                    try {
                        const data = JSON.parse(xhr.responseText);
                        console.log('📥 Parsed response:', data);
                        console.log('   Keys:', Object.keys(data));
                        console.log('   data.success:', data.success);
                        console.log('   data.filepath:', data.filepath);
                        
                        if (data.success) {
                            resolve(data);
                        } else {
                            reject(new Error(data.error || 'Upload failed'));
                        }
                    } catch (e) {
                        console.error('❌ Failed to parse response:', e);
                        reject(new Error('Failed to parse upload response: ' + e.message));
                    }
                } else {
                    console.error('❌ Upload failed with status:', xhr.status);
                    reject(new Error(`Upload failed with status ${xhr.status}`));
                }
            });
            
            xhr.addEventListener('error', () => {
                console.error('❌ Upload network error');
                reject(new Error('Upload failed - network error'));
            });
            
            xhr.addEventListener('abort', () => {
                console.error('❌ Upload aborted');
                reject(new Error('Upload aborted'));
            });
            
            console.log('📤 Opening XHR connection to:', `${API_BASE_URL}/video/upload`);
            xhr.open('POST', `${API_BASE_URL}/video/upload`);
            xhr.send(formData);
        });
        
        // Upload completed successfully
        console.log('═══════════════════════════════════════════════════════');
        console.log('✅ UPLOAD COMPLETED SUCCESSFULLY');
        console.log('═══════════════════════════════════════════════════════');
        console.log('📁 Received filepath:', result.filepath);
        console.log('📝 Filename:', result.filename);
        console.log('📊 Size:', result.size_mb, 'MB');
        
        // CRITICAL: Set global variable
        console.log('📁 Setting uploadedVideoPath to:', result.filepath);
        uploadedVideoPath = result.filepath;
        console.log('📁 uploadedVideoPath is now:', uploadedVideoPath);
        console.log('📁 Type:', typeof uploadedVideoPath);
        console.log('📁 Length:', uploadedVideoPath ? uploadedVideoPath.length : 'NULL');
        console.log('═══════════════════════════════════════════════════════');
        
        // Update UI - show success state
        uploadProgress.style.display = 'none';
        uploadSuccess.style.display = 'block';
        
        console.log('✅ UI updated - Start Processing button should now be visible');
        
        return result;
        
    } catch (error) {
        console.error('═══════════════════════════════════════════════════════');
        console.error('❌ UPLOAD FAILED');
        console.error('═══════════════════════════════════════════════════════');
        console.error('Error:', error.message);
        console.error('Stack:', error.stack);
        alert(`Upload failed: ${error.message}`);
        resetUploadModal();
        throw error;
    }
}

/**
 * Start video processing
 */
async function startVideoProcessing() {
    console.log('═══════════════════════════════════════════════════════');
    console.log('🎬 START PROCESSING BUTTON CLICKED');
    console.log('═══════════════════════════════════════════════════════');
    console.log('📁 CHECKING uploadedVideoPath:');
    console.log('   Value:', uploadedVideoPath);
    console.log('   Type:', typeof uploadedVideoPath);
    console.log('   Is null?', uploadedVideoPath === null);
    console.log('   Is undefined?', uploadedVideoPath === undefined);
    console.log('   Is empty string?', uploadedVideoPath === '');
    console.log('   Truthy?', !!uploadedVideoPath);
    
    if (!uploadedVideoPath) {
        console.error('❌ VALIDATION FAILED: uploadedVideoPath is falsy');
        console.error('   Actual value:', uploadedVideoPath);
        alert('No video file uploaded. Please upload a video first.');
        return;
    }
    
    // Additional validation
    if (typeof uploadedVideoPath !== 'string') {
        console.error('❌ VALIDATION FAILED: uploadedVideoPath is not a string');
        console.error('   Type:', typeof uploadedVideoPath);
        console.error('   Value:', uploadedVideoPath);
        alert('Invalid video path type. Please upload the video again.');
        uploadedVideoPath = null;
        resetUploadModal();
        return;
    }
    
    if (uploadedVideoPath.trim() === '') {
        console.error('❌ VALIDATION FAILED: uploadedVideoPath is empty string');
        alert('Invalid video path. Please upload the video again.');
        uploadedVideoPath = null;
        resetUploadModal();
        return;
    }
    
    console.log('✅ VALIDATION PASSED');
    console.log('═══════════════════════════════════════════════════════');
    
    try {
        console.log('🎬 Starting video processing...');
        console.log('📁 Using filepath:', uploadedVideoPath);
        
        // Close modal
        closeUploadModal();
        
        // Reset dashboard
        resetDashboard();
        
        // Stop any existing processing
        if (cameraRunning || videoProcessing) {
            console.log('🛑 Stopping existing processing...');
            await stopCamera();
            await new Promise(resolve => setTimeout(resolve, 500));
        }
        
        // Prepare request
        const requestBody = {
            filepath: uploadedVideoPath
        };
        
        console.log('═══════════════════════════════════════════════════════');
        console.log('📤 SENDING PROCESSING REQUEST');
        console.log('═══════════════════════════════════════════════════════');
        console.log('   URL:', `${API_BASE_URL}/video/process`);
        console.log('   Method: POST');
        console.log('   Body:', JSON.stringify(requestBody, null, 2));
        console.log('   Body.filepath:', requestBody.filepath);
        console.log('═══════════════════════════════════════════════════════');
        
        const response = await fetch(`${API_BASE_URL}/video/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        console.log('═══════════════════════════════════════════════════════');
        console.log('📥 PROCESSING RESPONSE RECEIVED');
        console.log('═══════════════════════════════════════════════════════');
        console.log('   Status:', response.status);
        console.log('   Status Text:', response.statusText);
        
        const data = await response.json();
        console.log('   Response data:', JSON.stringify(data, null, 2));
        console.log('═══════════════════════════════════════════════════════');
        
        if (data.success) {
            console.log('✅ Video processing started');
            console.log('📁 Session ID:', data.session_id);
            console.log('🎞️  Total frames:', data.total_frames);
            
            videoProcessing = true;
            sourceType = 'video';
            currentSessionId = data.session_id;
            
            // Update UI
            videoSectionTitle.textContent = 'Video Processing';
            startCameraBtn.style.display = 'none';
            uploadVideoBtn.style.display = 'none';
            stopCameraBtn.style.display = 'inline-flex';
            stopCameraBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Processing';
            videoPlaceholder.style.display = 'none';
            videoFeed.style.display = 'block';
            videoStats.style.display = 'flex';
            videoProgress.style.display = 'block';
            
            // Start video stream
            videoFeed.src = `${API_BASE_URL}/video_feed?t=${Date.now()}`;
            
            // Start stats polling
            startStatsPolling();
            
            // Show success message
            showNotification(`Video processing started: ${data.total_frames} frames`, 'success');
        } else {
            throw new Error(data.error || 'Failed to start video processing');
        }
        
    } catch (error) {
        console.error('❌ Video processing error:', error);
        alert(`Failed to start video processing: ${error.message}`);
    }
}

/**
 * Start camera — browser-webcam version that also handles the video-upload flow.
 * This overrides the earlier definition and is the one that actually runs.
 */
async function startCamera() {
    try {
        console.log('🎥 Starting new camera session (browser webcam)...');
        startCameraBtn.disabled = true;

        // Reset dashboard for new session
        resetDashboard();

        // 1. Tell backend to initialise a session (browser_mode — no cv2.VideoCapture)
        const response = await fetch(`${API_BASE_URL}/camera/start`, {
            method: 'POST'
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Failed to start camera session');
        }

        console.log('✅ Backend session started:', data.session_id);
        cameraRunning = true;
        sourceType = 'webcam';
        currentSessionId = data.session_id;

        // 2. Open browser webcam
        browserStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false
        });

        // 3. Hidden video element to feed the canvas
        localVideo = document.createElement('video');
        localVideo.srcObject = browserStream;
        localVideo.setAttribute('playsinline', '');
        localVideo.muted = true;
        await localVideo.play();

        // 4. Capture canvas
        captureCanvas = document.getElementById('captureCanvas');
        captureCanvas.width  = localVideo.videoWidth  || 640;
        captureCanvas.height = localVideo.videoHeight || 480;
        captureCtx = captureCanvas.getContext('2d');

        // 5. Update UI
        videoSectionTitle.textContent = 'Live Classroom Feed';
        startCameraBtn.style.display = 'none';
        uploadVideoBtn.style.display = 'none';
        stopCameraBtn.style.display = 'inline-flex';
        stopCameraBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Camera';
        videoPlaceholder.style.display = 'none';
        videoFeed.style.display = 'block';
        videoStats.style.display = 'flex';
        videoProgress.style.display = 'none';

        // 6. Start sending frames every 250 ms
        frameInterval = setInterval(sendFrameToBackend, 250);

        // 7. Stats polling
        startStatsPolling();

        showNotification(`New session started: ${data.session_id}`, 'success');

    } catch (error) {
        console.error('❌ Start camera error:', error);
        _stopBrowserStream();
        alert(`Failed to start camera: ${error.message}`);
    } finally {
        startCameraBtn.disabled = false;
    }
}

/**
 * Stop camera — stops browser stream + video processing and notifies backend.
 */
async function stopCamera() {
    try {
        console.log('🛑 Stopping...');
        stopCameraBtn.disabled = true;

        // Always stop browser stream (no-op if not running)
        _stopBrowserStream();

        const endpoint = videoProcessing ? '/video/stop' : '/camera/stop';
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            console.log('✅ Stopped');
            cameraRunning = false;
            videoProcessing = false;
            sourceType = null;

            // Update UI
            videoSectionTitle.textContent = 'Live Classroom Feed';
            startCameraBtn.style.display = 'inline-flex';
            uploadVideoBtn.style.display = 'inline-flex';
            stopCameraBtn.style.display = 'none';
            videoPlaceholder.style.display = 'flex';
            videoFeed.style.display = 'none';
            videoStats.style.display = 'none';
            videoProgress.style.display = 'none';
            videoFeed.src = '';

            stopStatsPolling();

            document.getElementById('statFps').textContent = '0';
            document.getElementById('statFaces').textContent = '0';
            document.getElementById('statStudents').textContent = '0';
            document.getElementById('statImages').textContent = '0';

            showNotification('Stopped - session data preserved', 'info');
        }

    } catch (error) {
        console.error('❌ Stop error:', error);
        alert(`Failed to stop: ${error.message}`);
    } finally {
        stopCameraBtn.disabled = false;
    }
}
