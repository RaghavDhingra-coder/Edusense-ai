// API Configuration
const API_BASE_URL = 'http://localhost:8080/api';

// State
let capturedImages = [];
let captureMethod = 'camera';

// DOM Elements
const studentNameInput = document.getElementById('studentName');
const studentIdInput = document.getElementById('studentId');
const captureBtn = document.getElementById('captureBtn');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const capturedImagesDiv = document.getElementById('capturedImages');
const registerBtn = document.getElementById('registerBtn');
const registrationForm = document.getElementById('registrationForm');
const statusMessage = document.getElementById('statusMessage');
const studentsList = document.getElementById('studentsList');
const studentCount = document.getElementById('studentCount');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Set up event listeners
    document.querySelectorAll('.method-option').forEach(option => {
        option.addEventListener('click', () => selectCaptureMethod(option.dataset.method));
    });
    
    captureBtn.addEventListener('click', captureFromCamera);
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileUpload);
    registrationForm.addEventListener('submit', registerStudent);
    
    // Load registered students
    loadStudents();
});

/**
 * Select capture method
 */
function selectCaptureMethod(method) {
    captureMethod = method;
    
    // Update UI
    document.querySelectorAll('.method-option').forEach(opt => {
        opt.classList.remove('active');
    });
    document.querySelector(`[data-method="${method}"]`).classList.add('active');
    
    // Show/hide sections
    if (method === 'camera') {
        document.getElementById('cameraCapture').style.display = 'block';
        document.getElementById('uploadCapture').style.display = 'none';
    } else {
        document.getElementById('cameraCapture').style.display = 'none';
        document.getElementById('uploadCapture').style.display = 'block';
    }
    
    // Clear images
    capturedImages = [];
    updateImagesDisplay();
}

/**
 * Capture images from camera
 */
async function captureFromCamera() {
    try {
        captureBtn.disabled = true;
        captureBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Capturing...';
        
        showStatus('Capturing face images from camera...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/students/capture`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                num_captures: 10
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            capturedImages = data.images;
            updateImagesDisplay();
            showStatus(`Successfully captured ${data.count} images!`, 'success');
        } else {
            throw new Error(data.error || 'Capture failed');
        }
        
    } catch (error) {
        console.error('Capture error:', error);
        showStatus(`Capture failed: ${error.message}`, 'error');
    } finally {
        captureBtn.disabled = false;
        captureBtn.innerHTML = '<i class="fas fa-camera"></i> Capture 10 Face Images';
    }
}

/**
 * Handle file upload
 */
function handleFileUpload(e) {
    const files = Array.from(e.target.files);
    
    if (files.length === 0) return;
    
    showStatus(`Loading ${files.length} images...`, 'info');
    
    capturedImages = [];
    let loaded = 0;
    
    files.forEach(file => {
        const reader = new FileReader();
        
        reader.onload = (event) => {
            capturedImages.push(event.target.result);
            loaded++;
            
            if (loaded === files.length) {
                updateImagesDisplay();
                showStatus(`Loaded ${files.length} images successfully!`, 'success');
            }
        };
        
        reader.readAsDataURL(file);
    });
}

/**
 * Update images display
 */
function updateImagesDisplay() {
    capturedImagesDiv.innerHTML = '';
    
    capturedImages.forEach((imgData, index) => {
        const div = document.createElement('div');
        div.className = 'captured-image';
        div.innerHTML = `
            <img src="${imgData}" alt="Face ${index + 1}">
            <button class="remove-btn" onclick="removeImage(${index})">
                <i class="fas fa-times"></i>
            </button>
        `;
        capturedImagesDiv.appendChild(div);
    });
    
    // Enable/disable register button
    registerBtn.disabled = capturedImages.length < 3;
}

/**
 * Remove image
 */
function removeImage(index) {
    capturedImages.splice(index, 1);
    updateImagesDisplay();
}

/**
 * Register student
 */
async function registerStudent(e) {
    e.preventDefault();
    
    const studentName = studentNameInput.value.trim();
    const studentId = studentIdInput.value.trim();
    
    if (!studentName || !studentId) {
        showStatus('Please enter student name and ID', 'error');
        return;
    }
    
    if (capturedImages.length < 3) {
        showStatus('Please provide at least 3 face images', 'error');
        return;
    }
    
    try {
        registerBtn.disabled = true;
        registerBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Registering...';
        
        showStatus('Registering student...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/students/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                student_name: studentName,
                student_id: studentId,
                face_images: capturedImages
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus(`Student ${studentName} registered successfully!`, 'success');
            
            // Reset form
            registrationForm.reset();
            capturedImages = [];
            updateImagesDisplay();
            
            // Reload students list
            loadStudents();
        } else {
            throw new Error(data.error || 'Registration failed');
        }
        
    } catch (error) {
        console.error('Registration error:', error);
        showStatus(`Registration failed: ${error.message}`, 'error');
    } finally {
        registerBtn.disabled = false;
        registerBtn.innerHTML = '<i class="fas fa-check"></i> Register Student';
    }
}

/**
 * Load registered students
 */
async function loadStudents() {
    try {
        const response = await fetch(`${API_BASE_URL}/students/list`);
        const data = await response.json();
        
        if (data.success) {
            displayStudents(data.students);
            studentCount.textContent = data.count;
        }
    } catch (error) {
        console.error('Load students error:', error);
    }
}

/**
 * Display students list
 */
function displayStudents(students) {
    if (students.length === 0) {
        studentsList.innerHTML = '<p style="text-align: center; color: #9ca3af; padding: 40px;">No students registered yet</p>';
        return;
    }
    
    studentsList.innerHTML = students.map(student => `
        <div class="student-item">
            <div class="student-info">
                <h3>${student.student_name}</h3>
                <p>ID: ${student.student_id} | Embeddings: ${student.num_embeddings} | Registered: ${formatDate(student.registered_date)}</p>
                ${student.last_seen ? `<p style="color: #10b981;">Last seen: ${formatDate(student.last_seen)}</p>` : ''}
            </div>
            <div class="student-actions">
                <button onclick="deleteStudent('${student.student_id}', '${student.student_name}')">
                    <i class="fas fa-trash"></i> Delete
                </button>
            </div>
        </div>
    `).join('');
}

/**
 * Delete student
 */
async function deleteStudent(studentId, studentName) {
    if (!confirm(`Are you sure you want to delete ${studentName}?`)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/students/delete/${studentId}`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus(`Student ${studentName} deleted successfully`, 'success');
            loadStudents();
        } else {
            throw new Error(data.error || 'Delete failed');
        }
    } catch (error) {
        console.error('Delete error:', error);
        showStatus(`Delete failed: ${error.message}`, 'error');
    }
}

/**
 * Show status message
 */
function showStatus(message, type) {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`;
    statusMessage.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        statusMessage.style.display = 'none';
    }, 5000);
}

/**
 * Format date
 */
function formatDate(dateString) {
    if (!dateString) return 'Never';
    const date = new Date(dateString);
    return date.toLocaleString();
}
