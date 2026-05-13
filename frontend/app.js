// API Configuration — use dynamic origin so this works both locally and on deployed VMs
const API_BASE_URL = `${window.location.origin}/api`;

// DOM Elements
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingState = document.getElementById('loadingState');
const emptyState = document.getElementById('emptyState');
const dashboardContent = document.getElementById('dashboardContent');
const studentGrid = document.getElementById('studentGrid');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');

// State
let isAnalyzing = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Check if there's cached data
    checkForCachedData();
    
    // Set up event listeners
    analyzeBtn.addEventListener('click', analyzeClassroom);
});

/**
 * Check for cached analysis data
 */
async function checkForCachedData() {
    try {
        const response = await fetch(`${API_BASE_URL}/summary`);
        
        if (response.ok) {
            const data = await response.json();
            if (data.success && data.cached) {
                // Load full analysis
                loadCachedAnalysis();
            }
        }
    } catch (error) {
        console.log('No cached data available');
    }
}

/**
 * Load cached analysis data
 */
async function loadCachedAnalysis() {
    try {
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST'
        });
        
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                displayResults(data);
            }
        }
    } catch (error) {
        console.error('Failed to load cached analysis:', error);
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
    
    // Simulate progress (since we don't have real-time updates)
    simulateProgress();
    
    try {
        console.log('🔍 Starting analysis...');
        console.log('API URL:', `${API_BASE_URL}/analyze`);
        
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
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
        
    } catch (error) {
        console.error('Analysis error:', error);
        
        let errorMessage = 'Failed to analyze classroom. ';
        
        if (error.message.includes('Failed to fetch')) {
            errorMessage += 'Cannot connect to server. Make sure the API server is running on port 5000.';
        } else if (error.message.includes('Model file not found')) {
            errorMessage += 'Engagement model not found. Please ensure best_model_state.bin exists.';
        } else if (error.message.includes('No student folders')) {
            errorMessage += 'No student data found. Please run the tracking system first.';
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
    const { summary, students, merge_info } = data;
    
    // Hide loading, show dashboard
    loadingState.style.display = 'none';
    dashboardContent.style.display = 'block';
    
    // Update summary cards
    updateSummaryCards(summary);
    
    // Update distribution chart
    updateDistributionChart(summary);
    
    // Display student cards
    displayStudentCards(students);
    
    // Display merge info if available
    if (merge_info && merge_info.merges_performed > 0) {
        displayMergeNotification(merge_info);
    }
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
    
    // Use engagement_percentage (primary) or engagement_score (fallback)
    const engagementScore = student.engagement_percentage || student.engagement_score || 0;
    
    // Determine score class based on thresholds
    let scoreClass = 'low';
    if (engagementScore >= 75) scoreClass = 'high';
    else if (engagementScore >= 40) scoreClass = 'medium';
    
    // Determine status badge class
    let statusClass = student.status.toLowerCase().replace(/\s+/g, '-');
    
    // Get image URL
    const imageUrl = student.sample_image 
        ? `${API_BASE_URL}/images/${student.sample_image.replace('students/', '')}`
        : 'placeholder.jpg';
    
    // Get top 3 states for breakdown
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
    } else {
        poseDebugHtml = `
            <div class="pose-debug">
                <div class="pose-debug-title">⚠️ Pose Detection Failed</div>
                <div class="pose-meta">
                    <span class="pose-fallback fallback-yes">Using quality-based fallback</span>
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
    // Convert "student_1" to "Student 1"
    return id.replace('student_', 'Student ');
}

/**
 * Handle errors
 */
function handleError(error) {
    console.error('Error:', error);
    alert('An error occurred. Please try again.');
}

/**
 * Display merge notification
 */
function displayMergeNotification(merge_info) {
    // Check if notification already exists
    let notification = document.getElementById('mergeNotification');
    
    if (!notification) {
        notification = document.createElement('div');
        notification.id = 'mergeNotification';
        notification.className = 'merge-notification';
        
        // Insert after header
        const header = document.querySelector('.header');
        header.parentNode.insertBefore(notification, header.nextSibling);
    }
    
    const { duplicates_found, merges_performed, merge_details } = merge_info;
    
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-check-circle"></i>
            <div class="notification-text">
                <strong>Identity Merge Complete</strong>
                <p>Found ${duplicates_found} duplicate ${duplicates_found === 1 ? 'identity' : 'identities'}, 
                   merged ${merges_performed} student ${merges_performed === 1 ? 'ID' : 'IDs'}</p>
            </div>
            <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    // Auto-hide after 10 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.style.opacity = '0';
            setTimeout(() => notification.remove(), 300);
        }
    }, 10000);
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        analyzeClassroom,
        displayResults,
        formatStudentId
    };
}
