# Quick Start: Recognition System

## 🎯 What's New

The system now recognizes **registered students** instead of using temporary tracking IDs.

**Before**: Track 1, Track 2, Track 3...  
**Now**: Raghav Dhingra, Rahul Kumar, Ananya Sharma...

## 🚀 Quick Start (3 Steps)

### 1. Start Server
```bash
python3 integrated_server.py
```

### 2. Register Students
Open: http://localhost:8080/register.html

- Enter name and student ID
- Capture 10 images from camera OR upload photos
- Click "Register Student"

### 3. Start Monitoring
Open: http://localhost:8080

- Click "Start Camera"
- Registered students appear with **green boxes** and their **names**
- Unknown persons appear with **orange boxes**

## 📋 Commands

### Run Server
```bash
python3 integrated_server.py
```

### Run Tests
```bash
python3 test_recognition_system.py
```

### Check Dependencies
```bash
pip3 show insightface onnxruntime
```

### Install Dependencies (if needed)
```bash
pip3 install insightface onnxruntime
```

## 🌐 URLs

- **Main Dashboard**: http://localhost:8080
- **Registration Page**: http://localhost:8080/register.html
- **Video Stream**: http://localhost:8080/api/video_feed

## 🎨 Visual Indicators

| Color | Meaning |
|-------|---------|
| 🟢 Green Box | Registered student recognized |
| 🟠 Orange Box | Unknown person |

## 📊 What You'll See

### Registered Student
```
┌─────────────────────────────┐
│ 🟢 Raghav Dhingra (95%)     │
│                             │
│         [Face]              │
│                             │
└─────────────────────────────┘
```

### Unknown Person
```
┌─────────────────────────────┐
│ 🟠 Unknown 1                │
│                             │
│         [Face]              │
│                             │
└─────────────────────────────┘
```

## ⚡ Key Features

✅ **Real Student Names**: No more "student_1", "student_2"  
✅ **Stable Identity**: Name persists as student moves  
✅ **Fast Recognition**: Cached for 5 seconds per track  
✅ **Graceful Fallback**: Unknown persons handled safely  
✅ **Session Isolation**: Each session independent  
✅ **Persistent Registry**: Students remembered across sessions  

## 🔧 Performance

- **FPS**: 15-20 (same as before)
- **Recognition**: Every 10 frames per track
- **Cache**: 5 seconds per identity
- **Threshold**: 0.65 cosine similarity

## 📝 Registration Tips

**For best results:**
- Use 10-20 face images per student
- Ensure good lighting
- Capture different angles (front, slight left, slight right)
- Avoid blurry images
- Keep face clearly visible

## 🎯 Testing Checklist

- [ ] Server starts without errors
- [ ] Can register a student
- [ ] Student appears in registered list
- [ ] Camera starts successfully
- [ ] Registered student recognized with green box
- [ ] Name appears correctly
- [ ] Unknown person shows orange box
- [ ] Analytics show real names
- [ ] Video upload works with recognition

## 🚨 Quick Troubleshooting

**Problem**: Recognition not working  
**Fix**: Check InsightFace installed: `pip3 install insightface onnxruntime`

**Problem**: "Camera not running" during capture  
**Fix**: Start camera in main dashboard first

**Problem**: Low FPS  
**Fix**: Normal - recognition is optimized with caching

**Problem**: Wrong person recognized  
**Fix**: Register more images per student (10-20)

## 📚 Full Documentation

- **Testing Guide**: `TESTING_GUIDE.md`
- **Architecture**: `RECOGNITION_INTEGRATION_COMPLETE.md`
- **Main README**: `README.md`

## 🎓 Workflow

```
1. Register Students
   ↓
2. Start Camera/Upload Video
   ↓
3. System Recognizes Faces
   ↓
4. Run Analytics
   ↓
5. View Results with Real Names
```

## 💡 Pro Tips

1. **Register students before monitoring**: Pre-register all students for best results
2. **Use good lighting**: Recognition accuracy depends on lighting
3. **Multiple angles**: Capture faces from different angles during registration
4. **Monitor FPS**: Should stay 15-20 FPS
5. **Check logs**: Server logs show recognition events in real-time

## 🎉 Success Indicators

You'll know it's working when:
- ✅ Green boxes appear on registered students
- ✅ Real names displayed (not "Track 1")
- ✅ Identity stable as student moves
- ✅ Analytics show real student names
- ✅ FPS remains smooth (15-20)

## 📞 Need Help?

1. Run automated tests: `python3 test_recognition_system.py`
2. Check server logs for errors
3. Review `TESTING_GUIDE.md` for detailed debugging
4. Verify InsightFace installed correctly
