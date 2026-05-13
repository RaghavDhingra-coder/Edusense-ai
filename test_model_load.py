"""
Test YOLO model loading with PyTorch 2.6+ safe globals
"""

import torch
print(f"PyTorch version: {torch.__version__}")

try:
    # Add all required safe globals
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.modules.block import C2f, SPPF, Bottleneck
    from ultralytics.nn.modules.head import Detect
    from torch.nn.modules.container import Sequential, ModuleList
    from torch.nn.modules.conv import Conv2d
    from torch.nn.modules.batchnorm import BatchNorm2d
    from torch.nn.modules.activation import SiLU
    from torch.nn import Module
    
    torch.serialization.add_safe_globals([
        DetectionModel,
        Conv,
        C2f,
        SPPF,
        Bottleneck,
        Detect,
        Sequential,
        ModuleList,
        Conv2d,
        BatchNorm2d,
        SiLU,
        Module,
    ])
    print("✅ Added safe globals")
    
    # Try loading model
    from ultralytics import YOLO
    print("🔄 Loading model...")
    model = YOLO("yolov8n-face.pt")
    print("✅ Model loaded successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
