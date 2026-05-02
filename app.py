import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from detection_engine import DetectionEngine
from audio_helper import AudioHelper
import os
from threading import Lock

# ====== THREAD-SAFE COUNTERS ======
frame_lock = Lock()
detection_data = {
    'frame_count': 0,
    'detection_count': 0,
    'last_guidance': '🟢 Path Clear - Move Forward',
    'guidance_level': 'clear'
}

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="NavSight AI - Real-Time Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 3em;
        font-weight: bold;
        background: linear-gradient(90deg, #00D9FF, #0099FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 10px;
        animation: glow 2s ease-in-out infinite;
    }
    @keyframes glow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1em;
        margin-bottom: 20px;
    }
    .guidance-clear {
        background: linear-gradient(135deg, #0E4620 0%, #0A7E3E 100%);
        border-left: 6px solid #00FF41;
        padding: 20px;
        border-radius: 8px;
        font-size: 1.3em;
        font-weight: bold;
        color: #00FF41;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 255, 65, 0.3);
    }
    .guidance-caution {
        background: linear-gradient(135deg, #4D3300 0%, #663300 100%);
        border-left: 6px solid #FFA500;
        padding: 20px;
        border-radius: 8px;
        font-size: 1.3em;
        font-weight: bold;
        color: #FFB74D;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(255, 165, 0, 0.3);
    }
    .guidance-danger {
        background: linear-gradient(135deg, #4D0000 0%, #800000 100%);
        border-left: 6px solid #FF0000;
        padding: 20px;
        border-radius: 8px;
        font-size: 1.3em;
        font-weight: bold;
        color: #FF6B6B;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(255, 0, 0, 0.4);
        animation: pulse 0.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    .stats-box {
        background: #1a1a2e;
        border: 2px solid #00D9FF;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .object-card {
        background: #16213e;
        border-left: 4px solid #00D9FF;
        padding: 12px;
        border-radius: 5px;
        margin: 8px 0;
    }
    [data-testid="stElementContainer"] video {
        max-width: 50%;
        height: auto;
        aspect-ratio: 1/1;
        object-fit: contain;
    }
    </style>
""", unsafe_allow_html=True)

# ====== INITIALIZE DETECTION ENGINE ======
model_path = "yolov8n.pt"
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: {model_path}")
    st.stop()

@st.cache_resource
def load_detection_engine():
    return DetectionEngine(model_path)

@st.cache_resource
def load_audio_helper():
    return AudioHelper()

engine = load_detection_engine()
audio_helper = load_audio_helper()

# ====== SIDEBAR CONFIG ======
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    confidence_threshold = st.slider(
        "🎯 Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    min_width = st.slider(
        "📏 Min Object Width",
        min_value=10,
        max_value=200,
        value=40,
        step=10,
        help="Minimum bounding box width"
    )
    
    enable_audio = st.checkbox(
        "🔊 Audio Guidance",
        value=audio_helper.available,
        help="Enable voice alerts" if audio_helper.available else "Audio unavailable (headless environment)",
        disabled=not audio_helper.available
    )
    
    if not audio_helper.available:
        st.warning("🎙️ **Audio unavailable** — Running on headless server (no speakers). Audio only works locally.", icon="⚠️")
    
    enable_boxes = st.checkbox(
        "📦 Draw Bounding Boxes",
        value=True,
        help="Show detection boxes"
    )
    
    enable_crosshair = st.checkbox(
        "➕ Show Crosshair",
        value=True,
        help="Center reference line"
    )


# ====== MAIN INTERFACE ======
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown('<div class="main-title">🎯 NavSight AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Real-Time Object Detection & Navigation Guidance</div>', unsafe_allow_html=True)

# ====== CALLBACK FUNCTION FOR VIDEO PROCESSING ======
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Process each video frame with YOLO detection."""
    
    # Convert to OpenCV format
    img = frame.to_ndarray(format="bgr24")
    h, w, _ = img.shape
    
    # Run detection
    detection_results = engine.detect_frame(img, confidence_threshold, min_width)
    closest_obj = detection_results['closest_obj']
    all_detections = detection_results['all_detections']
    
    # Update thread-safe detection data
    with frame_lock:
        detection_data['detection_count'] = len(all_detections)
        detection_data['frame_count'] += 1
    
    # Generate guidance
    guidance_text = "🟢 Path Clear - Move Forward"
    guidance_level = "clear"
    
    if closest_obj:
        label, x1, y1, x2, y2 = closest_obj
        box_width = x2 - x1
        
        distance = engine.get_distance(box_width)
        center_x = (x1 + x2) // 2
        direction = engine.get_direction(center_x, w)
        obj_type = engine.get_object_type(label)
        
        guidance_text = engine.generate_guidance(label, obj_type, distance, direction)
        
        # Determine urgency level
        if "Stop immediately" in guidance_text or distance == "very close":
            guidance_level = "danger"
        elif "Caution" in guidance_text or distance == "near":
            guidance_level = "caution"
        else:
            guidance_level = "clear"
    
    # Store for display (thread-safe)
    with frame_lock:
        detection_data['last_guidance'] = guidance_text
        detection_data['guidance_level'] = guidance_level
    
    # Draw bounding boxes
    if enable_boxes and all_detections:
        for detection in all_detections:
            x1, y1, x2, y2 = detection['box']
            label = detection['label']
            confidence = detection['confidence']
            
            # Draw box
            color = (0, 255, 0) if guidance_level == "clear" else (0, 165, 255) if guidance_level == "caution" else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with background
            text = f"{label} {confidence:.1%}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)
            cv2.putText(img, text, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw crosshair at center
    if enable_crosshair:
        center_x, center_y = w // 2, h // 2
        cv2.line(img, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 2)
        cv2.line(img, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 2)
        cv2.circle(img, (center_x, center_y), 5, (0, 255, 255), -1)
    
    # Audio alert (use thread-safe detection_data)
    with frame_lock:
        current_guidance_level = detection_data['guidance_level']
    
    # Trigger audio for ANY detection (not just danger/caution)
    if enable_audio and audio_helper.available and closest_obj:
        if audio_helper.should_speak(guidance_text):
            audio_helper.speak(guidance_text, async_mode=True)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ====== WEBRTC CONFIGURATION ======
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ====== VIDEO STREAM ======
st.markdown("### 📹 Live Detection Stream")
webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)