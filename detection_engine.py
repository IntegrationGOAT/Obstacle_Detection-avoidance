from ultralytics import YOLO
import cv2
import numpy as np

# -------- OBJECT GROUPS --------
PEOPLE = ["person"]

FURNITURE = ["chair", "couch", "bed", "dining table"]

VEHICLES = ["car", "motorbike", "bus", "truck", "bicycle"]

COMMON_OBSTACLES = [
    "bottle", "backpack", "handbag", "suitcase",
    "tv", "laptop", "keyboard", "cell phone",
    "book", "cup"
]

IMPORTANT_OBJECTS = PEOPLE + FURNITURE + VEHICLES + COMMON_OBSTACLES


class DetectionEngine:
    """YOLOv8-based object detection engine with guidance generation."""
    
    def __init__(self, model_path="yolov8n.pt"):
        """Initialize the detection model.
        
        Args:
            model_path: Path to the YOLOv8 model file
        """
        self.model = YOLO(model_path)
    
    def detect_frame(self, frame, confidence_threshold=0.5, min_width=40):
        """Run detection on a single frame.
        
        Args:
            frame: Input frame from OpenCV (numpy array)
            confidence_threshold: Minimum confidence score (0-1)
            min_width: Minimum bounding box width to consider
            
        Returns:
            dict: Contains 'closest_obj' (tuple or None), 'all_detections' (list), 'frame_shape'
        """
        results = self.model(frame, verbose=False)
        h, w, _ = frame.shape
        
        closest_obj = None
        max_width = 0
        all_detections = []
        
        # Find closest important object
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = self.model.names[cls]
                
                width = x2 - x1
                
                if width < min_width:
                    continue
                
                if label not in IMPORTANT_OBJECTS:
                    continue
                
                # Store detection
                all_detections.append({
                    'label': label,
                    'box': (x1, y1, x2, y2),
                    'confidence': conf,
                    'width': width
                })
                
                # Track closest
                if width > max_width:
                    max_width = width
                    closest_obj = (label, x1, y1, x2, y2)
        
        return {
            'closest_obj': closest_obj,
            'all_detections': all_detections,
            'frame_shape': (h, w)
        }
    
    @staticmethod
    def get_distance(width, thresholds=(220, 120)):
        """Classify distance based on bounding box width.
        
        Args:
            width: Bounding box width
            thresholds: (very_close_threshold, near_threshold)
            
        Returns:
            str: Distance classification ('very close', 'near', 'far')
        """
        if width > thresholds[0]:
            return "very close"
        elif width > thresholds[1]:
            return "near"
        else:
            return "far"
    
    @staticmethod
    def get_direction(center_x, frame_width):
        """Classify direction based on horizontal position.
        
        Args:
            center_x: Center X coordinate of bounding box
            frame_width: Width of frame
            
        Returns:
            str: Direction classification ('left', 'center', 'right')
        """
        if center_x < frame_width / 3:
            return "left"
        elif center_x > 2 * frame_width / 3:
            return "right"
        else:
            return "center"
    
    @staticmethod
    def get_object_type(label):
        """Classify object type.
        
        Args:
            label: Object label from YOLO
            
        Returns:
            str: Object type ('vehicle', 'person', 'furniture', 'object')
        """
        if label in VEHICLES:
            return "vehicle"
        elif label in PEOPLE:
            return "person"
        elif label in FURNITURE:
            return "furniture"
        else:
            return "object"
    
    @staticmethod
    def generate_guidance(label, obj_type, distance, direction):
        """Generate guidance text based on detected object.
        
        Args:
            label: Object label
            obj_type: Object type classification
            distance: Distance classification
            direction: Direction classification
            
        Returns:
            str: Guidance text for the user
        """
        if obj_type == "vehicle":
            if distance == "very close":
                return f"Stop immediately, {label} very close ahead"
            elif distance == "near":
                return f"Warning, {label} approaching on {direction}"
            else:
                return f"{label} detected ahead"
        
        elif obj_type == "person":
            if distance == "very close":
                return f"Person very close on {direction}, slow down"
            else:
                return f"Person on {direction}"
        
        else:  # furniture, obstacles, other objects
            if distance == "very close":
                if direction == "left":
                    return "Obstacle very close on left, move right"
                elif direction == "right":
                    return "Obstacle very close on right, move left"
                else:
                    return "Obstacle ahead, stop or move sideways"
            elif distance == "near":
                return f"Obstacle nearby on {direction}"
            else:
                return "Path clear, move forward"
        
        return "Path clear, move forward"
    
    @staticmethod
    def draw_annotations(frame, closest_obj, frame_width):
        """Draw bounding box and text annotations on frame.
        
        Args:
            frame: Input frame (will be modified in place)
            closest_obj: Tuple of (label, x1, y1, x2, y2) or None
            frame_width: Width of frame for centering text
            
        Returns:
            frame: Annotated frame
        """
        if closest_obj:
            label, x1, y1, x2, y2 = closest_obj
            
            # Determine color based on distance
            width = x2 - x1
            if width > 220:
                color = (0, 0, 255)  # Red for very close
            else:
                color = (0, 255, 255)  # Yellow
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with distance and direction
            distance = DetectionEngine.get_distance(width)
            center_x = (x1 + x2) // 2
            direction = DetectionEngine.get_direction(center_x, frame_width)
            
            text = f"{label} | {distance} | {direction}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
