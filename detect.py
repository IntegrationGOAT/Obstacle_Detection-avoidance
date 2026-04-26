from ultralytics import YOLO
import cv2
import pyttsx3
import time

# -------- INIT --------
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

engine = pyttsx3.init()
engine.setProperty('rate', 160)

last_spoken = ""
last_time = 0
memory = {}  # remembers last spoken per object

# -------- OBJECT GROUPS --------
people = ["person"]

furniture = ["chair", "couch", "bed", "dining table"]

vehicles = ["car", "motorbike", "bus", "truck", "bicycle"]

common_obstacles = [
    "bottle", "backpack", "handbag", "suitcase",
    "tv", "laptop", "keyboard", "cell phone",
    "book", "cup"
]

important_objects = people + furniture + vehicles + common_obstacles

# -------- MAIN LOOP --------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    h, w, _ = frame.shape

    closest_obj = None
    max_width = 0

    # -------- FIND CLOSEST IMPORTANT OBJECT --------
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            width = x2 - x1

            if width < 40:
                continue

            if label not in important_objects:
                continue

            if width > max_width:
                max_width = width
                closest_obj = (label, x1, y1, x2, y2)

    # -------- DEFAULT --------
    guidance_text = "Path clear, move forward"

    if closest_obj:
        label, x1, y1, x2, y2 = closest_obj

        width = x2 - x1

        # -------- DISTANCE --------
        if width > 220:
            distance = "very close"
        elif width > 120:
            distance = "near"
        else:
            distance = "far"

        # -------- DIRECTION --------
        center_x = (x1 + x2) // 2
        if center_x < w / 3:
            direction = "left"
        elif center_x > 2 * w / 3:
            direction = "right"
        else:
            direction = "center"

        # -------- CLASSIFY --------
        if label in vehicles:
            obj_type = "vehicle"
        elif label in people:
            obj_type = "person"
        elif label in furniture:
            obj_type = "furniture"
        else:
            obj_type = "object"

        # -------- GUIDANCE LOGIC --------
        if obj_type == "vehicle":
            if distance == "very close":
                guidance_text = f"Stop immediately, {label} very close ahead"
            elif distance == "near":
                guidance_text = f"Warning, {label} approaching on {direction}"
            else:
                guidance_text = f"{label} detected ahead"

        elif obj_type == "person":
            if distance == "very close":
                guidance_text = f"Person very close on {direction}, slow down"
            else:
                guidance_text = f"Person on {direction}"

        else:
            if distance == "very close":
                if direction == "left":
                    guidance_text = "Obstacle very close on left, move right"
                elif direction == "right":
                    guidance_text = "Obstacle very close on right, move left"
                else:
                    guidance_text = "Obstacle ahead, stop or move sideways"
            elif distance == "near":
                guidance_text = f"Obstacle nearby on {direction}"

        # -------- DRAW --------
        color = (0, 255, 255)
        if distance == "very close":
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} | {distance} | {direction}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # -------- MEMORY + VOICE CONTROL --------
    current_time = time.time()

    # only speak if changed or enough time passed
    if (guidance_text != last_spoken) and (current_time - last_time > 2):

        # prevent repeating same sentence too often
        if guidance_text not in memory or (current_time - memory[guidance_text] > 5):
            engine.stop()
            engine.say(guidance_text)
            engine.runAndWait()

            memory[guidance_text] = current_time
            last_spoken = guidance_text
            last_time = current_time

    # -------- UI --------
    cv2.putText(frame, "NAVSIGHT AI - SMART MODE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, guidance_text, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("NavSight AI", frame)

    if cv2.waitKey(1) == 27:
        break

# -------- CLEANUP --------
cap.release()
cv2.destroyAllWindows()