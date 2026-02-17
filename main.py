import threading 
import cv2
from deepface import DeepFace
import time

print("Starting camera...")
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit()

print("Camera opened successfully!")

# Give camera time to warm up
time.sleep(2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

counter = 0
face_match = False
check_in_progress = False
face_location = None

print("Loading reference image...")
refrence_img = cv2.imread("refrence.jpg")

if refrence_img is None:
    print("ERROR: Cannot load refrence.jpg!")
    exit()

print("Reference image loaded successfully!")

# Load face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def check_face(frame):
    global face_match, check_in_progress
    try:
        if DeepFace.verify(frame, refrence_img.copy())["verified"]:
            face_match = True
        else:
            face_match = False
    except Exception as e:
        face_match = False
    finally:
        check_in_progress = False

print("Starting main loop...")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("ERROR: Can't receive frame from camera!")
        break

    if ret:
        # Detect faces for tracking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            if face_match:
                # Green border for match
                color = (0, 255, 0)
                label = "MATCH"
                label_color = (0, 255, 0)
            else:
                # Red border for no match
                color = (0, 0, 255)
                label = "NO MATCH"
                label_color = (0, 0, 255)
            
            # Draw rounded rectangle effect with thicker lines
            thickness = 3
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
            
            # Add corner accents for modern look
            corner_length = 30
            corner_thickness = 5
            # Top-left corner
            cv2.line(frame, (x, y), (x + corner_length, y), color, corner_thickness)
            cv2.line(frame, (x, y), (x, y + corner_length), color, corner_thickness)
            # Top-right corner
            cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, corner_thickness)
            cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, corner_thickness)
            # Bottom-left corner
            cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, corner_thickness)
            cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, corner_thickness)
            # Bottom-right corner
            cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, corner_thickness)
            cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, corner_thickness)
            
            # Add label above face
            font = cv2.FONT_HERSHEY_DUPLEX
            label_size = cv2.getTextSize(label, font, 0.9, 2)[0]
            label_y = max(y - 10, label_size[1] + 10)
            
            # Background for text
            cv2.rectangle(frame, 
                         (x, label_y - label_size[1] - 10), 
                         (x + label_size[0] + 10, label_y + 5), 
                         color, -1)
            
            # Text
            cv2.putText(frame, label, (x + 5, label_y), 
                       font, 0.9, (255, 255, 255), 2)
        
        # Perform face verification check
        if counter % 30 == 0 and not check_in_progress:
            try:
                check_in_progress = True
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                check_in_progress = False

        counter += 1
        
        # Add status bar at bottom with elegant font
        status_height = 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - status_height), 
                     (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Status text
        status_text = "● AUTHENTICATED" if face_match else "● SCANNING..."
        status_color = (0, 255, 0) if face_match else (100, 100, 100)
        cv2.putText(frame, status_text, (20, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, status_color, 2)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {counter}", (frame.shape[1] - 200, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Face Recognition System", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()