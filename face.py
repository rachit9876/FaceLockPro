import cv2
import time
import os
import mediapipe as mp
import traceback
import face_recognition
import numpy as np
import logging
import platform

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
EYE_CLOSED_SECONDS = 10
UNAUTHORIZED_LOCK_DELAY = 1
DETECTION_CONFIDENCE = 0.9
FACE_MATCH_TOLERANCE = 0.5
AUTHORIZED_NAME = "Authorized User"
NEAR_THRESHOLD = 0.1

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE_CENTER = 468
RIGHT_EYE_CENTER = 473
NOSE_TIP = 1

def setup_mediapipe():
    try:
        face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.9, min_tracking_confidence=0.9)
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=DETECTION_CONFIDENCE)
        return face_mesh, face_detection
    except Exception as e:
        logging.error(f"Failed to initialize Mediapipe: {e}")
        raise

def align_face(frame, landmarks, width, height):
    try:
        left_eye = landmarks[LEFT_EYE_CENTER]
        right_eye = landmarks[RIGHT_EYE_CENTER]
        left_eye_pos = np.array([left_eye.x * width, left_eye.y * height])
        right_eye_pos = np.array([right_eye.x * width, right_eye.y * height])
        dY = right_eye_pos[1] - left_eye_pos[1]
        dX = right_eye_pos[0] - left_eye_pos[0]
        angle = np.degrees(np.arctan2(dY, dX))
        eyes_center = ((left_eye_pos[0] + right_eye_pos[0]) / 2, (left_eye_pos[1] + right_eye_pos[1]) / 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
        aligned_frame = cv2.warpAffine(frame, M, (width, height))
        aligned_landmarks = []
        for lm in landmarks:
            x, y = lm.x * width, lm.y * height
            new_x = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            new_y = M[1, 0] * x + M[1, 1] * y + M[1, 2]
            aligned_lm = type(lm)(x=new_x / width, y=new_y / height, z=lm.z)
            aligned_landmarks.append(aligned_lm)
        return aligned_frame, aligned_landmarks, M
    except Exception as e:
        logging.error(f"Error in face alignment: {e}")
        return frame, landmarks, np.eye(2, 3)

def lock_system():
    try:
        if platform.system() == "Windows":
            os.system("rundll32.exe user32.dll,LockWorkStation")
        elif platform.system() == "Linux":
            os.system("xdg-screensaver lock || gnome-screensaver-command -l")
        elif platform.system() == "Darwin":
            os.system("/System/Library/CoreServices/Menu\\ Extras/User.menu/Contents/Resources/CGSession -suspend")
        else:
            logging.warning("System lock not supported on this OS")
    except Exception as e:
        logging.error(f"Failed to lock system: {e}")

def get_webcam():
    """Try opening webcam with multiple indices."""
    for index in range(3):  # Try indices 0, 1, 2
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            logging.info(f"Webcam opened successfully on index {index}")
            return cap
        cap.release()
    logging.error("No webcam found on indices 0-2")
    return None

def run_face_lock(preview_enabled=True):
    logging.info(f"Starting face lock with preview_enabled={preview_enabled}")
    face_mesh, face_detection = setup_mediapipe()
    cap = get_webcam()
    if not cap:
        logging.error("Exiting due to webcam failure")
        return

    # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    closed_start_time = None
    unauthorized_start_time = None
    locked = False
    countdown_text = ""
    nose_connections = list(mp_face_mesh.FACEMESH_NOSE)
    reference_encoding = None
    is_face_locked = False

    try:
        while True:
            try:
                success, frame = cap.read()
                if not success:
                    logging.error("Failed to capture frame")
                    break

                logging.debug("Frame captured successfully")
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_results = face_detection.process(rgb_frame)
                mesh_results = face_mesh.process(rgb_frame)

                face_detected = False
                all_faces_authorized = True
                status_text = f"{AUTHORIZED_NAME} Detected" if is_face_locked else "Hold 'L' to lock face"
                status_color = (0, 255, 0)
                current_M = np.eye(2, 3)

                if face_results.detections:
                    face_detected = True
                    for detection in face_results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        x1 = int(np.clip(bboxC.xmin * w, 0, w-1))
                        y1 = int(np.clip(bboxC.ymin * h, 0, h-1))
                        bw = int(bboxC.width * w)
                        bh = int(bboxC.height * h)
                        x2 = int(np.clip(x1 + bw, 0, w-1))
                        y2 = int(np.clip(y1 + bh, 0, h-1))
                        relative_width = bboxC.width

                        if is_face_locked:
                            if relative_width >= NEAR_THRESHOLD:
                                if mesh_results.multi_face_landmarks:
                                    for face_landmarks in mesh_results.multi_face_landmarks:
                                        aligned_frame, aligned_landmarks, M = align_face(rgb_frame, face_landmarks.landmark, w, h)
                                        current_M = M
                                        face_locations = face_recognition.face_locations(aligned_frame)
                                        face_encodings = face_recognition.face_encodings(aligned_frame, face_locations)

                                        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                                            if reference_encoding is None:
                                                continue
                                            matches = face_recognition.compare_faces([reference_encoding], encoding, tolerance=FACE_MATCH_TOLERANCE)
                                            pts = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
                                            inv_M = cv2.invertAffineTransform(M)
                                            transformed_pts = cv2.transform(pts.reshape(1, -1, 2), inv_M).reshape(-1, 2)
                                            x_coords = transformed_pts[:, 0]
                                            y_coords = transformed_pts[:, 1]
                                            x1 = int(np.clip(np.min(x_coords), 0, w-1))
                                            y1 = int(np.clip(np.min(y_coords), 0, h-1))
                                            x2 = int(np.clip(np.max(x_coords), 0, w-1))
                                            y2 = int(np.clip(np.max(y_coords), 0, h-1))

                                            if matches[0]:
                                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            else:
                                                all_faces_authorized = False
                                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            else:
                                all_faces_authorized = False
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            if cv2.waitKey(1) & 0xFF == ord('l'):
                                if mesh_results.multi_face_landmarks:
                                    aligned_frame, _, M = align_face(rgb_frame, mesh_results.multi_face_landmarks[0].landmark, w, h)
                                    face_locations = face_recognition.face_locations(aligned_frame)
                                    if face_locations:
                                        face_encodings = face_recognition.face_encodings(aligned_frame, face_locations)
                                        if face_encodings:
                                            reference_encoding = face_encodings[0]
                                            is_face_locked = True
                                            status_text = "Face Locked"

                if is_face_locked and not all_faces_authorized:
                    if relative_width >= NEAR_THRESHOLD:
                        if unauthorized_start_time is None:
                            unauthorized_start_time = time.time()
                        elapsed = time.time() - unauthorized_start_time
                        remaining = UNAUTHORIZED_LOCK_DELAY - elapsed
                        if remaining > 0:
                            status_text = "Unauthorized Face (Near)"
                            status_color = (0, 0, 255)
                            countdown_text = f"Locking in {remaining:.1f}s"
                        else:
                            countdown_text = "Locking system..."
                            if not locked:
                                lock_system()
                                locked = True
                                is_face_locked = False
                                reference_encoding = None
                                unauthorized_start_time = None
                    else:
                        status_text = "Face Detected (Far)"
                        status_color = (0, 0, 255)
                        countdown_text = "Locking system..."
                        if not locked:
                            lock_system()
                            locked = True
                            is_face_locked = False
                            reference_encoding = None
                            unauthorized_start_time = None
                elif all_faces_authorized:
                    unauthorized_start_time = None

                if mesh_results.multi_face_landmarks and face_detected:
                    for face_landmarks in mesh_results.multi_face_landmarks:
                        for connection in mp_face_mesh.FACEMESH_TESSELATION:
                            start_idx, end_idx = connection
                            start = face_landmarks.landmark[start_idx]
                            end = face_landmarks.landmark[end_idx]
                            x1 = int(np.clip(start.x * w, 0, w-1))
                            y1 = int(np.clip(start.y * h, 0, h-1))
                            x2 = int(np.clip(end.x * w, 0, w-1))
                            y2 = int(np.clip(end.y * h, 0, h-1))
                            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
                        for connection in nose_connections:
                            start_idx, end_idx = connection
                            start = face_landmarks.landmark[start_idx]
                            end = face_landmarks.landmark[end_idx]
                            x1 = int(np.clip(start.x * w, 0, w-1))
                            y1 = int(np.clip(start.y * h, 0, h-1))
                            x2 = int(np.clip(end.x * w, 0, w-1))
                            y2 = int(np.clip(end.y * h, 0, h-1))
                            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if cv2.waitKey(1) & 0xFF == ord('u'):
                    is_face_locked = False
                    reference_encoding = None
                    locked = False
                    status_text = "Face Unlocked"
                    unauthorized_start_time = None

                if not face_detected and not locked:
                    if closed_start_time is None:
                        closed_start_time = time.time()
                    elapsed = time.time() - closed_start_time
                    countdown = EYE_CLOSED_SECONDS - int(elapsed)
                    if countdown > 0:
                        countdown_text = f"System Off in {countdown}s"
                        status_text = "System Off"
                        status_color = (0, 0, 255)
                    else:
                        countdown_text = "Shutting down..."
                        if not locked:
                            lock_system()
                            locked = True
                else:
                    closed_start_time = None
                    if face_detected and all_faces_authorized:
                        locked = False
                        countdown_text = ""

                if preview_enabled.value if hasattr(preview_enabled, "value") else preview_enabled:
                    logging.debug("Attempting to display preview window")
                    cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, status_color, 3)
                    if countdown_text:
                        cv2.putText(frame, countdown_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                    cv2.imshow("Neural Face Lock System", frame)
                    cv2.waitKey(1)  # Ensure OpenCV event loop processes
                else:
                    # Hide the window if it exists
                    if cv2.getWindowProperty("Neural Face Lock System", cv2.WND_PROP_VISIBLE) >= 1:
                        cv2.destroyWindow("Neural Face Lock System")
                    cv2.waitKey(1)  # Process OpenCV events even when preview is off

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                logging.error(f"Main loop error: {e}")
                traceback.print_exc()
                continue

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        face_detection.close()
        logging.info("Resources cleaned up.")

if __name__ == '__main__':
    try:
        run_face_lock()
    except ImportError as e:
        logging.error(f"Missing required library: {e}")
        logging.error("Please install dependencies: pip install opencv-python mediapipe face_recognition numpy")
    except Exception as e:
        logging.error(f"Startup error: {e}")