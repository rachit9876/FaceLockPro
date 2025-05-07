from pystray import Icon, MenuItem, Menu
from multiprocessing import Process, Value
from PIL import Image
import face
import os
import sys
import logging
import psutil

# Set environment variable
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

face_proc = None
preview_enabled = Value('b', True)  # Shared boolean for preview state
icon_instance = None

def is_process_running(name):
    """Check if a process with the given name is running (excluding this process)."""
    current_pid = os.getpid()
    for proc in psutil.process_iter(['name', 'pid']):
        if proc.info['name'].lower() == name.lower() and proc.info['pid'] != current_pid:
            return True
    return False

def start_face_lock(icon, item):
    global face_proc
    if face_proc is None or not face_proc.is_alive():
        logging.info("Starting face lock process")
        face_proc = Process(target=face.run_face_lock, args=(preview_enabled,))
        face_proc.start()
        icon.notify("Face Lock Started", "Neural Face Lock")
    else:
        logging.info("Face lock process already running")
        icon.notify("Face Lock Already Running", "Neural Face Lock")

def stop_face_lock(icon, item):
    global face_proc
    if face_proc and face_proc.is_alive():
        logging.info("Stopping face lock process")
        face_proc.terminate()
        face_proc.join(timeout=2)
        if face_proc.is_alive():
            logging.warning("Force terminating face lock process")
            face_proc.kill()
        face_proc = None
        icon.notify("Face Lock Stopped", "Neural Face Lock")
    else:
        logging.info("No face lock process running")
        icon.notify("No Face Lock Running", "Neural Face Lock")

def get_preview_text(item=None):
    """Return the current preview state text."""
    return f"Preview {'On' if preview_enabled.value else 'Off'}"

def toggle_preview(icon, item):
    with preview_enabled.get_lock():
        preview_enabled.value = not preview_enabled.value
    logging.info(f"Preview toggled to {'On' if preview_enabled.value else 'Off'}")
    icon.notify(f"Preview {'On' if preview_enabled.value else 'Off'}", "Neural Face Lock")

def quit_app(icon, item):
    logging.info("Exiting application")
    stop_face_lock(icon, item)
    icon.stop()

# Resolve icon path
script_dir = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(script_dir, 'icon.png')
try:
    icon_image = Image.open(icon_path)
except FileNotFoundError:
    logging.warning("Icon file not found, using default image")
    icon_image = Image.new("RGB", (64, 64), (0, 0, 0))

menu = Menu(
    MenuItem("Start", start_face_lock),
    MenuItem("Stop", stop_face_lock),
    MenuItem(get_preview_text, toggle_preview),
    MenuItem("Exit", quit_app)
)

if __name__ == '__main__':
    try:
        # Prevent multiple instances
        if is_process_running("python.exe") and icon_instance is not None:
            logging.error("Application already running")
            sys.exit(1)

        icon_instance = Icon("Neural Face Lock", icon_image, menu=menu)
        icon_instance.run()
    except Exception as e:
        logging.error(f"System tray error: {e}")
        sys.exit(1)
    finally:
        if icon_instance:
            icon_instance.stop()