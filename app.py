#!/usr/bin/env python3
import os
import json
import pandas as pd
import cv2
import numpy as np
import face_recognition
import threading
import queue
import time
import pickle
import hashlib
from datetime import datetime, date
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from dotenv import load_dotenv
load_dotenv()

from werkzeug.utils import secure_filename

try:
    from twilio.rest import Client as TwilioClient
except Exception:
    TwilioClient = None

# ---------- Config / globals ----------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "replace-with-a-secret-key")

# allow up to 50 MB uploads (adjust as needed)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# allowed upload file extensions (used with filename.lower().endswith(ALLOWED_EXT))
ALLOWED_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

# directory where uploaded class photos (and JSON reports) are saved
CLASS_PHOTOS_DIR = os.path.join(os.getcwd(), 'class_photos')
os.makedirs(CLASS_PHOTOS_DIR, exist_ok=True)

# cache/encodings globals
_encodings_lock = threading.Lock()
_known_encodings_by_name = {}
_encodings_ready = threading.Event()
_encodings_cache_file = os.path.join(os.getcwd(), 'encodings.pkl')

_camera_lock = threading.Lock()
DEFAULT_CAPTURE_WIDTH = 1280
DEFAULT_CAPTURE_HEIGHT = 720

attendance_queue = queue.Queue(maxsize=2000)
NAME_COOLDOWN_SECONDS = 60
_recent_mark_times = {}

stop_recognition = False
stop_lock = threading.Lock()

MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", 0.50))
MATCH_MARGIN = float(os.environ.get("MATCH_MARGIN", 0.08))
AUTO_NOTIFY_ABSENTEES = os.environ.get("AUTO_NOTIFY_ABSENTEES", "true").lower() in ("1", "true", "yes")

# SMS / Twilio config
ENABLE_SMS = os.environ.get("ENABLE_SMS", "true").lower() in ("1", "true", "yes")
TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.environ.get("TWILIO_FROM")


# ---------- utilities ----------
def _training_folder_hash(path):
    items = []
    try:
        for fname in sorted(os.listdir(path)):
            full = os.path.join(path, fname)
            try:
                st = os.path.getmtime(full)
            except Exception:
                st = 0
            items.append(f"{fname}:{int(st)}")
    except Exception:
        return None
    return hashlib.md5(("|".join(items)).encode()).hexdigest()


def load_cached_encodings(path):
    if not os.path.exists(_encodings_cache_file):
        return None, None
    try:
        with open(_encodings_cache_file, 'rb') as f:
            data = pickle.load(f)
        if data.get('folder_hash') != _training_folder_hash(path):
            return None, None
        return data.get('enc_by_name'), data.get('names')
    except Exception as e:
        print("Failed to load enc cache:", e)
        return None, None


def save_cached_encodings(path, enc_by_name, names):
    try:
        data = {'folder_hash': _training_folder_hash(path), 'enc_by_name': enc_by_name, 'names': names}
        with open(_encodings_cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print("Failed to save enc cache:", e)


def ensure_encodings_ready(wait_seconds=3.0):
    """
    Ensure _known_encodings_by_name is populated.
    - Tries to load cached encodings first.
    - If not available, starts background compute and waits up to wait_seconds for it to finish.
    Returns True if encodings are ready (non-empty), False otherwise.
    """
    path = os.path.join(os.getcwd(), 'Training images')

    # Try to load cache quickly
    try:
        cached_by_name, cached_names = load_cached_encodings(path)
    except Exception:
        cached_by_name, cached_names = None, None

    if cached_by_name is not None and len(cached_by_name) > 0:
        with _encodings_lock:
            global _known_encodings_by_name
            _known_encodings_by_name = cached_by_name
            _encodings_ready.set()
        return True

    # If encodings already computed and event set, trust it
    if _encodings_ready.is_set():
        return len(_known_encodings_by_name) > 0

    # Start background compute (daemon) and wait a short time
    _encodings_ready.clear()
    bg = threading.Thread(target=compute_encodings_background, args=(path,), daemon=True)
    bg.start()

    waited = 0.0
    step = 0.2
    while waited < wait_seconds and not _encodings_ready.is_set():
        time.sleep(step)
        waited += step

    # final check
    with _encodings_lock:
        return len(_known_encodings_by_name) > 0


# ---------- face helpers ----------
def safe_face_encodings(img):
    try:
        encs = face_recognition.face_encodings(img)
        return encs if encs is not None else []
    except Exception as e:
        print("face_encodings error:", e)
        return []


def compute_encodings_background(path):
    """
    Walk the training images folder recursively and compute per-person encodings.
    Each image file becomes a candidate encoding under a root name (strip trailing _N digits).
    This runs in background and updates _known_encodings_by_name when done.
    """
    global _known_encodings_by_name
    by_name = {}
    img_paths = []
    for root, dirs, files in os.walk(path):
        for f in sorted(files):
            if f.lower().endswith(ALLOWED_EXT):
                img_paths.append(os.path.join(root, f))

    print("Background: computing encodings for", len(img_paths), "files (recursive)")
    for full in img_paths:
        img = cv2.imread(full)
        if img is None:
            continue
        try:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            continue
        encs = safe_face_encodings(rgb)
        if len(encs) == 0:
            print("Warning: no face found in", full)
            continue
        base = os.path.splitext(os.path.basename(full))[0]
        parts = base.rsplit('_', 1)
        root_name = base
        if len(parts) == 2 and parts[1].isdigit():
            root_name = parts[0]
        key = root_name
        by_name.setdefault(key, []).append(encs[0])

    with _encodings_lock:
        _known_encodings_by_name = by_name
        try:
            save_cached_encodings(path, by_name, list(by_name.keys()))
        except Exception:
            pass
        _encodings_ready.set()
    print("Background: encodings ready for", len(by_name), "people")


# ---------- CSV-only attendance writers ----------
def markAttendanceCSV(name):
    today = date.today().isoformat()
    filename = 'attendance.csv'
    if not os.path.exists(filename):
        with open(filename, 'w', errors='ignore') as f:
            f.write("name,time,date\n")
    try:
        # Append if not present today
        with open(filename, 'r+', errors='ignore') as f:
            lines = f.readlines()
            present = set()
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 3 and parts[2] == today:
                    present.add(parts[0])
            if name not in present:
                dtString = datetime.now().strftime('%H:%M')
                f.write(f'{name},{dtString},{today}\n')
                print(f"attendance.csv: wrote {name},{dtString},{today}")
    except Exception as e:
        print("markAttendanceCSV error:", e)


def markTeacherAttendanceCSV(name):
    today = date.today().isoformat()
    filename = 'attendance_teachers.csv'
    if not os.path.exists(filename):
        with open(filename, 'w', errors='ignore') as f:
            f.write("name,time,date\n")
    try:
        with open(filename, 'r+', errors='ignore') as f:
            lines = f.readlines()
            present = set()
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 3 and parts[2] == today:
                    present.add(parts[0])
            if name not in present:
                dtString = datetime.now().strftime('%H:%M')
                f.write(f'{name},{dtString},{today}\n')
                print(f"attendance_teachers.csv: wrote {name},{dtString},{today}")
    except Exception as e:
        print("markTeacherAttendanceCSV error:", e)


# ---------- attendance writer worker ----------
def attendance_worker():
    while True:
        item = attendance_queue.get()
        try:
            if isinstance(item, tuple) and len(item) == 2:
                name, kind = item
            else:
                name, kind = item, 'student'
            now = time.time()
            key = (name, kind)
            last = _recent_mark_times.get(key, 0)
            if now - last >= 0:
                if kind == 'teacher':
                    try:
                        markTeacherAttendanceCSV(name)
                    except Exception as e:
                        print("Error writing teacher CSV:", e)
                else:
                    try:
                        markAttendanceCSV(name)
                    except Exception as e:
                        print("Error writing student CSV:", e)
                _recent_mark_times[key] = now
                if len(_recent_mark_times) > 2000:
                    cutoff = now - 24 * 3600
                    for k, v in list(_recent_mark_times.items()):
                        if v < cutoff:
                            del _recent_mark_times[k]
        except Exception as e:
            print("attendance_worker error:", e)
        finally:
            attendance_queue.task_done()


_att_thread = threading.Thread(target=attendance_worker, daemon=True)
_att_thread.start()


# ---------- Routes ----------
@app.route('/new', methods=['GET'])
def new_page():
    return render_template('new.html')


@app.route('/index', methods=['GET'])
def index_page():
    return render_template('index.html')


@app.route('/name', methods=['POST'])
def name_capture():
    name1 = request.form.get('name1', '').strip()
    extra = request.form.get('name2', '').strip()
    parent_phone_raw = request.form.get('parent_phone', '').strip()

    if not name1:
        return "Name is required", 400
    if not parent_phone_raw:
        return "Parent phone is required", 400

    parent_phone_norm = normalize_phone_number(parent_phone_raw, default_region='IN')
    if not parent_phone_norm:
        digits = ''.join(ch for ch in parent_phone_raw if ch.isdigit())
        if len(digits) >= 7:
            parent_phone_norm = '+' + digits if not parent_phone_raw.startswith('+') else parent_phone_raw
            print(f"Warning: phone normalization failed for '{parent_phone_raw}', using fallback '{parent_phone_norm}'")
        else:
            parent_phone_norm = parent_phone_raw
            print(f"Warning: phone looks invalid and couldn't normalize: '{parent_phone_raw}'")

    csv_path = os.path.join(os.getcwd(), 'students.csv')
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, dtype=str).fillna('')
        else:
            df = pd.DataFrame(columns=['name', 'parent_phone', 'extra'])

        mask = (df['name'].astype(str).str.strip() == name1)
        if mask.any():
            idx = df.index[mask][0]
            df.loc[idx, 'parent_phone'] = parent_phone_norm
            if extra:
                df.loc[idx, 'extra'] = extra
            print(f"Updated student contact for {name1} in students.csv")
        else:
            new_row = {'name': name1, 'parent_phone': parent_phone_norm, 'extra': extra}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"Added new student {name1} to students.csv")

        df.to_csv(csv_path, index=False)
    except Exception as e:
        print("Error updating students.csv:", e)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return "Cannot open camera", 500
    window_title = "Press Space to capture image"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    try:
        try:
            cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            cv2.imshow(window_title, frame)
            k = cv2.waitKey(1)
            if k % 256 == 27:
                break
            elif k % 256 == 32:
                img_name = f"{name1}.png"
                path = os.path.join(os.getcwd(), 'Training images')
                os.makedirs(path, exist_ok=True)
                cv2.imwrite(os.path.join(path, img_name), frame)
                print("{} written!".format(img_name))
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
    return render_template('image.html')


try:
    import phonenumbers
except Exception:
    phonenumbers = None


def normalize_phone_number(raw_phone, default_region='IN'):
    if raw_phone is None:
        return ''
    s = str(raw_phone).strip()
    if not s:
        return ''
    if s.startswith('+') and s[1:].replace(' ', '').isdigit():
        return s.replace(' ', '')
    if phonenumbers:
        try:
            parsed = phonenumbers.parse(s, default_region)
            if phonenumbers.is_possible_number(parsed) and phonenumbers.is_valid_number(parsed):
                return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        except Exception:
            pass
    digits = ''.join(ch for ch in s if ch.isdigit())
    if len(digits) == 10 and default_region == 'IN':
        return '+91' + digits
    if 11 <= len(digits) <= 15:
        return '+' + digits
    return ''


@app.route("/", methods=["GET", "POST"])
def recognize():
    global stop_recognition
    if request.method == "POST":
        path = os.path.join(os.getcwd(), 'Training images')
        os.makedirs(path, exist_ok=True)

        cached_by_name, cached_names = load_cached_encodings(path)
        if cached_by_name is not None and len(cached_by_name) > 0:
            with _encodings_lock:
                global _known_encodings_by_name
                _known_encodings_by_name = cached_by_name
                _encodings_ready.set()
            print("Loaded encodings from cache:", len(_known_encodings_by_name))
        else:
            _encodings_ready.clear()

        if not _encodings_ready.is_set():
            bg = threading.Thread(target=compute_encodings_background, args=(path,), daemon=True)
            bg.start()

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Cannot open camera", 500

        window_name = 'Punch your Attendance'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass

        waited = 0.0
        wait_step = 0.1
        max_wait = 5.0
        while not _encodings_ready.is_set() and waited < max_wait:
            ret, preview = cap.read()
            if not ret:
                break
            cv2.putText(preview, "Loading database... please wait", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow(window_name, preview)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                return render_template('first.html', recognized=None)
            time.sleep(wait_step)
            waited += wait_step

        with _encodings_lock:
            enc_by_name = dict(_known_encodings_by_name)
        name_list = list(enc_by_name.keys())
        encs_list = [enc_by_name[n] for n in name_list]

        if len(name_list) == 0:
            ret, img = cap.read()
            if ret:
                cv2.putText(img, "No training images found. Press ESC.", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow(window_name, img)
                cv2.waitKey(2000)
            cap.release()
            cv2.destroyAllWindows()
            return "No training images found. Add images and try again."

        process_every_n_frames = 2
        frame_count = 0
        recognized_set = set()
        per_name_cooldown = NAME_COOLDOWN_SECONDS

        try:
            while True:
                with stop_lock:
                    if stop_recognition:
                        stop_recognition = False
                        print("Stop requested, exiting recognition loop.")
                        break

                success, img = cap.read()
                if not success:
                    print("Failed to read from camera")
                    break
                frame_count += 1
                status_text = f"Marked: {len(recognized_set)}"

                if frame_count % process_every_n_frames == 0:
                    small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
                    rgb_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
                    faces = face_recognition.face_locations(rgb_small, model='hog')
                    encs = face_recognition.face_encodings(rgb_small, faces)

                    for encodeFace, faceLoc in zip(encs, faces):
                        per_name_min = []
                        for enc_list in encs_list:
                            if not enc_list:
                                per_name_min.append(np.inf)
                                continue
                            try:
                                dists = face_recognition.face_distance(enc_list, encodeFace)
                                per_name_min.append(float(np.min(dists)))
                            except Exception:
                                per_name_min.append(np.inf)

                        per_name_min = np.array(per_name_min)
                        idx_sorted = np.argsort(per_name_min)
                        best_idx = int(idx_sorted[0])
                        best_name_key = name_list[best_idx]
                        best_dist = float(per_name_min[best_idx])
                        second_dist = float(per_name_min[idx_sorted[1]]) if len(idx_sorted) > 1 else np.inf

                        matched = False
                        matched_teacher = False
                        display_name = None

                        if best_dist < MATCH_THRESHOLD and (second_dist - best_dist) >= MATCH_MARGIN:
                            matched = True
                            if best_name_key.startswith('T_'):
                                matched_teacher = True
                                display_name = best_name_key[2:]
                            else:
                                display_name = best_name_key

                            key = (display_name, 'teacher' if matched_teacher else 'student')
                            nowt = time.time()
                            last = _recent_mark_times.get(key, 0)
                            if nowt - last >= per_name_cooldown:
                                recognized_set.add(display_name)
                                _recent_mark_times[key] = nowt
                                try:
                                    attendance_queue.put_nowait((display_name, 'teacher' if matched_teacher else 'student'))
                                    print("Enqueued for marking:", display_name, 'teacher' if matched_teacher else 'student')
                                except queue.Full:
                                    print("attendance_queue full, skipping", display_name)
                        else:
                            display_name = 'Unknown'

                        top, right, bottom, left = faceLoc
                        top *= 4; right *= 4; bottom *= 4; left *= 4
                        label = display_name if display_name is not None else 'Unknown'
                        if label != 'Unknown' and matched_teacher:
                            label = f"{label} (Teacher)"
                        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow(window_name, img)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('s')):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

            # automatic absent notifications (SMS-only)
            if AUTO_NOTIFY_ABSENTEES:
                try:
                    wait_start = time.time()
                    max_wait_seconds = 10
                    while not attendance_queue.empty() and (time.time() - wait_start) < max_wait_seconds:
                        time.sleep(0.2)

                    try:
                        results = send_absent_notifications_for_date(date.today())
                        print("Auto absent notifications result:", results)
                    except Exception as e:
                        print("Error while sending auto absent notifications:", e)
                except Exception as e:
                    print("Unexpected error in auto-notify section:", e)

        return render_template('first.html', recognized=None)

    else:
        return render_template('main.html')


@app.route('/stop', methods=['POST'])
def stop_route():
    global stop_recognition
    with stop_lock:
        stop_recognition = True
    return 'stopping', 200


# teacher_register: GET -> show form, POST -> save teacher info and capture/upload image
@app.route('/teacher_register', methods=['GET', 'POST'])
def teacher_register():
    if request.method == 'GET':
        return render_template('teacher_register.html')

    name = request.form.get('name', '').strip()
    subject = request.form.get('subject', '').strip()

    if not name:
        return "Teacher name is required", 400

    csv_path = os.path.join(os.getcwd(), 'teachers.csv')
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, dtype=str).fillna('')
        else:
            df = pd.DataFrame(columns=['name', 'subject'])
        mask = (df['name'].astype(str).str.strip() == name)
        if mask.any():
            idx = df.index[mask][0]
            if subject:
                df.loc[idx, 'subject'] = subject
        else:
            new_row = {'name': name, 'subject': subject}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print("Error updating teachers.csv:", e)

    train_path = os.path.join(os.getcwd(), 'Training images')
    os.makedirs(train_path, exist_ok=True)

    if 'image' in request.files and request.files['image'].filename:
        f = request.files['image']
        base = f"T_{name}"
        i = 1
        while os.path.exists(os.path.join(train_path, f"{base}_{i}.png")):
            i += 1
        dst = os.path.join(train_path, f"{base}_{i}.png")
        try:
            f.save(dst)
            print("Teacher image uploaded:", dst)
            return render_template('image.html')
        except Exception as e:
            print("Failed to save uploaded teacher image:", e)
            return f"Failed to save image: {e}", 500

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        return "Cannot open camera on server. Upload an image instead.", 500

    window_title = "Press Space to capture teacher image (ESC to cancel)"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            cv2.imshow(window_title, frame)
            k = cv2.waitKey(1)
            if k % 256 == 27:  # ESC
                break
            elif k % 256 == 32:  # SPACE
                base = f"T_{name}"
                i = 1
                while os.path.exists(os.path.join(train_path, f"{base}_{i}.png")):
                    i += 1
                fname = f"{base}_{i}.png"
                cv2.imwrite(os.path.join(train_path, fname), frame)
                print("Teacher image saved:", fname)
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    return render_template('image.html')

@app.route('/upload_class_photo', methods=['POST'])
def upload_class_photo():
    """
    Robust multi-face detection + recognition with extra diagnostics.
    - Ensures correct RGB/BGR handling
    - Precise mapping from scaled detections back to full-res coords
    - Returns extra debug fields: detections_by_scale, candidate_boxes, encoding_count
    """
    try:
        import os, time, json, cv2, numpy as np
        from datetime import datetime
        from flask import request, jsonify
        from werkzeug.utils import secure_filename

        def normalize_name(n):
            if n is None:
                return None
            return str(n).strip().lower()

        # ---- validate file ----
        if 'photo' not in request.files:
            return jsonify({'success': False, 'error': 'no photo file uploaded'}), 400
        f = request.files['photo']
        if not f or f.filename == '':
            return jsonify({'success': False, 'error': 'no photo file provided'}), 400
        filename = secure_filename(f.filename)
        if not filename.lower().endswith(ALLOWED_EXT):
            return jsonify({'success': False, 'error': 'unsupported file type'}), 400

        os.makedirs(CLASS_PHOTOS_DIR, exist_ok=True)
        save_name = f"{int(time.time())}_{filename}"
        save_path = os.path.join(CLASS_PHOTOS_DIR, save_name)
        try:
            f.save(save_path)
        except Exception as e:
            print("Failed to save uploaded class photo:", e)
            return jsonify({'success': False, 'error': 'failed to save photo'}), 500

        # optional params
        try:
            max_faces = int(request.form.get('max_faces', 0))
            if max_faces < 0:
                max_faces = 0
        except Exception:
            max_faces = 0
        annotate = str(request.form.get('annotate', 'false')).lower() in ('1', 'true', 'yes', 'y')

        # ---- ensure encodings ----
        if not ensure_encodings_ready(wait_seconds=3.0):
            try:
                compute_encodings_background(os.path.join(os.getcwd(), 'Training images'))
            except Exception:
                pass
            with _encodings_lock:
                ready_len = len(_known_encodings_by_name)
            if ready_len == 0:
                return jsonify({'success': False, 'error': 'no training images / encodings found'}), 400

        with _encodings_lock:
            enc_by_name = dict(_known_encodings_by_name)

        name_list = list(enc_by_name.keys())
        if len(name_list) == 0:
            return jsonify({'success': False, 'error': 'no encodings loaded'}), 400

        # canonical names and master enc arrays
        def build_master(cnames, enc_map):
            master_enc_list = []
            master_name_idx = []
            for ni, nm in enumerate(cnames):
                encs = enc_map.get(nm) or []
                for e in encs:
                    try:
                        master_enc_list.append(np.array(e, dtype=np.float32))
                        master_name_idx.append(ni)
                    except Exception:
                        continue
            return master_enc_list, np.array(master_name_idx, dtype=np.int32) if master_name_idx else np.array([], dtype=np.int32)

        # canonical names normalized mapping
        name_norm_to_orig = {}
        canonical_names = []
        for orig in name_list:
            nrm = normalize_name(orig)
            if nrm not in name_norm_to_orig:
                name_norm_to_orig[nrm] = orig
                canonical_names.append(orig)

        master_enc_list, master_name_idx = build_master(canonical_names, enc_by_name)
        if len(master_enc_list) == 0:
            return jsonify({'success': False, 'error': 'no encoded faces in training images'}), 400
        master_encs = np.vstack(master_enc_list)
        name_to_indices = [np.nonzero(master_name_idx == ni)[0] for ni in range(len(canonical_names))]

        # ---- load image (BGR from disk), convert to RGB for face_recognition ----
        try:
            img_bgr = cv2.imread(save_path)
            if img_bgr is None:
                raise RuntimeError("cv2.imread returned None")
        except Exception as e:
            print("Failed to load uploaded image for recognition:", e)
            return jsonify({'success': False, 'error': 'failed to read uploaded image'}), 500

        # clamp large images
        MAX_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", 2400))
        h0, w0 = img_bgr.shape[:2]
        if max(h0, w0) > MAX_SIDE:
            scale_small = MAX_SIDE / float(max(h0, w0))
            img_bgr = cv2.resize(img_bgr, (0, 0), fx=scale_small, fy=scale_small, interpolation=cv2.INTER_AREA)

        # convert to RGB (face_recognition expects RGB)
        rgb_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H_full, W_full = rgb_full.shape[:2]

        # detection params
        scales_env = os.environ.get("DETECTION_SCALES", "0.25,0.5,0.75,1.0")
        try:
            scales = [float(s.strip()) for s in scales_env.split(',') if s.strip()]
        except Exception:
            scales = [0.25, 0.5, 1.0]
        if 1.0 not in scales:
            scales.append(1.0)
        # try larger scales first (fewer faces missed), but keep deterministic order
        scales = sorted(set([s for s in scales if 0.05 <= s <= 1.0]), reverse=True)

        use_cnn = os.environ.get("DETECTION_MODEL", "hog").lower() == "cnn"
        detect_model = 'cnn' if use_cnn else 'hog'

        base_threshold = float(os.environ.get("MATCH_THRESHOLD", MATCH_THRESHOLD))
        base_margin = float(os.environ.get("MATCH_MARGIN", MATCH_MARGIN))
        relax_delta = float(os.environ.get("RELAX_THRESHOLD_DELTA", 0.18))
        relaxed_slack = float(os.environ.get("RELAXED_SLACK", 0.08))

        # NMS + helpers
        def iou(a, b):
            xA = max(a[0], b[0]); yA = max(a[1], b[1])
            xB = min(a[2], b[2]); yB = min(a[3], b[3])
            interW = max(0, xB - xA); interH = max(0, yB - yA)
            inter = interW * interH
            areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
            areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
            union = areaA + areaB - inter
            return 0.0 if union == 0 else inter / union

        def nms_greedy(boxes, iou_thresh=0.18, max_keep=0):
            """
            Greedy NMS: sort by area (largest first) and skip boxes with IoU > iou_thresh.
            - iou_thresh: lower values keep more boxes (less merging)
            - max_keep: if >0, stop after keeping max_keep boxes
            """
            if not boxes:
                return []
            boxes_arr = np.array(boxes, dtype=np.int32)
            areas = (boxes_arr[:, 2] - boxes_arr[:, 0]) * (boxes_arr[:, 3] - boxes_arr[:, 1])
            order = np.argsort(-areas)
            kept = []
            while order.size > 0:
                i = int(order[0]); kept.append(boxes[i])
                if max_keep and len(kept) >= max_keep:
                    break
                if order.size == 1:
                    break
                rest = order[1:]
                new_rest = []
                for j in rest:
                    if iou(boxes[i], boxes[int(j)]) <= iou_thresh:
                        new_rest.append(int(j))
                order = np.array(new_rest, dtype=np.int64)
            return kept

        # get NMS IoU from env (allow testing different values without code change)
        try:
            env_iou = float(os.environ.get('DETECTION_NMS_IOU', 0.18))
        except Exception:
            env_iou = 0.18

        # ---- multi-scale detection (collect diagnostics) ----
        candidate_boxes = []
        detections_by_scale = {}
        detect_start = time.time()
        max_detect_seconds = float(os.environ.get('MAX_DETECT_SECONDS', 10.0))
        for s in scales:
            if time.time() - detect_start > max_detect_seconds:
                print("detection timed out after", max_detect_seconds, "s")
                break
            if s <= 0 or s > 1.0:
                continue
            # resize copy for this scale (if s==1.0 we use full)
            if s == 1.0:
                img_small = rgb_full
            else:
                img_small = cv2.resize(rgb_full, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
            try:
                locs = face_recognition.face_locations(img_small, model=detect_model)
            except Exception as e:
                print(f"face_locations error at scale {s}: {e}")
                locs = []

            detections_by_scale[s] = len(locs)
            # map locs back to full-res coords precisely
            scale_back = 1.0 / s
            for (top, right, bottom, left) in locs:
                l_full = int(round(left * scale_back)); t_full = int(round(top * scale_back))
                r_full = int(round(right * scale_back)); b_full = int(round(bottom * scale_back))
                # clamp
                l_full = max(0, min(W_full - 1, l_full))
                r_full = max(0, min(W_full - 1, r_full))
                t_full = max(0, min(H_full - 1, t_full))
                b_full = max(0, min(H_full - 1, b_full))
                # ignore very small boxes
                if (r_full - l_full) < 24 or (b_full - t_full) < 24:
                    continue
                candidate_boxes.append([l_full, t_full, r_full, b_full])

        debug = {'detections_by_scale': detections_by_scale, 'candidate_boxes_before_nms': len(candidate_boxes)}

        # NMS and apply max_faces. Use the env_iou value (lower reduces merging)
        unique_boxes = nms_greedy(candidate_boxes, iou_thresh=env_iou, max_keep=max_faces if max_faces > 0 else 0)
        if len(unique_boxes) == 0:
            # fallback: try a single pass on full-size image
            try:
                locs = face_recognition.face_locations(rgb_full, model=detect_model)
                for (top, right, bottom, left) in locs:
                    unique_boxes.append([left, top, right, bottom])
                debug['fallback_fullsize_pass'] = len(locs)
            except Exception as e:
                print("Fallback full-size detection failed:", e)
                debug['fallback_fullsize_error'] = str(e)

        # final cap
        if max_faces and len(unique_boxes) > max_faces:
            unique_boxes = unique_boxes[:max_faces]

        detection_count = len(unique_boxes)
        debug['num_boxes_after_nms'] = detection_count
        debug['nms_iou_used'] = env_iou
        debug['candidate_boxes_sample'] = candidate_boxes[:30]  # small sample so JSON isn't huge

        # ---- prepare locations (top,right,bottom,left) for face_encodings ----
        locations_for_encoding = []
        boxes_for_report = []
        # reduce padding slightly to avoid overlap in dense photos
        PAD_PCT = float(os.environ.get('DETECTION_PAD_PCT', 0.08))  # previously 0.12
        for (l, t, r, b) in unique_boxes:
            pad_x = int((r - l) * PAD_PCT)
            pad_y = int((b - t) * PAD_PCT)
            l2 = max(0, l - pad_x); t2 = max(0, t - pad_y)
            r2 = min(W_full - 1, r + pad_x); b2 = min(H_full - 1, b + pad_y)
            # face_recognition expects (top, right, bottom, left)
            locations_for_encoding.append((t2, r2, b2, l2))
            boxes_for_report.append([l2, t2, r2, b2])

        if len(locations_for_encoding) == 0:
            return jsonify({'success': True, 'detection_count': 0, 'matched_count': 0,
                            'present': [], 'absent': sorted(list(get_registered_students_from_training_images())),
                            'matches': [], 'photoPath': os.path.relpath(save_path, os.getcwd()),
                            'debug': debug})

        # ---- compute encodings (this uses rgb_full and known_face_locations in (top,right,bottom,left) order) ----
        try:
            encodings = face_recognition.face_encodings(rgb_full, known_face_locations=locations_for_encoding)
        except Exception as e:
            print("face_encodings batch error:", e)
            encodings = []
        debug['encoding_count'] = len(encodings)

        matches = []
        matched_norm_names = set()
        faces_debug = []

        for idx, enc in enumerate(encodings):
            box = boxes_for_report[idx]
            try:
                probe = np.array(enc, dtype=np.float32)
            except Exception:
                faces_debug.append({'box_full': box, 'error': 'encoding_cast_failed'})
                matches.append({'studentId': None, 'distance': None, 'box': box, 'top_candidates': []})
                continue

            try:
                dif = master_encs - probe
                dists = np.linalg.norm(dif, axis=1)
            except Exception:
                try:
                    dists = face_recognition.face_distance(master_encs, probe)
                except Exception:
                    dists = np.full((master_encs.shape[0],), np.inf)

            per_name_min = np.full(len(canonical_names), np.inf, dtype=np.float32)
            for ni, inds in enumerate(name_to_indices):
                if inds.size == 0:
                    continue
                per_name_min[ni] = float(np.min(dists[inds]))

            if per_name_min.size == 1:
                best_idx = 0
                best_dist = float(per_name_min[0])
                second_dist = np.inf
            else:
                idx_sorted = np.argsort(per_name_min)
                best_idx = int(idx_sorted[0])
                best_dist = float(per_name_min[best_idx])
                second_dist = float(per_name_min[idx_sorted[1]]) if per_name_min.size > 1 else np.inf

            chosen = None
            reason = 'none'
            if best_dist < base_threshold and (second_dist - best_dist) >= base_margin:
                chosen = canonical_names[best_idx]; reason = 'strict'
            else:
                relaxed_thresh = base_threshold + relaxed_slack
                relaxed_margin = max(0.0, base_margin * 0.5)
                if best_dist < relaxed_thresh and (second_dist - best_dist) >= relaxed_margin:
                    chosen = canonical_names[best_idx]; reason = 'relaxed'
                elif len(canonical_names) == 1 and best_dist < (base_threshold + 0.15):
                    chosen = canonical_names[best_idx]; reason = 'single_name_loose'
                else:
                    reason = 'no_match'

            chosen_norm = normalize_name(chosen) if chosen else None
            if chosen_norm:
                matched_norm_names.add(chosen_norm)

            topk = min(6, len(canonical_names))
            if per_name_min.size > 0:
                sorted_idxs = np.argsort(per_name_min)[:topk]
                top_cands = [{'name': canonical_names[i], 'dist': float(per_name_min[i])} for i in sorted_idxs]
            else:
                top_cands = []

            matches.append({'studentId': chosen, 'distance': float(best_dist) if chosen else None, 'box': box, 'top_candidates': top_cands, 'reason': reason})
            faces_debug.append({'box_full': box, 'top_candidates': top_cands, 'chosen': {'name': chosen, 'dist': float(best_dist) if chosen else None, 'reason': reason}})

            print(f"[face #{idx}] best='{canonical_names[best_idx] if per_name_min.size>0 else 'N/A'}' best_dist={best_dist:.4f} second_dist={second_dist:.4f} chosen={chosen} reason={reason}")

        # ---- determine present / absent using normalized names ----
        registered_original = list(get_registered_students_from_training_images())
        registered_norm_to_orig = {normalize_name(x): x for x in registered_original}
        present_norm = set(n for n in matched_norm_names if n in registered_norm_to_orig)
        present = [registered_norm_to_orig[n] for n in present_norm]
        absent = sorted(list(set(registered_original) - set(present)))

        # write attendance
        for name in present:
            try:
                markAttendanceCSV(name)
            except Exception as e:
                print("markAttendanceCSV error:", e)
            try:
                attendance_queue.put_nowait((name, 'student'))
            except Exception:
                pass

        matched_count = len(present)

        # ---- save audit report atomically ----
        try:
            rpt = {'timestamp': datetime.utcnow().isoformat(), 'present': present, 'absent': absent, 'matches': matches, 'faces_debug': faces_debug}
            rpt_path = os.path.splitext(save_path)[0] + '.json'
            tmp_rpt = rpt_path + '.tmp'
            with open(tmp_rpt, 'w') as outf:
                json.dump(rpt, outf)
            try:
                os.replace(tmp_rpt, rpt_path)
            except Exception:
                os.rename(tmp_rpt, rpt_path)
        except Exception as e:
            print("Failed to write class photo report:", e)

        # ---- annotated image ----
        annotated_path = None
        if annotate:
            try:
                annotated = img_bgr.copy()
                for m in matches:
                    box = m.get('box')
                    if not box:
                        continue
                    l, t, r, b = box
                    cv2.rectangle(annotated, (l, t), (r, b), (0, 255, 0), 2)
                    label = m['studentId'] if m.get('studentId') else "Unknown"
                    if m.get('distance') is not None:
                        label = f"{label} ({m['distance']:.2f})"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = max(0.4, min(0.8, (r - l) / 150.0))
                    thickness = 1
                    ((tw, th), _) = cv2.getTextSize(label, font, font_scale, thickness)
                    tbx1 = l; tby1 = max(0, t - th - 6); tbx2 = l + tw + 6; tby2 = t
                    cv2.rectangle(annotated, (tbx1, tby1), (tbx2, tby2), (0, 255, 0), -1)
                    cv2.putText(annotated, label, (l + 3, t - 4), font, font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
                annotated_name = os.path.splitext(save_name)[0] + '_annotated.jpg'
                annotated_path = os.path.join(CLASS_PHOTOS_DIR, annotated_name)
                cv2.imwrite(annotated_path, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            except Exception as e:
                print("Failed to create/save annotated image:", e)
                annotated_path = None

        # final debug add-ons
        debug['candidate_boxes_after_nms'] = unique_boxes
        debug['encoding_count'] = debug.get('encoding_count', 0)

        resp = {'success': True, 'detection_count': detection_count, 'matched_count': matched_count,
                'present': present, 'absent': absent, 'matches': matches,
                'photoPath': os.path.relpath(save_path, os.getcwd()), 'debug': debug, 'faces_debug': faces_debug}
        if annotated_path:
            resp['annotatedPath'] = os.path.relpath(annotated_path, os.getcwd())

        return jsonify(resp)

    except Exception as e:
        # Ensure we always return JSON (so frontend won't try to parse HTML)
        import traceback
        tb = traceback.format_exc()
        print("upload_class_photo: unexpected error:", tb)
        # Avoid leaking huge trace to client in production — here we include it for debugging.
        return jsonify({'success': False, 'error': 'server_exception', 'message': str(e), 'trace': tb}), 500


@app.route('/teacher_attendance', methods=['GET'])
def teacher_attendance():
    today = date.today().isoformat()
    rows_out = []

    fname = 'attendance_teachers.csv'
    if os.path.exists(fname):
        try:
            with open(fname, 'r', errors='ignore') as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split(',')
                    if len(parts) >= 3 and parts[2] == today:
                        rows_out.append({
                            'NAME': parts[0],
                            'Time': parts[1],
                            'Date': parts[2]
                        })
        except Exception as e:
            print("teacher_attendance read error:", e)

    return render_template('teacher_attendance.html', rows=rows_out)


@app.route('/login', methods=['POST'])
def login():
    try:
        json_data = json.loads(request.data.decode())
    except Exception:
        return 'failed'
    username = json_data.get('username')
    password = json_data.get('password')
    try:
        df = pd.read_csv('cred.csv')
    except Exception:
        return 'failed'
    if len(df.loc[df['username'] == username]['password'].values) > 0:
        if df.loc[df['username'] == username]['password'].values[0] == password:
            session['username'] = username
            return 'success'
    return 'failed'


@app.route('/checklogin')
def checklogin():
    if 'username' in session:
        return session['username']
    return 'False'


@app.route('/how', methods=["GET", "POST"])
def how():
    return render_template('form.html')


@app.route('/data', methods=["GET", "POST"])
def data():
    if request.method == "POST":
        today = date.today().isoformat()
        rows = []
        fname = 'attendance.csv'
        if os.path.exists(fname):
            try:
                with open(fname, 'r', errors='ignore') as f:
                    for line in f.readlines()[1:]:
                        parts = line.strip().split(',')
                        if len(parts) >= 3 and parts[2] == today:
                            rows.append({'NAME': parts[0], 'Time': parts[1], 'Date': parts[2]})
            except Exception as e:
                print("data route read error:", e)
        return render_template('form2.html', rows=rows)
    else:
        return render_template('form1.html')


@app.route('/whole', methods=["GET", "POST"])
def whole():
    rows = []
    fname = 'attendance.csv'
    if os.path.exists(fname):
        try:
            with open(fname, 'r', errors='ignore') as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        rows.append({'NAME': parts[0], 'Time': parts[1], 'Date': parts[2]})
        except Exception as e:
            print("whole route read error:", e)
    return render_template('form3.html', rows=rows)


@app.route('/dashboard', methods=["GET", "POST"])
def dashboard():
    """
    Dashboard now shows absent students for a given date (default = today) and parent phone numbers.
    Accepts optional query param ?date=YYYY-MM-DD
    """
    date_str = request.args.get('date') or date.today().isoformat()
    try:
        # validate date
        date.fromisoformat(date_str)
    except Exception:
        date_str = date.today().isoformat()

    registered = get_registered_students_from_training_images()
    present = get_present_student_names_for_date(date.fromisoformat(date_str))
    absentees = sorted(list(registered - present))
    parent_map = load_parent_contacts_from_csv()

    rows = []
    for name in absentees:
        rows.append({
            'name': name,
            'parent_phone': parent_map.get(name, {}).get('parent_phone', '')
        })

    return render_template('dashboard.html', rows=rows, date=date_str)


def get_registered_students_from_training_images(path=None):
    if path is None:
        path = os.path.join(os.getcwd(), 'Training images')
    names = set()
    try:
        if not os.path.exists(path):
            return set()
        for fname in os.listdir(path):
            if not fname.lower().endswith(ALLOWED_EXT):
                continue
            base = os.path.splitext(fname)[0]
            parts = base.rsplit('_', 1)
            root_name = base
            if len(parts) == 2 and parts[1].isdigit():
                root_name = parts[0]
            if root_name.startswith('T_'):
                continue
            names.add(root_name)
    except Exception as e:
        print("get_registered_students_from_training_images error:", e)
    return names


def get_present_student_names_for_date(att_date):
    dstr = att_date if isinstance(att_date, str) else att_date.isoformat()
    present = set()
    try:
        fname = 'attendance.csv'
        if os.path.exists(fname):
            with open(fname, 'r', errors='ignore') as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split(',')
                    if len(parts) >= 3 and parts[2] == dstr:
                        present.add(parts[0])
    except Exception as e:
        print("get_present_student_names_for_date error:", e)
    return present


def load_parent_contacts_from_csv(csv_path='students.csv'):
    mapping = {}
    if not os.path.exists(csv_path):
        return mapping
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna('')
        for _, row in df.iterrows():
            n = str(row.get('name') or row.get('Name') or '').strip()
            if not n:
                continue
            mapping[n] = {
                'parent_phone': (row.get('parent_phone') or row.get('Parent_Phone') or row.get('phone') or '').strip()
            }
    except Exception as e:
        print("load_parent_contacts_from_csv error:", e)
    return mapping


def send_sms(phone_number: str, body: str):
    if not ENABLE_SMS:
        return
    if not TwilioClient:
        raise RuntimeError("Twilio library not installed. Install twilio or disable ENABLE_SMS.")
    if not (TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM):
        raise RuntimeError("Twilio config missing. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM.")
    client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
    client.messages.create(body=body, from_=TWILIO_FROM, to=phone_number)


def build_absent_message(name, att_date):
    date_str = att_date.strftime("%d %b %Y") if isinstance(att_date, date) else att_date
    body = f"Absent: {name} on {date_str}. Contact school for details."
    return body


def send_absent_notifications_for_date(att_date):
    if isinstance(att_date, str):
        try:
            att_date_obj = date.fromisoformat(att_date)
        except Exception:
            att_date_obj = date.today()
    else:
        att_date_obj = att_date

    registered = get_registered_students_from_training_images()
    present = get_present_student_names_for_date(att_date_obj)
    absentees = sorted(list(registered - present))
    parent_map = load_parent_contacts_from_csv()

    results = []
    for name in absentees:
        parent_info = parent_map.get(name, {})
        parent_phone = parent_info.get('parent_phone') if parent_info else ''
        sms_text = build_absent_message(name, att_date_obj)

        item = {"name": name, "sms_sent": False, "errors": []}
        if parent_phone:
            try:
                send_sms(parent_phone, sms_text)
                item["sms_sent"] = True
            except Exception as e:
                item["errors"].append(f"sms_error:{e}")
                print(f"Error sending sms to {parent_phone} for {name} ->", e)
        else:
            item["errors"].append("no_parent_phone")

        results.append(item)
    return {"date": att_date_obj.isoformat(), "absentees_count": len(absentees), "results": results}


@app.route('/notify_absentees', methods=['POST'])
def notify_absentees_route():
    data = request.get_json(silent=True) or {}
    date_str = data.get('date')
    if date_str:
        try:
            date.fromisoformat(date_str)
        except Exception:
            return jsonify({"success": False, "error": "invalid date format (expected YYYY-MM-DD)"}), 400
    else:
        date_str = date.today().isoformat()

    try:
        out = send_absent_notifications_for_date(date.fromisoformat(date_str))
        return jsonify({"success": True, "payload": out})
    except Exception as e:
        print("notify_absentees_route error:", e)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    print("Routes:", app.url_map)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
