# -*- coding: utf-8 -*-
# Import required libraries
import cv2
import torch
import numpy as np
import NDIlib as ndi
import time
import os
from ultralytics import YOLO
from pythonosc import udp_client

print("--- Script Started Running (Dual C270 + ultralytics + NDI + OSC) ---")

# --- Create debug frames folder ---
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = "C:\\YOLO_Stream_Project"
if not os.path.exists(project_root_dir):
    try: os.makedirs(project_root_dir); print(f"Created project root directory: {project_root_dir}")
    except OSError as e: print(f"!!! Failed to create project root directory {project_root_dir}: {e}"); project_root_dir = base_dir
output_folder = os.path.join(project_root_dir, "debug_frames")
if not os.path.exists(output_folder):
    try: os.makedirs(output_folder); print(f"Created folder: {output_folder}")
    except OSError as e: print(f"!!! Failed to create folder {output_folder}: {e}"); output_folder = project_root_dir
else: print(f"Debug frames will be saved to: {output_folder}")

# --- NDI Initialization ---
print("Initializing NDI...")
ndi_initialized = False
if ndi.initialize(): print("NDI initialized successfully."); ndi_initialized = True
else: print("!!! Unable to initialize NDI, please make sure NDI Runtime is installed!")

# --- OSC Initialization ---
osc_ip = "127.0.0.1"
osc_port_count_cam1 = 9001
osc_port_contour_cam1 = 9002
osc_client_count_cam1 = None
osc_client_contour_cam1 = None
try:
    osc_client_count_cam1 = udp_client.SimpleUDPClient(osc_ip, osc_port_count_cam1)
    print(f"OSC Client (Count Cam1) will send to {osc_ip}:{osc_port_count_cam1}")
    osc_client_contour_cam1 = udp_client.SimpleUDPClient(osc_ip, osc_port_contour_cam1)
    print(f"OSC Client (Contour Cam1) will send to {osc_ip}:{osc_port_contour_cam1}")
except Exception as e_osc_init:
    print(f"!!! Failed to initialize OSC Clients: {e_osc_init}")

# --- 1. Load YOLOv8 Segmentation Model ---
print("Loading YOLOv8 segmentation model...")
try:
    model_name = 'yolov8n-seg.pt'
    model = YOLO(model_name)
    print(f"Model '{model_name}' loaded successfully!")
except Exception as e_load:
    print(f"!!! Failed to load model: {e_load}");
    if ndi_initialized: ndi.destroy()
    exit()

# --- Model Parameters ---
conf_threshold = 0.25 # You can adjust this value
person_class_id = 0
model.conf = conf_threshold
model.classes = [person_class_id]
model.iou = 0.45
print(f"Model parameters set: confidence > {model.conf}, IoU < {model.iou}, only detect class {person_class_id} (person)")

# --- 2. Initialize Cameras ---
print("Initializing two USB cameras...")
cap1, cap2 = None, None
width1, height1, fps1 = 640, 480, 30
width2, height2, fps2 = 640, 480, 30
cam_id1, cam_id2 = -1, -1
cap2_fail_count = 0  # Initialize camera 2 failure counter
camera_indices_to_try = [6, 7]

# Add more debug information in initialize_camera function
def initialize_camera(camera_label, exclude_id=-1, target_width=640, target_height=480, specific_id=None):
    print(f"---- Initializing {camera_label} ----")
    # If a specific ID is specified, only try that ID
    cam_ids_to_try = [specific_id] if specific_id is not None else camera_indices_to_try
    
    print(f"DEBUG: Will try the following camera IDs: {cam_ids_to_try}")
    
    for cam_id in cam_ids_to_try:
        if cam_id == exclude_id: continue
        print(f"  Trying to open camera ID: {cam_id} (using DSHOW backend)...")
        temp_cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        time.sleep(2)
        print(f"DEBUG: Camera ID {cam_id} isOpened(): {temp_cap.isOpened()}")
        if temp_cap.isOpened():
            print(f"    ID {cam_id}: isOpened() True. Trying to set resolution {target_width}x{target_height}...")
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
            time.sleep(1)
            ret_test, test_frame = temp_cap.read()
            actual_w = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"    ID {cam_id}: Read test frame: {ret_test}, actual resolution: {actual_w}x{actual_h}")
            if ret_test and test_frame is not None:
                if abs(actual_w - target_width) < 50 and abs(actual_h - target_height) < 50 :
                    print(f"  Camera ID {cam_id} ({camera_label}) is readable and successfully set to approximately {target_width}x{target_height}!")
                    # Removed code for saving test frame images
                    # temp_cap.fail_count = 0 # <--- Remove this line
                    return temp_cap, cam_id, actual_w, actual_h
                else:
                    print(f"  ID {cam_id} ({camera_label}): Resolution {actual_w}x{actual_h}, but readable. Continue searching for preferred...")
                    temp_cap.release()
            else:
                print(f"  ID {cam_id} ({camera_label}): isOpened() is True but cannot read valid frame.")
                temp_cap.release()
        else:
            if temp_cap is not None: temp_cap.release()
    print(f"---- {camera_label} initialization attempts ended ----")
    return None, -1, target_width, target_height

print("Initializing camera 1 (for contour and counting)...")
# Camera 1 tries ID 5
cap1, cam_id1, width1, height1 = initialize_camera("Camera 1", specific_id=5)
if cap1:
    _fps1 = cap1.get(cv2.CAP_PROP_FPS); fps1 = _fps1 if 0 < _fps1 <= 60 else 30
    print(f"Camera 1 (ID {cam_id1}) initialized successfully! Resolution: {width1}x{height1}, Frame rate: {fps1:.2f} fps")
else: print("!!! Failed to initialize camera 1")

print("Initializing camera 2 (for top real-time video)...")
# Camera 2 forced to use ID 6
cap2, cam_id2, width2, height2 = initialize_camera("Camera 2", exclude_id=cam_id1, specific_id=6)
if cap2:
    _fps2 = cap2.get(cv2.CAP_PROP_FPS); fps2 = _fps2 if 0 < _fps2 <= 60 else 30
    print(f"Camera 2 (ID {cam_id2}) initialized successfully! Resolution: {width2}x{height2}, Frame rate: {fps2:.2f} fps")
else: print("!!! Failed to initialize camera 2")

if cap1 is None:
    print("!!! Camera 1 could not be initialized, contour-related functions will not be available.")
    if cap2 is None and ndi_initialized: ndi.destroy(); exit()
elif cap2 is None:
    print("!!! Camera 2 could not be initialized, top real-time video NDI will not be available.")

# --- 3. Create NDI Senders ---
ndi_sender_cam1_sil, ndi_sender_cam2_orig = None, None
video_frame_cam1_sil, video_frame_cam2_orig = None, None

if ndi_initialized:
    if cap1:
        sender_name_cam1_sil = 'PythonYOLO_NDI_Cam1_Silhouette'
        send_settings1 = ndi.SendCreate(); send_settings1.ndi_name = sender_name_cam1_sil
        ndi_sender_cam1_sil = ndi.send_create(send_settings1)
        if ndi_sender_cam1_sil: print(f"NDI Sender '{sender_name_cam1_sil}' created successfully."); video_frame_cam1_sil = ndi.VideoFrameV2()
        else: print(f"!!! Unable to create NDI Sender '{sender_name_cam1_sil}'!")

    if cap2:
        sender_name_cam2_orig = 'PythonYOLO_NDI_Cam2_Original'
        send_settings2 = ndi.SendCreate(); send_settings2.ndi_name = sender_name_cam2_orig
        print(f"DEBUG: Trying to create camera 2 NDI sender '{sender_name_cam2_orig}'")
        ndi_sender_cam2_orig = ndi.send_create(send_settings2)
        print(f"DEBUG: Camera 2 NDI sender creation result: {ndi_sender_cam2_orig}")
        if ndi_sender_cam2_orig: 
            print(f"NDI Sender '{sender_name_cam2_orig}' created successfully.")
            video_frame_cam2_orig = ndi.VideoFrameV2()
            print(f"DEBUG: Camera 2 VideoFrameV2 creation result: {video_frame_cam2_orig}")
        else: 
            print(f"!!! Unable to create NDI Sender '{sender_name_cam2_orig}'!")

# --- 5. Main Loop ---
# Add the following code before starting the main loop
print("--- Start processing video streams (press 'q' to exit) ---")
frame_count = 0
debug_mode = False

while True:
    # Add frame rate control variables
    target_fps = 15.0  # Target frame rate
    frame_time = 1.0 / target_fps  # Ideal time interval per frame
    last_frame_time = time.time()  # Timestamp of the last frame

    # Add frame rate monitoring variables
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display_interval = 5.0  # Display frame rate every 5 seconds

    while True:
        # Frame rate control - calculate wait time
        current_time = time.time()
        elapsed = current_time - last_frame_time
        
        sleep_time = max(0, frame_time - elapsed)
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Update timestamp
        last_frame_time = time.time()
        
        # --- Process Camera 1 ---
        if cap1 and cap1.isOpened():
            ret1, frame1 = cap1.read()
            if ret1 and frame1 is not None:
                silhouette_frame1 = np.zeros_like(frame1)
                current_person_count_cam1 = 0
                detected_person_indices_cam1 = []
                
                # Perform YOLO inference every 3 frames to reduce CPU load
                if frame_count % 3 == 0:  # This value can be adjusted
                    results1 = model(frame1)
                    if len(results1) > 0:
                        result1 = results1[0]
                        boxes1 = result1.boxes
                        masks1 = result1.masks

                if boxes1 is not None and hasattr(boxes1, 'cls') and hasattr(boxes1, 'conf'):
                    for i in range(len(boxes1.cls)):
                        if int(boxes1.cls[i]) == person_class_id and boxes1.conf[i].item() >= conf_threshold:
                            detected_person_indices_cam1.append(i)
                    current_person_count_cam1 = len(detected_person_indices_cam1)

                # Find the index of the person with the highest confidence
                highest_conf_person_idx = -1
                highest_conf = 0
                
                if current_person_count_cam1 > 0:
                    for i in detected_person_indices_cam1:
                        if boxes1.conf[i].item() > highest_conf:
                            highest_conf = boxes1.conf[i].item()
                            highest_conf_person_idx = i
                    
                    if frame_count % 60 == 0:
                        print(f"Cam1: Selected person with highest confidence (index:{highest_conf_person_idx}, confidence:{highest_conf:.2f})")

                if current_person_count_cam1 > 0 and masks1 is not None and highest_conf_person_idx >= 0:
                    processed_indices_for_osc_cam1 = set()
                    try:
                        if hasattr(masks1, 'data') and masks1.data is not None:
                            masks_tensor1 = masks1.data.cpu()
                            if masks_tensor1.shape[0] == len(boxes1.cls):
                                masks_data1 = masks_tensor1.numpy().astype(np.uint8)
                                # Only process the contour of the person with the highest confidence
                                i = highest_conf_person_idx
                                mask1 = masks_data1[i]
                                if mask1.shape == (height1, width1):
                                    silhouette_frame1[mask1 > 0] = [255, 255, 255]
                                else:
                                    try:
                                        mask_resized1 = cv2.resize(mask1, (width1, height1), interpolation=cv2.INTER_NEAREST)
                                        silhouette_frame1[mask_resized1 > 0] = [255, 255, 255]
                                    except: pass
                                processed_indices_for_osc_cam1.add(i)
                        
                        if hasattr(masks1, 'xy') and masks1.xy is not None and len(masks1.xy) > 0:
                             if len(masks1.xy) == len(boxes1.cls):
                                # Only process the contour of the person with the highest confidence
                                i = highest_conf_person_idx
                                if i not in processed_indices_for_osc_cam1: 
                                    mask_points1 = masks1.xy[i]
                                    points1 = np.array(mask_points1, dtype=np.int32)
                                    cv2.fillPoly(silhouette_frame1, [points1], (255, 255, 255))
                                    if osc_client_contour_cam1 and mask_points1 is not None and len(mask_points1) > 2:
                                        try:
                                            flat_normalized_points1 = []
                                            for point in mask_points1:
                                                px = float(point[0]) / width1
                                                py = float(point[1]) / height1
                                                flat_normalized_points1.append(px)
                                                flat_normalized_points1.append(py)
                                            osc_client_contour_cam1.send_message(f"/cam1/people/new_contour_normalized", flat_normalized_points1)
                                        except Exception as e_osc_send_contour:
                                            if frame_count < 10: print(f"!!! OSC Contour Send Error Cam1: {e_osc_send_contour}")
                    except Exception as e_masks:
                        if frame_count < 10: print(f"!!! Error processing masks Cam1: {e_masks}")
            
            if osc_client_count_cam1:
                try: 
                    osc_client_count_cam1.send_message(f"/cam1/people/current_count", current_person_count_cam1)
                    if frame_count % 60 == 0 : print(f"Cam1: Detected {current_person_count_cam1} person(s). OSC sent.")
                except Exception as e_osc_count:
                    if frame_count < 10: print(f"!!! OSC Count Send Error Cam1: {e_osc_count}")
            
            if debug_mode and silhouette_frame1 is not None: cv2.imshow('Cam 1 Silhouette', silhouette_frame1)

            if ndi_sender_cam1_sil and video_frame_cam1_sil and silhouette_frame1 is not None:
                 try:
                     frame_bgra1 = cv2.cvtColor(silhouette_frame1, cv2.COLOR_BGR2BGRA)
                     frame_bgra_cont1 = np.ascontiguousarray(frame_bgra1)
                     video_frame_cam1_sil.data = frame_bgra_cont1; video_frame_cam1_sil.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
                     video_frame_cam1_sil.xres = width1; video_frame_cam1_sil.yres = height1
                     video_frame_cam1_sil.line_stride_in_bytes = width1 * 4
                     ndi.send_send_video_v2(ndi_sender_cam1_sil, video_frame_cam1_sil)
                 except Exception as e_ndi1: print(f"!!! NDI Send Cam 1 Error: {e_ndi1}")
        elif frame_count < 10 or frame_count % 60 == 0 : 
            print("[ INFO ] Cam 1: No detections or unable to read frame.")
            if ndi_sender_cam1_sil and video_frame_cam1_sil and silhouette_frame1 is not None: # Send black contour frame even if no detection
                 try:
                     frame_bgra1 = cv2.cvtColor(silhouette_frame1, cv2.COLOR_BGR2BGRA) 
                     frame_bgra_cont1 = np.ascontiguousarray(frame_bgra1)
                     video_frame_cam1_sil.data = frame_bgra_cont1; video_frame_cam1_sil.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
                     video_frame_cam1_sil.xres = width1; video_frame_cam1_sil.yres = height1
                     video_frame_cam1_sil.line_stride_in_bytes = width1 * 4
                     ndi.send_send_video_v2(ndi_sender_cam1_sil, video_frame_cam1_sil)
                 except: pass # Ignore errors when sending black frames


        # Check if camera 2 exists and is open
        if cap2 is not None and hasattr(cap2, 'isOpened') and cap2.isOpened():
            print(f"DEBUG: Frame {frame_count} - Camera 2 status: isOpened={cap2.isOpened()}")
            ret2, frame2 = cap2.read()
            print(f"DEBUG: Frame {frame_count} - Camera 2 read result: ret2={ret2}, frame2 is None: {frame2 is None}")
            if ret2 and frame2 is not None:
                if debug_mode: cv2.imshow('Cam 2 Original', frame2)
                # Removed code for saving debug frames
                        
                if ndi_sender_cam2_orig and video_frame_cam2_orig:
                     print(f"DEBUG: Frame {frame_count} - Trying to send Camera 2 NDI")
                     try:
                         frame_bgra2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2BGRA)
                         frame_bgra_cont2 = np.ascontiguousarray(frame_bgra2)
                         video_frame_cam2_orig.data = frame_bgra_cont2; video_frame_cam2_orig.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
                         video_frame_cam2_orig.xres = width2; video_frame_cam2_orig.yres = height2
                         video_frame_cam2_orig.line_stride_in_bytes = width2 * 4
                         ndi.send_send_video_v2(ndi_sender_cam2_orig, video_frame_cam2_orig)
                         print(f"DEBUG: Frame {frame_count} - Camera 2 NDI sent successfully")
                     except Exception as e_ndi2: 
                         print(f"!!! NDI Send Cam 2 Error: {e_ndi2}")
                         print(f"DEBUG: NDI error details: {type(e_ndi2).__name__}, {str(e_ndi2)}")
                else:
                    print(f"DEBUG: Frame {frame_count} - Camera 2 NDI sender not initialized: ndi_sender_cam2_orig={ndi_sender_cam2_orig}, video_frame_cam2_orig={video_frame_cam2_orig}")
            else:
                # Camera 2 read failure count
                # if not hasattr(cap2, 'fail_count'): # <--- Remove this check
                #     cap2.fail_count = 0
                cap2_fail_count += 1 # Use independent counter
                
                if frame_count % 60 == 0 or cap2_fail_count >= 10: 
                    print(f"[ WARN ] Cam 2: Unable to read current frame, skipped processing... (consecutive failures: {cap2_fail_count})")
                
                # If consecutive failures exceed 30, try to reinitialize camera 2
                if cap2_fail_count >= 30:
                    print("Trying to reinitialize camera 2...")
                    if cap2 and cap2.isOpened(): # Ensure cap2 is not None
                        cap2.release()
                    time.sleep(2)  # Wait for camera to release resources
                    cap2, cam_id2, width2, height2 = initialize_camera("Camera 2", exclude_id=cam_id1, specific_id=6)
                    if cap2:
                        _fps2 = cap2.get(cv2.CAP_PROP_FPS); fps2 = _fps2 if 0 < _fps2 <= 60 else 30
                        print(f"Camera 2 (ID {cam_id2}) reinitialized successfully! Resolution: {width2}x{height2}, Frame rate: {fps2:.2f} fps")
                        # Reset counter even if reinitialization fails, to avoid immediate retry
                        cap2_fail_count = 0 

    frame_count += 1
    key = cv2.waitKey(1)  # Fix indentation here - align with frame_count
    if key != -1 and key & 0xFF == ord('q'):
        print("Received exit signal 'q'...")
        break
    elif key != -1 and key & 0xFF == ord('d'):
        debug_mode = not debug_mode
        print(f"Debug mode: {'enabled' if debug_mode else 'disabled'}")

# --- Code outside the main loop ---
print("Loop ended, manually releasing resources...")
# --- NDI Cleanup ---
if 'ndi_sender_cam1_sil' in locals() and ndi_sender_cam1_sil is not None:
    try:
        ndi.send_destroy(ndi_sender_cam1_sil)
        print("NDI Sender 1 released.")
    except Exception as e:
        print(f"Error destroying NDI Sender 1: {e}")
if 'ndi_sender_cam2_orig' in locals() and ndi_sender_cam2_orig is not None:
    try:
        ndi.send_destroy(ndi_sender_cam2_orig)
        print("NDI Sender 2 released.")
    except Exception as e:
        print(f"Error destroying NDI Sender 2: {e}")
if 'ndi' in locals() and 'ndi_initialized' in locals() and ndi_initialized:
    try:
        ndi.destroy()
        print("NDI system destroyed.")
    except Exception as e:
        print(f"Error destroying NDI system: {e}")
# -----------------
if 'cap1' in locals() and cap1 is not None and cap1.isOpened():
    cap1.release()
    print("Camera 1 released.")
if 'cap2' in locals() and cap2 is not None and cap2.isOpened():
    cap2.release()
    print("Camera 2 released.")
# --- Close debug windows ---
if debug_mode:
    try:
        cv2.destroyAllWindows()
        print("Debug windows closed.")
    except Exception as e:
        print(f"Error closing debug windows: {e}")
print("--- Script ended normally ---")

