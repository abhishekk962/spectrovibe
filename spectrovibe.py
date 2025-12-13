import cv2
import numpy as np
import collections
import os
from datetime import datetime
import sounddevice as sd
from scipy.signal import butter, lfilter, lfilter_zi

# Webcam capture instead of video file
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam")

# Get original frame size for upsampling display
ret_temp, temp_frame = cap.read()
if not ret_temp:
    raise SystemExit("Cannot read initial frame")
original_height, original_width = temp_frame.shape[:2]
cap.release()  # Restart cap to ensure consistency
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Downsampling factor for processing
scale_factor = 0.25
low_height = int(original_height * scale_factor)
low_width = int(original_width * scale_factor)
low_res_size = (low_width, low_height)

MIN_AREA = 1500
THRESH = 6
BLUR_KSIZE = (7, 7)
DILATE_KERNEL = np.ones((1, 1), np.uint8)
WAIT_MS = 5  # adjust to match video FPS or desired playback speed
HEATMAP_DEPTH = 50  # number of past frames to accumulate for heatmap

# Audio settings
CHANNELS = 1
RATE = 44100
CUTOFF = 200  # Hz for bass
current_rms = 0
MIN_DEPTH = 10
MAX_DEPTH = 100

# Filter setup
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

b, a = butter_lowpass(CUTOFF, RATE)
zi = lfilter_zi(b, a)

# Audio callback
def audio_callback(indata, frames, time, status):
    global zi, current_rms
    if status:
        print(status)
    samples = indata[:, 0]
    filtered, zi = lfilter(b, a, samples, zi=zi)
    rms = np.sqrt(np.mean(filtered**2))
    current_rms = rms * 65534  # Scale to 0-65534

# Colormap options for switching
COLORMAPS = ['HOT', 'JET', 'BONE', 'OCEAN', 'SUMMER', 'PINK']
current_colormap = 0

print("Controls: SPACE to cycle colormaps, ESC to quit")
print("Trackbar: Adjust 'Heatmap Depth' slider to control motion trail length")

frame1 = None
frame2 = None

# Read first two frames
ret1, frame1 = cap.read()
if not ret1:
    print("Cannot read from webcam.")
    cap.release()
    raise SystemExit(1)
frame1_low = cv2.resize(frame1, low_res_size, interpolation=cv2.INTER_AREA)

ret2, frame2 = cap.read()
if not ret2:
    print("Cannot read second frame from webcam.")
    cap.release()
    raise SystemExit(1)
frame2_low = cv2.resize(frame2, low_res_size, interpolation=cv2.INTER_AREA)

# Buffer for last HEATMAP_DEPTH dilated images to build heatmap
dilated_history = collections.deque(maxlen=500)  # large enough for dynamic depth

# Create window and trackbar
cv2.namedWindow("motion", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("motion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# cv2.createTrackbar('Heatmap Depth', 'motion', HEATMAP_DEPTH, 300, lambda x: None)

# Audio device selection
devices = sd.query_devices()
print("Available devices:")
for i, dev in enumerate(devices):
    print(f"{i}: {dev['name']}")
# device_index = int(input("Enter audio device index: "))
device_index = 2

# Start audio stream
stream = sd.InputStream(device=device_index, channels=CHANNELS, samplerate=RATE, callback=audio_callback)
stream.start()

# Load the logo
logo_path = "C:\\Data\\DigitalDoubles\\SpectroVibe\\logo.png"  # Ensure the logo file is in the current directory
if not os.path.exists(logo_path):
    raise SystemExit(f"Logo file '{logo_path}' not found in the current directory.")

logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
if logo is None:
    raise SystemExit("Failed to load the logo image.")

# Resize the logo to fit in the top-left corner
logo_height, logo_width = 100, 240  # Adjust dimensions as needed
logo = cv2.resize(logo, (logo_width, logo_height), interpolation=cv2.INTER_AREA)

# Ensure the logo has an alpha channel for transparency
if logo.shape[2] != 4:
    alpha_channel = np.ones((logo.shape[0], logo.shape[1]), dtype=logo.dtype) * 255
    logo = np.dstack((logo, alpha_channel))

# Function to overlay the logo on a frame
def overlay_logo(frame, logo):
    y1, y2 = 50, 50 + logo.shape[0]
    x1, x2 = 50, 50 + logo.shape[1]

    # Extract the alpha channel from the logo
    alpha_logo = logo[:, :, 3] / 255.0
    alpha_frame = 1.0 - alpha_logo

    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (
            alpha_logo * logo[:, :, c] + alpha_frame * frame[y1:y2, x1:x2, c]
        )

while True:
    diff = cv2.absdiff(frame1_low, frame2_low)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)
    _, thresh = cv2.threshold(blur, THRESH, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, DILATE_KERNEL, iterations=2)
    dilated = cv2.GaussianBlur(dilated, (5, 5), 0)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    display_low = frame1_low.copy()
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(display_low, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Add dilated to history for heatmap
    dilated_history.append(dilated.copy())

    # Build and overlay heatmap if we have enough frames
    # depth = max(10, cv2.getTrackbarPos('Heatmap Depth', 'motion'))
    depth = int(MIN_DEPTH + (current_rms / 65534) * (MAX_DEPTH - MIN_DEPTH))  # Scale RMS to MIN_DEPTH-MAX_DEPTH
    # if len(dilated_history) >= 200:
    accum = np.sum(list(dilated_history)[-depth:], axis=0)
    accum_norm = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colormap_attr = getattr(cv2, f'COLORMAP_{COLORMAPS[current_colormap]}')
    heatmap = cv2.applyColorMap(accum_norm, colormap_attr)
    # Stylize base layer: convert to dark black and white
    base_gray = cv2.cvtColor(display_low, cv2.COLOR_BGR2GRAY)
    base_bw = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    base_dark = (base_bw * 1).astype(np.uint8)  # darken to 30% brightness
    blended = cv2.addWeighted(base_dark, 0.6, heatmap, 0.4, 0)
    # cv2.putText(blended, f"Colormap: {COLORMAPS[current_colormap]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # cv2.putText(blended, f"Depth: {depth} (RMS controlled)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Upsample to full resolution for display
    blended_full = cv2.resize(blended, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

    # Overlay the logo on the blended frame
    # overlay_logo(blended_full, logo)

    # cv2.putText(blended_full, f"Colormap: {COLORMAPS[current_colormap]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # cv2.putText(blended_full, f"Depth: {depth} (RMS controlled)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("motion", blended_full)
    # During buffering, don't show the window yet

    frame1_low = frame2_low
    ret, frame2 = cap.read()
    if not ret:
        print("Cannot read frame from webcam.")
        break
    frame2_low = cv2.resize(frame2, low_res_size, interpolation=cv2.INTER_AREA)

    key = cv2.waitKey(WAIT_MS) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # Spacebar
        current_colormap = (current_colormap + 1) % len(COLORMAPS)
        print(f"Switched to colormap: {COLORMAPS[current_colormap]}")

cv2.destroyAllWindows()
cap.release()
stream.stop()
stream.close()
