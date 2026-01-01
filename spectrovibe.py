import cv2
import numpy as np
import collections
import os
import sounddevice as sd
from scipy.signal import butter, lfilter, lfilter_zi
import time


# Display settings
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
SCALE_FACTOR = 0.25  # Process at 25% resolution for performance
WAIT_MS = 5  # Frame delay in milliseconds
LOAD_LOGO = True  # Set to False to disable logo overlay

# Motion detection parameters
MOTION_THRESHOLD = 6  # Sensitivity for detecting motion
BLUR_KERNEL = (7, 7)  # Gaussian blur kernel for noise reduction
DILATE_KERNEL = np.ones((1, 1), np.uint8)  # Kernel for motion dilation

# Audio processing settings
AUDIO_CHANNELS = 1
SAMPLE_RATE = 44100
BASS_CUTOFF_HZ = 200  # Low-pass filter cutoff for bass detection
AUDIO_DEVICE_INDEX = 2  # Change this to match your audio input device

# Heatmap depth range (controlled by audio RMS)
MIN_DEPTH = 10  # Minimum frames to accumulate
MAX_DEPTH = 100  # Maximum frames to accumulate

# Available colormaps for visualization
COLORMAPS = ['HOT', 'JET', 'BONE', 'OCEAN', 'SUMMER', 'PINK']

# Logo overlay settings
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(SCRIPT_DIR, "logo.png")
LOGO_SIZE = (987//4, 277//4)  # width, height
LOGO_POSITION = (50, 50)  # x, y offset from top-left

current_rms = 0  # Global RMS value updated by audio callback


def create_lowpass_filter(cutoff, sample_rate, order=5):
    """Create Butterworth low-pass filter coefficients for bass isolation."""
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    return b, a


# Initialize audio filter
filter_b, filter_a = create_lowpass_filter(BASS_CUTOFF_HZ, SAMPLE_RATE)
filter_state = lfilter_zi(filter_b, filter_a)


def audio_callback(indata, frames, time, status):
    """
    Process incoming audio to extract bass RMS value.
    This callback runs in a separate thread for real-time audio analysis.
    """
    global filter_state, current_rms
    if status:
        print(f"Audio status: {status}")
    
    samples = indata[:, 0]
    filtered, filter_state = lfilter(filter_b, filter_a, samples, zi=filter_state)
    rms = np.sqrt(np.mean(filtered ** 2))
    current_rms = rms * 65534  # Scale to usable range


def initialize_webcam():
    """Set up webcam capture with HD resolution."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")
    return cap


# Initialize capture and get frame dimensions
cap = initialize_webcam()
ret_temp, temp_frame = cap.read()
if not ret_temp:
    raise SystemExit("Cannot read initial frame")

original_height, original_width = temp_frame.shape[:2]
low_width = int(original_width * SCALE_FACTOR)
low_height = int(original_height * SCALE_FACTOR)
low_res_size = (low_width, low_height)

# Restart capture for clean state
cap.release()
cap = initialize_webcam()

def load_logo(path, size):
    """Load and prepare logo image with alpha channel for overlay."""
    if not os.path.exists(path):
        raise SystemExit(f"Logo file not found: {path}")
    
    logo = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        raise SystemExit("Failed to load logo image")
    
    logo = cv2.resize(logo, size, interpolation=cv2.INTER_AREA)
    
    # Ensure alpha channel exists
    if logo.shape[2] != 4:
        alpha = np.ones((logo.shape[0], logo.shape[1]), dtype=logo.dtype) * 255
        logo = np.dstack((logo, alpha))
    
    return logo


def overlay_logo(frame, logo, position):
    """Blend logo onto frame using alpha compositing."""
    x, y = position
    h, w = logo.shape[:2]
    
    alpha = logo[:, :, 3] / 255.0
    inverse_alpha = 1.0 - alpha
    
    for channel in range(3):
        frame[y:y+h, x:x+w, channel] = (
            alpha * logo[:, :, channel] + 
            inverse_alpha * frame[y:y+h, x:x+w, channel]
        )


logo = load_logo(LOGO_PATH, LOGO_SIZE) if os.path.exists(LOGO_PATH) else None
logo_visible = True
show_message = True
start_time = time.time()

print("SpectroVibe Controls:")
print("  SPACE - Cycle through colormaps")
print("  ESC   - Exit")

# Read initial frames for motion differencing
ret1, frame1 = cap.read()
if not ret1:
    cap.release()
    raise SystemExit("Cannot read from webcam")
frame1_low = cv2.resize(frame1, low_res_size, interpolation=cv2.INTER_AREA)

ret2, frame2 = cap.read()
if not ret2:
    cap.release()
    raise SystemExit("Cannot read second frame from webcam")
frame2_low = cv2.resize(frame2, low_res_size, interpolation=cv2.INTER_AREA)

# Motion history buffer for heatmap accumulation
dilated_history = collections.deque(maxlen=500)
current_colormap = 0

# Create fullscreen display window
cv2.namedWindow("motion", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("motion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Start audio input stream
stream = sd.InputStream(
    device=AUDIO_DEVICE_INDEX,
    channels=AUDIO_CHANNELS,
    samplerate=SAMPLE_RATE,
    callback=audio_callback
)
stream.start()

try:
    while True:
        # Detect motion using frame differencing
        diff = cv2.absdiff(frame1_low, frame2_low)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
        _, thresh = cv2.threshold(blur, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, DILATE_KERNEL, iterations=2)
        dilated = cv2.GaussianBlur(dilated, (5, 5), 0)
        
        # Store motion mask in history
        dilated_history.append(dilated.copy())
        
        # Calculate heatmap depth based on audio bass intensity
        depth = int(MIN_DEPTH + (current_rms / 65534) * (MAX_DEPTH - MIN_DEPTH))
        
        # Accumulate recent motion frames into heatmap
        accum = np.sum(list(dilated_history)[-depth:], axis=0)
        accum_norm = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply colormap to motion heatmap
        colormap_id = getattr(cv2, f'COLORMAP_{COLORMAPS[current_colormap]}')
        heatmap = cv2.applyColorMap(accum_norm, colormap_id)
        
        # Create grayscale base layer from current frame
        base_gray = cv2.cvtColor(frame1_low, cv2.COLOR_BGR2GRAY)
        base_bw = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
        
        # Blend base layer with heatmap overlay
        blended = cv2.addWeighted(base_bw, 0.6, heatmap, 0.4, 0)
        
        # Upscale to full resolution for display
        blended_full = cv2.resize(blended, (original_width, original_height), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Overlay logo if enabled
        if logo_visible and logo is not None:
            overlay_logo(blended_full, logo, LOGO_POSITION)
        
        # Overlay message if enabled and within timeout
        if show_message and (time.time() - start_time) < 20:
            cv2.putText(blended_full, "press SPACE to cycle through colors, press L to hide the logo, press X to hide this message, press ESC to close the app", 
                        (50, original_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("motion", blended_full)
        
        # Advance frame buffer
        frame1_low = frame2_low
        ret, frame2 = cap.read()
        if not ret:
            print("Lost webcam connection")
            break
        frame2_low = cv2.resize(frame2, low_res_size, interpolation=cv2.INTER_AREA)
        
        # Handle keyboard input
        key = cv2.waitKey(WAIT_MS) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == ord(' '):  # Space to cycle colormap
            current_colormap = (current_colormap + 1) % len(COLORMAPS)
            print(f"Colormap: {COLORMAPS[current_colormap]}")
        elif key == ord('l'):  # L to toggle logo visibility
            logo_visible = not logo_visible
            print(f"Logo visible: {logo_visible}")
        elif key == ord('x'):  # X to hide message
            show_message = False

finally:
    # Clean up resources
    cv2.destroyAllWindows()
    cap.release()
    stream.stop()
    stream.close()
