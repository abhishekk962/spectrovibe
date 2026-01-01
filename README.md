# SpectroVibe

A real-time motion detection heatmap visualizer that responds to audio bass levels, using webcam input.

## Preview

https://github.com/user-attachments/assets/974d9e36-8dab-43d2-b210-24fa7ad956bb


## Features

- Real-time motion detection with colorful heatmap overlay
- Audio-reactive heatmap depth adjustment based on bass intensity
- Multiple colormap options (HOT, JET, BONE, OCEAN, SUMMER, PINK)
- Optional logo overlay
- Fullscreen display mode

## Requirements

- Python 3.x
- Webcam
- Audio input device (microphone)
- Dependencies: numpy, opencv-python, scipy, sounddevice

## Installation

1. Clone or download the repository
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. (Optional) Place a `logo.png` file in the same directory as `spectrovibe.py` for logo overlay

## Audio Setup (Windows)

To use music playback for visualization while still hearing the audio through your speakers:

1. Download and install VB-Audio Virtual Cable from https://vb-audio.com/Cable/
2. Set "CABLE Input (VB-Audio Virtual Cable)" as your **default playback device** in Windows sound settings.
3. Open Sound settings again > More sound settings > Recording tab
4. Select "CABLE Output (VB-Audio Virtual Cable)" and click **Properties**
5. Go to the **Listen** tab
6. Check "Listen to this device"
7. In the dropdown, select your preferred playback device (speakers or headphones)
8. Click **Apply** and **OK**

Now play your music through any media player - it will be routed through the virtual cable for SpectroVibe's audio capture, and you'll hear it through your selected speakers/headphones simultaneously.

This setup allows real-time audio visualization without interrupting your music playback.

## Usage

Run the application:
```
python spectrovibe.py
```

### Controls

- **SPACE**: Cycle through available colormaps
- **L**: Toggle logo visibility (if logo.png exists)
- **X**: Hide the help message
- **ESC**: Exit the application

A help message with these controls appears on-screen for 20 seconds at startup.

## Configuration

You can modify various settings at the top of `spectrovibe.py`:

- `AUDIO_DEVICE_INDEX`: Audio input device index (check with `python -c "import sounddevice as sd; print(sd.query_devices())"`)
- `FRAME_WIDTH` / `FRAME_HEIGHT`: Webcam resolution
- `MOTION_THRESHOLD`: Sensitivity for motion detection
- `BASS_CUTOFF_HZ`: Frequency cutoff for bass detection
- `MIN_DEPTH` / `MAX_DEPTH`: Heatmap accumulation range
- `LOGO_SIZE` / `LOGO_POSITION`: Logo dimensions and placement

## How It Works

1. Captures webcam video at reduced resolution for performance
2. Detects motion using frame differencing
3. Applies audio analysis to bass frequencies to control heatmap intensity
4. Overlays the heatmap on a grayscale base image
5. Displays in fullscreen mode with optional logo and help text

## Troubleshooting

- If the webcam doesn't work, check `cv2.VideoCapture(0)` index
- For audio issues, verify `AUDIO_DEVICE_INDEX` (default 2)
- Ensure all dependencies are installed
- Logo won't appear if `logo.png` is missing or invalid
