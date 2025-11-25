# POA Camera Viewer

A high-performance Python viewer for Player One Astronomy (POA) cameras, built with `pyqtgraph` and `PyQt5`.

## Features

- **Real-time Display**: High-speed image rendering using `pyqtgraph`.
- **Camera Control**:
    - **Exposure**: Adjustable exposure time (ms).
    - **Gain**: Manual or **Auto Gain** control.
    - **Frame Rate Limit**: Cap the frame rate (0-2000 FPS) with real-time **FPS Estimate**.
    - **Binning**: Hardware/Software binning (1x, 2x, 3x, 4x).
    - **Image Format**: Support for RAW8, RAW16, RGB24, and MONO8.
- **Histogram**:
    - Real-time histogram for Mono and RGB channels.
    - **Filled curves** for better visualization.
    - **Statistics**: Live Mean and Standard Deviation for each channel.
    - **Offloaded Calculation**: Histogram processing runs in a background thread for smooth UI performance.
- **Mock Mode**: Automatically falls back to a simulated camera if no physical camera is detected or the library is missing.
    - Generates noise/patterns to test UI responsiveness.
    - Optimized for performance.
- **Performance**:
    - Multithreaded architecture (Image acquisition and Histogram calculation in background).
    - Efficient memory usage and image transposition.

## Requirements

- Python 3.8+
- `numpy`
- `pyqtgraph`
- `PyQt5`
- `opencv-python` (optional, for additional testing scripts)
- **Player One Camera SDK**: `libPlayerOneCamera.dylib` (macOS), `.dll` (Windows), or `.so` (Linux).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/poa_view.git
    cd poa_view
    ```

2.  **Install dependencies**:
    ```bash
    pip install .
    # OR
    pip install numpy pyqtgraph PyQt5 opencv-python
    ```

3.  **SDK Setup**:
    - Download the Player One Astronomy Camera SDK from the official website.
    - Place the shared library file (`libPlayerOneCamera.dylib`, `PlayerOneCamera.dll`, or `libPlayerOneCamera.so`) in the project root directory or ensure it is in your system's library path.

## Usage

Run the viewer:

```bash
python poa_viewer.py
```

### Controls

- **Start/Stop Capture**: Begin or end the video stream.
- **Exposure**: Adjust the slider or spinbox to change exposure time.
- **Gain**: Use the slider/spinbox for manual gain, or check "Auto" for automatic gain control.
- **Frame Rate Limit**: Set a maximum FPS to reduce CPU load.
- **Binning**: Select binning level (1-4) to trade resolution for sensitivity/speed.
- **Image Format**: Choose the desired output format from the camera.
- **Show Histogram**: Open the real-time histogram window.

## Troubleshooting

- **"Failed to load library"**: Ensure the correct SDK library file for your OS is present in the directory. The application will default to **Mock Mode** if the library is missing.
- **Performance Issues**:
    - Increase **Binning** (e.g., to 2x or 4x).
    - Set a **Frame Rate Limit** (e.g., 30 FPS).
    - Hide the **Histogram** window if not needed.

## License

[BSD 2-Clause License](LICENSE)
