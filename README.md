# Video DeepDream CLI Tool

A command-line tool to apply Google's DeepDream effect to video files (specifically `.mov` format), running directly on the host using `uv` to maximize performance.

This tool extracts frames from the input video, applies the DeepDream algorithm utilizing a pre-trained `InceptionV3` convolutional neural network, visualizes progress, and outputs the processed video while preserving the original audio track.

---

## Features

- **Multi-Scale Processing (Octaves)**: Renders DeepDream patterns at different resolutions to capture both fine details and abstract structures.
- **Tiled Gradient Ascent**: Splits frames into smaller tiles during backpropagation to prevent out-of-memory errors on high-resolution videos.
- **Random Rolls**: Shifts the image tile boundaries randomly at each step to prevent visible boundaries and seams.
- **Visual Progress Bar**: Displays real-time progress, including percentage processed, frame counts, frame-rate speed (fps), and estimated time remaining (ETA).
- **Audio Copying**: Automatically extracts and merges the audio track from the original video using `ffmpeg` (falls back gracefully to silent output with a warning if `ffmpeg` is not installed).
- **Time-stamped Outputs**: Saves processed videos in an `output/` folder named with the exact timestamp of when the generation began (e.g. `output/2026-06-20_13-09-24.mov`).
- **Configuration Presets**: Offers visual presets (`default`, `edges`, `detailed`, `trippy`) to easily generate different styles without tweaking parameters.

---

## Prerequisites & Installation

### 1. General Requirements
- **Python 3.9+**
- **`uv` Package Manager**:
  - macOS (via Homebrew): `brew install uv`
  - Windows (via PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - Or via curl (macOS/Linux): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **`ffmpeg`** (Highly recommended to copy original audio):
  - macOS (via Homebrew): `brew install ffmpeg`
  - Windows: Install via winget: `winget install Gyan.FFmpeg` (or download manually and add to PATH).

---

### 2. Setup by Operating System

Select your operating system to set up Python dependencies and GPU acceleration:

#### A. macOS (Apple Silicon GPU Acceleration)
To run the script leveraging the M1/M2/M3/M4 GPU:
1. Navigate to the project directory and create the virtual environment:
   ```bash
   uv venv
   ```
2. Install base dependencies and the Apple Metal plugin for TensorFlow:
   ```bash
   uv pip install tensorflow opencv-python numpy tqdm
   uv pip install tensorflow-metal
   ```

#### B. Windows (NVIDIA GPU Acceleration - WSL2 - Recommended)
Since TensorFlow 2.11+ only supports GPU on Windows via WSL2, this is the recommended path for NVIDIA RTX cards (like the RTX 4060):
1. Open PowerShell as Admin and install WSL2:
   ```powershell
   wsl --install
   ```
2. Open your WSL2 Linux terminal, navigate to the project directory, and run:
   ```bash
   uv venv
   uv pip install tensorflow[and-cuda] opencv-python numpy tqdm
   ```

#### C. Windows (NVIDIA GPU Acceleration - Native Windows - TF 2.10)
If you prefer not to use WSL2, you must use TensorFlow 2.10.0:
1. Install **CUDA Toolkit 11.2** and **cuDNN 8.1** from NVIDIA.
2. In your Windows Command Prompt or PowerShell, run:
   ```powershell
   uv venv
   .venv\Scripts\pip install tensorflow==2.10.0 opencv-python numpy tqdm
   ```

*(Note: The first time you run the script, TensorFlow will automatically download the required InceptionV3 weights from Google.)*

---

## Usage

Run the tool using the virtual environment's Python interpreter:

```bash
.venv/bin/python3 deepdream_video.py --input <path_to_video> [options]
```

### CLI Arguments Reference

| Argument | Shorthand | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | *Required* | Path to the input video file (e.g., `IMG_1736.mov`). |
| `--output-dir` | `-o` | `output` | Directory where output videos will be saved. |
| `--preset` | `-p` | `None` | Configuration preset (`default`, `edges`, `detailed`, `trippy`). Overrides individual parameters. |
| `--layers` | | `mixed3,mixed5` | Comma-separated list of InceptionV3 layers to maximize. |
| `--steps` | | `15` | Gradient ascent steps per octave. |
| `--step-size` | | `0.01` | Size of the update step for gradient ascent. |
| `--octaves` | | `3` | Number of scales (octaves) to process. |
| `--octave-scale`| | `1.4` | Scale multiplier between octaves. |
| `--tile-size` | | `512` | Size of tiles for gradient computation (helps save memory). |
| `--resize-width`| | `None` | Resize frame width (maintains aspect ratio, highly recommended for speed). |
| `--fps` | | *Source FPS* | Override the frame rate of the output video. |
| `--max-duration`| | `None` | Limit the output video duration in seconds (stops processing early). |
| `--pattern-scale-width`| | `320` | Reference width for pattern scaling. Keeps style waves/details consistent regardless of output resolution (set to 0 to disable). |
| `--use-flow` | `-f` | `False` | Use optical flow warping to make dream patterns follow the movement of objects in the video (requires -b > 0). |

---

## Presets and Visual Styles

You can use the `--preset` (or `-p`) argument to quickly configure the script for different visual aesthetics:

- **`default`**: A balanced aesthetic maximizing intermediate layers (`mixed3`, `mixed5`) with 3 octaves. Great for general hallucination patterns.
- **`edges`**: Focuses on early layers (`mixed0`, `mixed1`, `mixed2`) with fewer steps. Enhances edges, textures, and outlines, creating an abstract, painted look.
- **`detailed`**: Uses overlapping mid-level layers (`mixed3`, `mixed4`, `mixed5`) with more steps to extract sharp, repetitive fractal structures.
- **`trippy`**: Maximizes deep layers (`mixed5`, `mixed6`, `mixed7`) with 4 octaves. Produces complex, recognizable patterns like eyes, dog faces, and animal-like shapes.

---

## Examples

### 1. Run with Defaults (Original Size)
Apply the standard DeepDream effect to a video, keeping its original size and framerate:
```bash
.venv/bin/python3 deepdream_video.py --input IMG_1736.mov
```

### 2. Fast Render / Speed Optimization (Recommended for CPU)
Downscale the frame width to `640px` (preserving aspect ratio) to run much faster:
```bash
.venv/bin/python3 deepdream_video.py --input IMG_1736.mov --resize-width 640
```

### 3. Generate a Highly Complex "Trippy" Aesthetic
Produce animal-like shapes at `800px` width:
```bash
.venv/bin/python3 deepdream_video.py -i IMG_1736.mov -p trippy --resize-width 800
```

### 4. Create an Edge-Enhanced Outlined Look
Enhance textures and borders at `640px` width:
```bash
.venv/bin/python3 deepdream_video.py -i IMG_1736.mov -p edges --resize-width 640
```

### 5. Generate a Short Preview (Cut processing early)
Process only the first 5 seconds of the video to quickly test your settings:
```bash
.venv/bin/python3 deepdream_video.py -i IMG_1736.mov --max-duration 5 --resize-width 480
```

---

## Troubleshooting & Performance Optimization

- **Slow Renders**: DeepDream performs backpropagation on every frame. Processing standard 1080p or 4K videos at native resolution on a CPU will take a long time. Use `--resize-width 480` or `--resize-width 640` to drastically speed up generation.
- **Out of Memory (OOM)**: If you experience memory crashes, reduce the `--tile-size` to `256` or reduce the `--resize-width`.
- **No Audio in Output**: Ensure `ffmpeg` is installed on your Mac. You can check with `ffmpeg -version`. If it is missing, install it with `brew install ffmpeg`. Once installed, re-run your command and the tool will automatically copy the audio track.
