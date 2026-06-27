#!/usr/bin/env python3
"""
Video DeepDream CLI Tool
Applies Google's DeepDream effect to a video using TensorFlow 2.x and OpenCV.
"""

import os
import sys

# Suppress TensorFlow logging to keep console clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import argparse
import shutil
import subprocess
import time
from datetime import datetime
import numpy as np
import cv2
from tqdm import tqdm

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# -----------------------------------------------------------------------------
# Configuration Presets
# -----------------------------------------------------------------------------
PRESETS = {
    'default': {
        'layers': ['mixed3', 'mixed5'],
        'steps': 15,
        'step_size': 0.01,
        'octaves': 3,
        'octave_scale': 1.4,
    },
    'edges': {
        'layers': ['mixed0', 'mixed1', 'mixed2'],
        'steps': 10,
        'step_size': 0.015,
        'octaves': 2,
        'octave_scale': 1.5,
    },
    'detailed': {
        'layers': ['mixed3', 'mixed4', 'mixed5'],
        'steps': 25,
        'step_size': 0.015,
        'octaves': 3,
        'octave_scale': 1.3,
    },
    'trippy': {
        'layers': ['mixed5', 'mixed6', 'mixed7'],
        'steps': 20,
        'step_size': 0.01,
        'octaves': 4,
        'octave_scale': 1.4,
    }
}

# -----------------------------------------------------------------------------
# DeepDream Helper Functions
# -----------------------------------------------------------------------------
def get_dream_model(layer_names):
    """Loads InceptionV3 and extracts the chosen layers for dreaming."""
    print("Loading pre-trained InceptionV3 model (downloading weights if necessary)...")
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    
    # Verify layer names exist in the model
    for name in layer_names:
        try:
            base_model.get_layer(name)
        except ValueError:
            print(f"\nError: Layer '{name}' not found in InceptionV3 model.")
            print("Available layers include: mixed0, mixed1, mixed2, mixed3, mixed4, mixed5, mixed6, mixed7, mixed8, etc.")
            sys.exit(1)
            
    layers = [base_model.get_layer(name).output for name in layer_names]
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    return dream_model

def calc_loss(img, model):
    """Calculates the sum of the mean activations of all target layers."""
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for act in layer_activations:
        loss = tf.math.reduce_mean(act)
        losses.append(loss)

    return tf.reduce_sum(losses)

def deprocess(img):
    """Normalizes the image tensor back to 0-255 range."""
    img = 255.0 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)

def random_roll(img, maxroll):
    """Randomly shifts the image to avoid visible seams at tile boundaries."""
    maxroll = tf.cast(maxroll, tf.int32)
    # Ensure shift values are in bounds
    shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
    img_rolled = tf.roll(img, shift=shift, axis=[0, 1])
    return shift, img_rolled

class TiledGradients(tf.Module):
    """Calculates image gradients using tiles to save memory."""
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[2], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
        )
    )
    def __call__(self, img, img_size, tile_size=512):
        shift, img_rolled = random_roll(img, tile_size)

        # Initialize the image gradients to zero
        gradients = tf.zeros_like(img_rolled)
        
        # Skip the last tile, unless there's only one tile.
        xs = tf.range(0, img_size[1], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])
        ys = tf.range(0, img_size[0], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])

        for x in xs:
            for y in ys:
                # Calculate the gradients for this tile
                with tf.GradientTape() as tape:
                    tape.watch(img_rolled)
                    img_tile = img_rolled[y : y + tile_size, x : x + tile_size]
                    loss = calc_loss(img_tile, self.model)

                # Update the image gradients for this tile
                gradients = gradients + tape.gradient(loss, img_rolled)

        # Undo the random shift applied to the image and its gradients
        gradients = tf.roll(gradients, shift=-shift, axis=[0, 1])

        # Normalize the gradients
        gradients /= tf.math.reduce_std(gradients) + 1e-8 

        return gradients

def run_deep_dream_frame(img_np, dream_model, get_tiled_gradients, steps_per_octave, step_size, octaves_range, octave_scale, tile_size, target_width, target_height, pattern_scale_width=320):
    """Applies the multi-scale DeepDream algorithm to a single frame."""
    # Convert numpy array to tensorflow tensor
    img = tf.convert_to_tensor(img_np, dtype=tf.float32)
    base_shape = tf.shape(img)
    
    # Preprocess image (scales pixels to [-1, 1])
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    for octave in octaves_range:
        # Scale the image based on the octave and pattern_scale_width reference
        if pattern_scale_width is not None and pattern_scale_width > 0:
            ref_ratio = pattern_scale_width / target_width
            float_base_shape = tf.cast(base_shape[:-1], tf.float32) * ref_ratio
        else:
            float_base_shape = tf.cast(base_shape[:-1], tf.float32)
            
        new_size = tf.cast(float_base_shape * (octave_scale ** octave), tf.int32)
        img = tf.image.resize(img, new_size)

        for step in range(steps_per_octave):
            gradients = get_tiled_gradients(img, new_size, tf.constant(tile_size, dtype=tf.int32))
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1.0, 1.0)
            
    # Deprocess to get uint8 image [0, 255]
    result = deprocess(img)
    
    # Convert back to numpy array
    result_np = result.numpy()
    
    # Resize back to target output dimensions using fast OpenCV resize
    if result_np.shape[0] != target_height or result_np.shape[1] != target_width:
        result_np = cv2.resize(result_np, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
    return result_np

# -----------------------------------------------------------------------------
# Audio Merging Function
# -----------------------------------------------------------------------------
def merge_audio(original_video_path, processed_video_path, final_output_path):
    """Merges the audio track from the original video using ffmpeg subprocess."""
    if not shutil.which("ffmpeg"):
        print("\n[Warning] 'ffmpeg' command line tool was not found.")
        print("[Warning] Original audio could not be merged. Output video has no audio.")
        print("[Warning] To preserve audio, install ffmpeg (e.g., 'brew install ffmpeg' on macOS).")
        try:
            shutil.move(processed_video_path, final_output_path)
            print(f"Output saved to: {final_output_path}")
        except Exception as e:
            print(f"Error saving output: {e}")
        return False

    print("\nMerging original audio track into the processed video using ffmpeg...")
    cmd = [
        "ffmpeg", "-y",
        "-i", processed_video_path,
        "-i", original_video_path,
        "-c:v", "copy",
        "-c:a", "copy",
        "-map", "0:v",
        "-map", "1:a?",
        "-shortest",
        final_output_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"Successfully merged audio! Output saved to: {final_output_path}")
            try:
                os.remove(processed_video_path)
            except Exception:
                pass
            return True
        else:
            print(f"[Warning] ffmpeg execution failed (exit code {result.returncode}).")
            print(f"[Warning] Error details: {result.stderr}")
            print("[Warning] Output video saved without audio.")
            shutil.move(processed_video_path, final_output_path)
            print(f"Output saved to: {final_output_path}")
            return False
    except Exception as e:
        print(f"[Warning] Failed to run ffmpeg: {e}")
        print("[Warning] Output video saved without audio.")
        shutil.move(processed_video_path, final_output_path)
        print(f"Output saved to: {final_output_path}")
        return False

# -----------------------------------------------------------------------------
# Main CLI Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="CLI tool to apply DeepDream effect to a video file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required Input Video
    parser.add_argument('--input', '-i', type=str, required=True,
                        help="Path to the input video file (e.g., .mov)")
    
    # Optional Output Directory
    parser.add_argument('--output-dir', '-o', type=str, default="output",
                        help="Directory to save the output video file")
    
    # Presets & Parameters
    parser.add_argument('--preset', '-p', type=str, choices=PRESETS.keys(), default=None,
                        help="Choose a preset config (overrides specific layer/step options)")
    parser.add_argument('--layers', type=str, default="mixed3,mixed5",
                        help="Comma-separated InceptionV3 layers to maximize (e.g., 'mixed3,mixed5')")
    parser.add_argument('--steps', type=int, default=15,
                        help="Gradient ascent steps per octave")
    parser.add_argument('--step-size', type=float, default=0.01,
                        help="Step size for gradient ascent")
    parser.add_argument('--octaves', type=int, default=3,
                        help="Number of octaves (scales) to process")
    parser.add_argument('--octave-scale', type=float, default=1.4,
                        help="Scale ratio between octaves")
    parser.add_argument('--tile-size', type=int, default=512,
                        help="Tile size for memory-safe gradient calculation")
    
    # Speed & Codec Optimizations
    parser.add_argument('--resize-width', type=int, default=None,
                        help="Resize video width (maintains aspect ratio, highly recommended for speed)")
    parser.add_argument('--fps', type=float, default=None,
                        help="Override frame rate of the output video")
    parser.add_argument('--max-duration', type=float, default=None,
                        help="Maximum duration of the output video in seconds (cuts processing short)")
    parser.add_argument('--blend-alpha', '-b', type=float, default=0.0,
                        help="Blend factor (0.0 to 1.0) with the previous dream frame for temporal consistency (e.g., 0.8)")
    parser.add_argument('--pattern-scale-width', type=int, default=320,
                        help="Reference width for pattern scaling. Keeps pattern size consistent regardless of output resolution (set to 0 to disable)")
    parser.add_argument('--use-flow', '-f', action='store_true',
                        help="Use optical flow to warp the previous frame's patterns so they follow moving objects (requires -b > 0)")
    parser.add_argument('--contrast-boost', type=float, default=1.0,
                        help="Boost contrast of the frame before dreaming (e.g., 1.2 or 1.5)")
    parser.add_argument('--saturation-boost', type=float, default=1.0,
                        help="Boost saturation of the frame before dreaming (e.g., 1.3 or 1.6)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input video file '{args.input}' does not exist.")
        sys.exit(1)

    # Determine parameters based on preset or specific arguments
    if args.preset:
        preset_cfg = PRESETS[args.preset]
        print(f"Using configuration preset: '{args.preset}'")
        layer_names = preset_cfg['layers']
        steps = preset_cfg['steps']
        step_size = preset_cfg['step_size']
        num_octaves = preset_cfg['octaves']
        octave_scale = preset_cfg['octave_scale']
    else:
        layer_names = [l.strip() for l in args.layers.split(',') if l.strip()]
        steps = args.steps
        step_size = args.step_size
        num_octaves = args.octaves
        octave_scale = args.octave_scale

    # Calculate octave ranges centered around 0
    octave_start = -(num_octaves // 2)
    octave_end = octave_start + num_octaves
    octaves_range = range(octave_start, octave_end)

    print(f"DeepDream Configuration:")
    print(f"  Layers: {', '.join(layer_names)}")
    print(f"  Octaves: {num_octaves} (range: {list(octaves_range)})")
    print(f"  Steps per octave: {steps}")
    print(f"  Step size: {step_size}")
    print(f"  Tile size: {args.tile_size}")

    # Load Model
    dream_model = get_dream_model(layer_names)
    get_tiled_gradients = TiledGradients(dream_model)

    # Open Input Video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video reader for '{args.input}'")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        total_frames = None

    # Calculate max frames if max-duration limit is set
    max_frames = None
    if args.max_duration is not None:
        max_frames = int(args.max_duration * fps)
        if total_frames is not None:
            total_frames = min(total_frames, max_frames)
        else:
            total_frames = max_frames

    print(f"\nSource Video Metadata:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames if total_frames is not None else 'Unknown'}")

    # Calculate target dimensions (even integers for codecs)
    if args.resize_width:
        target_width = args.resize_width
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)
        # Ensure dimensions are divisible by 2 for standard video codecs
        target_width = (target_width // 2) * 2
        target_height = (target_height // 2) * 2
        print(f"  Resizing to: {target_width}x{target_height} (for optimization)")
    else:
        target_width = width
        target_height = height

    # Setup directories and outputs
    os.makedirs(args.output_dir, exist_ok=True)
    start_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Include preset name in output filename if used
    preset_suffix = f"_{args.preset}" if args.preset else ""
    temp_output_path = os.path.join(args.output_dir, f"{start_timestamp}{preset_suffix}_temp.mov")
    final_output_path = os.path.join(args.output_dir, f"{start_timestamp}{preset_suffix}.mov")

    output_fps = args.fps or fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, output_fps, (target_width, target_height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for '{temp_output_path}'")
        cap.release()
        sys.exit(1)

    print(f"\nProcessing video frames...")
    
    # Initialize tqdm progress bar
    progress_bar = tqdm(
        total=total_frames,
        desc="DeepDreaming Video",
        unit="frame",
        dynamic_ncols=True
    )

    frame_idx = 0
    start_processing_time = time.time()
    previous_dream = None
    previous_raw_gray = None

    try:
        while cap.isOpened():
            if max_frames is not None and frame_idx >= max_frames:
                print(f"\nReached maximum duration limit of {args.max_duration} seconds ({max_frames} frames).")
                break

            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. Resize frame to target width/height if processing optimization is active
            if target_width != width or target_height != height:
                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            
            # Convert current frame to grayscale for optical flow
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Blend with previous frame if consistency blending is active
            if args.blend_alpha > 0.0 and previous_dream is not None:
                if args.use_flow and previous_raw_gray is not None:
                    # Calculate Farneback optical flow from current to previous frame (backward flow)
                    flow = cv2.calcOpticalFlowFarneback(
                        curr_gray, previous_raw_gray, None, 
                        pyr_scale=0.5, levels=3, winsize=15, 
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                    )
                    h, w = flow.shape[:2]
                    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
                    map_x = grid_x + flow[..., 0]
                    map_y = grid_y + flow[..., 1]
                    
                    # Warp the previous dream frame to follow the optical flow
                    warped_previous = cv2.remap(
                        previous_dream, map_x, map_y, 
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                    )
                    frame = cv2.addWeighted(warped_previous, args.blend_alpha, frame, 1.0 - args.blend_alpha, 0.0)
                else:
                    frame = cv2.addWeighted(previous_dream, args.blend_alpha, frame, 1.0 - args.blend_alpha, 0.0)
            
            # Save current raw grayscale for next frame's optical flow
            if args.blend_alpha > 0.0 and args.use_flow:
                previous_raw_gray = curr_gray.copy()

            # Apply contrast and saturation boost if requested
            if args.contrast_boost != 1.0 or args.saturation_boost != 1.0:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                if args.saturation_boost != 1.0:
                    hsv[..., 1] = np.clip(hsv[..., 1] * args.saturation_boost, 0, 255)
                frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                
                if args.contrast_boost != 1.0:
                    frame = np.clip((frame.astype(np.float32) - 127.5) * args.contrast_boost + 127.5, 0, 255).astype(np.uint8)
            
            # 2. Convert BGR (OpenCV standard) to RGB (TensorFlow/Dream standard)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
            
            # 3. Apply DeepDream algorithm
            dream_rgb = run_deep_dream_frame(
                img_np=frame_rgb,
                dream_model=dream_model,
                get_tiled_gradients=get_tiled_gradients,
                steps_per_octave=steps,
                step_size=step_size,
                octaves_range=octaves_range,
                octave_scale=octave_scale,
                tile_size=args.tile_size,
                target_width=target_width,
                target_height=target_height,
                pattern_scale_width=args.pattern_scale_width
            )
            
            # 4. Convert back to BGR for OpenCV VideoWriter
            dream_bgr = cv2.cvtColor(dream_rgb, cv2.COLOR_RGB2BGR)
            
            # 5. Write frame to output
            out.write(dream_bgr)
            
            # Save for next frame blending
            if args.blend_alpha > 0.0:
                previous_dream = dream_bgr.copy()
            
            frame_idx += 1
            progress_bar.update(1)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user. Saving partial video...")
    finally:
        cap.release()
        out.release()
        progress_bar.close()

    total_processing_time = time.time() - start_processing_time
    avg_fps = frame_idx / total_processing_time if total_processing_time > 0 else 0
    print(f"\nFinished processing {frame_idx} frames.")
    print(f"Total time elapsed: {total_processing_time:.2f}s | Avg Speed: {avg_fps:.2f} fps")

    # Merge audio if available
    merge_audio(args.input, temp_output_path, final_output_path)

if __name__ == '__main__':
    main()
