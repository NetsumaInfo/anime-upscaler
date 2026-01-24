"""
Video Processing Module for Anime Upscaler

This module handles video frame extraction, duplicate detection, encoding,
and FFmpeg/FFprobe integration.
"""

import os
import json
import subprocess
import numpy as np
import time
import threading
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import VIDEO_CODECS, FRAME_FORMAT_OPTIONS, ENABLE_PARALLEL_HASH_DETECTION, HASH_WORKERS


# ============================================================================
# Frame Duplicate Detection (Perceptual Hashing)
# ============================================================================

def compute_frame_hash(img_path: str) -> str:
    """
    Compute perceptual hash of frame for duplicate detection.

    Uses perceptual hashing to detect visually identical frames even if they have
    minor pixel differences from compression artifacts.

    Args:
        img_path: Path to frame image file

    Returns:
        Hexadecimal hash string representing the perceptual fingerprint
    """
    with Image.open(img_path) as img:
        # Convert to RGB
        img_rgb = img.convert('RGB')

        # Resize to 32x32 for perceptual comparison (balances accuracy and tolerance)
        # Higher resolution = more strict matching, less false positives
        img_small = img_rgb.resize((32, 32), Image.Resampling.LANCZOS)

        # Convert to numpy array
        pixels = np.array(img_small, dtype=np.float32)

        # Compute average color per channel
        avg = pixels.mean(axis=(0, 1))

        # Create binary hash: 1 if pixel > average, 0 otherwise
        # This creates a perceptual fingerprint of the image structure
        binary = (pixels > avg).astype(np.uint8)

        # Convert to hashable string
        hash_str = binary.tobytes().hex()

    return hash_str


def analyze_duplicate_frames(
    frames_dir: str,
    progress_callback=None
) -> Optional[Dict]:
    """
    Analyze all frames and create a mapping of unique frames to their duplicates.

    This enables skipping upscaling of duplicate frames for huge performance gains
    (20-50% speedup on videos with static scenes).

    Args:
        frames_dir: Directory containing extracted PNG frames
        progress_callback: Optional callback(progress, desc) for progress updates

    Returns:
        Dictionary containing:
        {
            "unique_frames": {hash: [list of frame paths with this hash]},
            "frame_to_unique": {frame_path: unique_frame_path},
            "stats": {total, unique, duplicates, duplicate_percentage}
        }
        or None if no frames found
    """
    frames_path = Path(frames_dir)
    frames = sorted([f for f in frames_path.iterdir() if f.suffix.lower() == '.png'])

    if not frames:
        return None

    # Phase 1: Compute hashes for all frames
    hash_to_frames = {}  # {hash: [frame_paths]}
    frame_to_hash = {}   # {frame_path: hash}

    # Timing for performance measurement
    start_time = time.time()

    if ENABLE_PARALLEL_HASH_DETECTION and len(frames) > 1:
        # ===== PARALLEL MODE (ThreadPoolExecutor) =====
        print(f"🚀 Parallel hash detection enabled: {HASH_WORKERS} workers")

        # Thread-safe dict access (for hash_to_frames updates)
        hash_lock = threading.Lock()

        def _compute_hash_worker(frame_path):
            """Worker function for parallel hash computation"""
            norm_path = os.path.normpath(os.path.abspath(str(frame_path)))
            frame_hash = compute_frame_hash(norm_path)
            return norm_path, frame_hash

        # Parallel hash computation with ThreadPoolExecutor
        completed = 0
        with ThreadPoolExecutor(max_workers=HASH_WORKERS) as executor:
            # Submit all frames for processing
            futures = {executor.submit(_compute_hash_worker, frame): frame for frame in frames}

            # Collect results as they complete
            for future in as_completed(futures):
                norm_path, frame_hash = future.result()

                # Store results
                frame_to_hash[norm_path] = frame_hash

                # Thread-safe hash_to_frames update
                with hash_lock:
                    if frame_hash not in hash_to_frames:
                        hash_to_frames[frame_hash] = []
                    hash_to_frames[frame_hash].append(norm_path)

                # Progress update
                completed += 1
                if progress_callback:
                    progress_callback(
                        completed / len(frames),
                        desc=f"Analyzing frames {completed}/{len(frames)} (parallel)"
                    )

        elapsed = time.time() - start_time
        print(f"⏱️  Parallel hash detection: {elapsed:.2f}s ({len(frames)} frames)")

    else:
        # ===== SEQUENTIAL MODE (Fallback) =====
        if not ENABLE_PARALLEL_HASH_DETECTION:
            print("⚠️  Parallel hash detection disabled, using sequential mode")

        for i, frame_path in enumerate(frames):
            if progress_callback:
                progress_callback(
                    (i + 1) / len(frames),
                    desc=f"Analyzing frames {i+1}/{len(frames)} (sequential)"
                )

            # Normalize path to ensure consistency (resolve absolute path)
            norm_path = os.path.normpath(os.path.abspath(str(frame_path)))

            frame_hash = compute_frame_hash(norm_path)
            frame_to_hash[norm_path] = frame_hash

            if frame_hash not in hash_to_frames:
                hash_to_frames[frame_hash] = []
            hash_to_frames[frame_hash].append(norm_path)

        elapsed = time.time() - start_time
        print(f"⏱️  Sequential hash detection: {elapsed:.2f}s ({len(frames)} frames)")

    # Phase 2: Create mapping (first occurrence = unique frame)
    frame_to_unique = {}
    unique_frames = {}

    for frame_hash, frame_list in hash_to_frames.items():
        # First frame with this hash is the "unique" one
        unique_frame = frame_list[0]
        unique_frames[unique_frame] = frame_list

        # Map all frames to their unique representative
        for frame in frame_list:
            frame_to_unique[frame] = unique_frame

    # Stats
    total_frames = len(frames)
    unique_count = len(unique_frames)
    duplicate_count = total_frames - unique_count

    mapping = {
        "unique_frames": unique_frames,
        "frame_to_unique": frame_to_unique,
        "stats": {
            "total_frames": total_frames,
            "unique_frames": unique_count,
            "duplicates": duplicate_count,
            "duplicate_percentage": (
                (duplicate_count / total_frames * 100) if total_frames > 0 else 0
            )
        }
    }

    # Save mapping to JSON file for debugging/inspection
    mapping_file = frames_path.parent / "frame_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)

    return mapping


def plan_parallel_video_processing(
    frames_dir: str,
    detect_duplicates: bool = True,
    progress_callback=None
) -> Optional[Dict]:
    """
    Create an optimized parallel processing plan using duplicate detection results.

    PIPELINE INTELLIGENT:
    1. Si detect_duplicates=True → Génère frame_mapping.json avec analyse des doublons
    2. Utilise ce mapping → Génère parallel_processing_plan.json optimisé
    3. Résultat: Upscale uniquement les frames uniques, copie pour les doublons

    Args:
        frames_dir: Directory containing extracted PNG frames
        detect_duplicates: If True, uses duplicate detection to optimize. If False, treats all frames as unique.
        progress_callback: Optional callback(progress, desc) for progress updates

    Returns:
        Dictionary containing:
        {
            "frames_to_process": [list of ONLY unique frame paths to upscale],
            "frame_output_mapping": {frame_index: {input_path, unique_frame, output_path, is_duplicate}},
            "duplicate_mapping": {duplicate_frame_path: unique_frame_path},
            "stats": {
                "total_frames": int,
                "unique_frames": int,
                "duplicates": int,
                "duplicate_percentage": float,
                "parallel_jobs": int (ONLY unique frames)
            }
        }
        or None if no frames found
    """
    frames_path = Path(frames_dir)
    frames = sorted([f for f in frames_path.iterdir() if f.suffix.lower() == '.png'])

    if not frames:
        return None

    total_frames = len(frames)

    if detect_duplicates:
        # PHASE 1: DUPLICATE DETECTION → frame_mapping.json
        if progress_callback:
            progress_callback(0.0, desc="🔍 Analyzing duplicates...")

        # Use existing analyze_duplicate_frames() which already saves frame_mapping.json
        duplicate_analysis = analyze_duplicate_frames(str(frames_dir), progress_callback)

        if not duplicate_analysis:
            # Fallback: treat all as unique
            if progress_callback:
                progress_callback(0.5, desc="⚠️ Duplicate detection failed, treating all frames as unique")
            unique_frames_map = {}
            frame_to_unique = {}
            for frame_path in frames:
                norm_path = os.path.normpath(os.path.abspath(str(frame_path)))
                unique_frames_map[norm_path] = [norm_path]
                frame_to_unique[norm_path] = norm_path
            unique_count = total_frames
            duplicate_count = 0
        else:
            # Extract results from duplicate analysis
            unique_frames_map = duplicate_analysis["unique_frames"]
            frame_to_unique = duplicate_analysis["frame_to_unique"]
            stats = duplicate_analysis["stats"]
            unique_count = stats["unique_frames"]
            duplicate_count = stats["duplicates"]

            if progress_callback:
                progress_callback(0.5, desc=f"✅ Found {duplicate_count} duplicates ({stats['duplicate_percentage']:.1f}%)")

    else:
        # CASE 2: NO DUPLICATE DETECTION - treat all frames as unique
        if progress_callback:
            progress_callback(0.3, desc="📋 Planning parallel processing (all frames unique)...")

        unique_frames_map = {}
        frame_to_unique = {}

        for frame_path in frames:
            norm_path = os.path.normpath(os.path.abspath(str(frame_path)))
            unique_frames_map[norm_path] = [norm_path]
            frame_to_unique[norm_path] = norm_path

        unique_count = total_frames
        duplicate_count = 0

    # PHASE 2: BUILD OPTIMIZED PARALLEL PROCESSING PLAN
    if progress_callback:
        progress_callback(0.6, desc="📊 Building parallel processing plan...")

    # CRITICAL: frames_to_process contains ONLY unique frames (not duplicates)
    frames_to_process = list(unique_frames_map.keys())

    # Build output mapping for ALL frames (unique + duplicates)
    frames_out_dir = frames_path.parent / "output"
    frame_output_mapping = {}
    duplicate_mapping = {}  # {duplicate_path: unique_path} for quick lookup

    for i, frame_path in enumerate(frames):
        norm_path = os.path.normpath(os.path.abspath(str(frame_path)))
        unique_frame = frame_to_unique[norm_path]
        output_frame_path = frames_out_dir / f"frame_{i:05d}.png"
        is_duplicate = (unique_frame != norm_path)

        frame_output_mapping[i] = {
            "input_path": norm_path,
            "unique_frame": unique_frame,
            "output_path": str(output_frame_path),
            "is_duplicate": is_duplicate
        }

        # Track duplicate relationships for fast lookup
        if is_duplicate:
            duplicate_mapping[norm_path] = unique_frame

    # PHASE 3: CREATE FINAL PROCESSING PLAN
    processing_plan = {
        "frames_to_process": frames_to_process,  # ONLY unique frames
        "frame_output_mapping": frame_output_mapping,  # ALL frames with mapping info
        "duplicate_mapping": duplicate_mapping,  # Quick duplicate lookup
        "unique_frames_map": unique_frames_map,  # Full duplicate groups
        "stats": {
            "total_frames": total_frames,
            "unique_frames": unique_count,
            "duplicates": duplicate_count,
            "duplicate_percentage": (duplicate_count / total_frames * 100) if total_frames > 0 else 0,
            "parallel_jobs": unique_count  # ONLY unique frames will be upscaled
        }
    }

    # PHASE 4: SAVE parallel_processing_plan.json
    if progress_callback:
        progress_callback(0.8, desc="💾 Saving processing plan...")

    plan_file = frames_path.parent / "parallel_processing_plan.json"
    with open(plan_file, 'w') as f:
        # Convert all Path objects to strings for JSON serialization
        serializable_plan = {
            "frames_to_process": [str(p) for p in processing_plan["frames_to_process"]],
            "frame_output_mapping": processing_plan["frame_output_mapping"],
            "duplicate_mapping": {str(k): str(v) for k, v in duplicate_mapping.items()},
            "stats": processing_plan["stats"]
        }
        json.dump(serializable_plan, f, indent=2)

    if progress_callback:
        if detect_duplicates and duplicate_count > 0:
            progress_callback(1.0, desc=f"⚡ Plan ready: {unique_count} unique frames (skipping {duplicate_count} duplicates)")
        else:
            progress_callback(1.0, desc=f"✅ Plan ready: {unique_count} frames to upscale in parallel")

    return processing_plan


# ============================================================================
# Frame Saving with Intermediate Format Support
# ============================================================================

def save_frame_with_format(
    img: Image.Image,
    path: Path,
    frame_format_name: str
) -> None:
    """
    Save frame with specified intermediate format.

    This is used for video frames before encoding, allowing quality/space trade-offs.

    Args:
        img: PIL Image to save
        path: Output path (extension will be adjusted)
        frame_format_name: Format key from FRAME_FORMAT_OPTIONS
    """
    config = FRAME_FORMAT_OPTIONS[frame_format_name]

    if config["format"] == "PNG":
        # Handle bit depth for PNG
        if config["bits"] == 16:
            # Convert to 16-bit PNG
            if img.mode == 'RGB':
                # For RGB 16-bit, save with bits flag
                img.save(
                    path.with_suffix('.png'),
                    'PNG',
                    compress_level=config["compress_level"],
                    bits=16
                )
            else:
                img.save(
                    path.with_suffix('.png'),
                    'PNG',
                    compress_level=config["compress_level"]
                )
        else:
            # 8-bit PNG
            img.save(
                path.with_suffix('.png'),
                'PNG',
                compress_level=config["compress_level"]
            )

    elif config["format"] == "JPEG":
        # Convert RGBA to RGB for JPEG (no transparency support)
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(
                img,
                mask=img.split()[3] if img.mode == 'RGBA' else img.split()[1]
            )
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        img.save(
            path.with_suffix('.jpg'),
            'JPEG',
            quality=config["quality"],
            optimize=True
        )


# ============================================================================
# Video Metadata Extraction (FFprobe)
# ============================================================================

def get_video_frame_count(video_path: str) -> Optional[int]:
    """
    Get total frame count of video using FFprobe.

    Uses -count_packets method with fallback to duration×FPS calculation.

    Args:
        video_path: Path to video file

    Returns:
        Frame count, or None if detection failed
    """
    try:
        # Primary method: count packets
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            video_path
        ], capture_output=True, text=True)
        frame_count = int(result.stdout.strip())
        return frame_count
    except:
        # Fallback method: duration × FPS
        try:
            duration_result = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ], capture_output=True, text=True)
            duration = float(duration_result.stdout.strip())
            fps = get_video_fps(video_path)
            return int(duration * fps)
        except:
            return None


def get_video_fps(video_path: str) -> float:
    """
    Get FPS (frames per second) of input video.

    Args:
        video_path: Path to video file

    Returns:
        FPS as float, or 24.0 as fallback
    """
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ], capture_output=True, text=True)
        fps_str = result.stdout.strip()

        # Parse fractional FPS (e.g., "30000/1001" for 29.97)
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            return num / den

        return float(fps_str)
    except:
        return 24.0  # Default fallback


# ============================================================================
# Frame Extraction (FFmpeg)
# ============================================================================

def extract_frames(video: str, out_dir: str) -> list:
    """
    Extract frames from video with alpha channel support and verification.

    Verifies that extraction completed successfully before returning.

    Args:
        video: Path to input video file
        out_dir: Output directory for frames

    Returns:
        List of extracted frame paths (sorted)

    Raises:
        RuntimeError: If frame extraction failed or was incomplete
    """
    os.makedirs(out_dir, exist_ok=True)

    # Get expected frame count from video
    expected_frames = get_video_frame_count(video)

    # Extract frames with RGBA support (preserves alpha channel)
    result = subprocess.run([
        "ffmpeg", "-i", video,
        "-pix_fmt", "rgba",  # Support alpha channel
        "-start_number", "0",
        os.path.join(out_dir, "frame_%05d.png"),
        "-y"
    ], capture_output=True, text=True)

    # Verify extraction
    extracted_frames = sorted([
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.endswith('.png')
    ])
    extracted_count = len(extracted_frames)

    # Check if extraction was complete
    if expected_frames is not None:
        if extracted_count == 0:
            raise RuntimeError(
                f"❌ Frame extraction failed: No frames extracted from video. "
                f"FFmpeg error: {result.stderr}"
            )
        elif extracted_count < expected_frames:
            raise RuntimeError(
                f"❌ Incomplete frame extraction: Expected {expected_frames} frames "
                f"but only extracted {extracted_count} frames. "
                f"FFmpeg may have encountered an error."
            )
        elif extracted_count > expected_frames:
            # This shouldn't happen but log it as a warning
            print(f"⚠️ Warning: Extracted {extracted_count} frames but expected {expected_frames}")
    else:
        # Couldn't detect expected count, just verify we got something
        if extracted_count == 0:
            raise RuntimeError(
                f"❌ Frame extraction failed: No frames extracted from video. "
                f"FFmpeg error: {result.stderr}"
            )

    print(
        f"✅ Successfully extracted {extracted_count} frames"
        f"{f' (expected: {expected_frames})' if expected_frames else ''}"
    )

    return extracted_frames


# ============================================================================
# Video Encoding (FFmpeg)
# ============================================================================

def encode_video(
    frames_dir: str,
    output_path: str,
    codec_name: str,
    profile_name: str,
    fps: float,
    preserve_alpha: bool = True,
    original_video_path: Optional[str] = None,
    keep_audio: bool = False
) -> Tuple[bool, str]:
    """
    Encode video from frames with specified codec and profile.

    Optionally copies audio track from original video.

    Args:
        frames_dir: Directory containing frame_%05d.png files
        output_path: Output video file path
        codec_name: Codec name from VIDEO_CODECS (e.g., "H.264 (AVC)")
        profile_name: Profile name for the codec
        fps: Target frames per second
        preserve_alpha: Preserve transparency if codec supports it
        original_video_path: Path to original video for audio extraction
        keep_audio: Whether to copy audio from original video

    Returns:
        Tuple of (success, message_or_path)
        - success=True: message_or_path is output_path
        - success=False: message_or_path is error message
    """
    codec_config = VIDEO_CODECS[codec_name]
    profile_config = codec_config["profiles"][profile_name]
    codec = codec_config["codec"]

    # Check if alpha should be preserved
    has_alpha_support = codec_config["alpha_support"] and preserve_alpha

    # Build FFmpeg command
    # CRITICAL: Address "Washed Out" colors (PC vs TV Range mismatch)
    # 1. Input: PNGs are Full Range (0-255). We tell FFmpeg this explicitly.
    # 2. Filter: We convert to TV Range (16-235) because that's what video players expect.
    #    If we sent Full Range to a player, it might interpret 0 as gray (lifted blacks) or clip whites.
    # 3. Output: We tag the video as TV Range (bt709) so the player knows how to display it.

    cmd = [
        "ffmpeg", "-y",
        # Input properties: Our PNGs are sRGB Full Range
        "-color_range", "pc",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
    ]

    # Add original video as audio source if keep_audio is enabled
    if keep_audio and original_video_path:
        cmd.extend(["-i", original_video_path])

    # Standard Output Metadata (HD Standard Rec.709, TV Range)
    # This metadata tells the video player "Treat this as standard TV video"
    color_metadata = [
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-color_range", "tv"  # TV Range (Limited) is standard for video compatibility
    ]

    # Codec-specific settings
    if codec_name == "H.264 (AVC)":
        cmd.extend([
            "-c:v", codec,
            "-preset", profile_config["preset"],
            "-crf", str(profile_config["crf"]),
            "-profile:v", profile_config["profile"],
            # FILTER: Scale PC Range (0-255) to TV Range (16-235) + Convert RGB to YUV420P
            "-vf", "scale=in_range=pc:out_range=tv,format=yuv420p",
            "-pix_fmt", "yuv420p"
        ])
        cmd.extend(color_metadata)

    elif codec_name == "H.265 (HEVC)":
        pix_fmt = profile_config.get("pix_fmt", "yuv420p")
        cmd.extend([
            "-c:v", codec,
            "-preset", profile_config["preset"],
            "-crf", str(profile_config["crf"]),
            "-tag:v", "hvc1",
        ])

        if "profile" in profile_config:
            cmd.extend(["-profile:v", profile_config["profile"]])

        # FILTER: Scale PC Range (0-255) to TV Range (16-235)
        cmd.extend(["-vf", f"scale=in_range=pc:out_range=tv,format={pix_fmt}"])
        cmd.extend(["-pix_fmt", pix_fmt])
        cmd.extend(color_metadata)

    elif codec_name == "ProRes":
        pix_fmt = profile_config["pix_fmt"] if has_alpha_support else "yuv422p10le"
        cmd.extend([
            "-c:v", codec,
            "-profile:v", profile_config["profile"],
            # ProRes handles its own color typically, but explicit conversion helps consistency
            "-vf", f"scale=in_range=pc:out_range=tv,format={pix_fmt}",
            "-pix_fmt", pix_fmt,
            "-vendor", "apl0"
        ])
        cmd.extend(color_metadata)

    elif codec_name == "DNxHD/DNxHR":
        profile = profile_config["profile"]

        # DNxHR often works with full range, but standardizing on TV range resolves
        # the "washed out" issue for most common players unless using specialized NLEs.

        if "dnxhr" in profile:
            cmd.extend(["-c:v", profile])

            if has_alpha_support and "444" in profile:
                dnx_pix_fmt = "yuva444p10le"
            else:
                dnx_pix_fmt = "yuv422p10le"

            cmd.extend(["-vf", f"scale=in_range=pc:out_range=tv,format={dnx_pix_fmt}"])
            cmd.extend(["-pix_fmt", dnx_pix_fmt])
        else:
            # DNxHD
            cmd.extend([
                "-c:v", "dnxhd",
                "-b:v", profile_config["bitrate"],
                "-vf", "scale=in_range=pc:out_range=tv,format=yuv422p",
                "-pix_fmt", "yuv422p"
            ])

        cmd.extend(color_metadata)

    # Add audio settings if keeping audio
    if keep_audio and original_video_path:
        cmd.extend([
            "-map", "0:v:0",      # Map video from frames input
            "-map", "1:a:0?",     # Map audio from original video (optional with ?)
            "-c:a", "aac",        # Encode audio as AAC
            "-b:a", "192k",       # Audio bitrate
            "-shortest"           # End when shortest stream ends
        ])

    cmd.append(output_path)

    # Run encoding
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return False, f"Encoding error: {result.stderr}"

    return True, output_path
