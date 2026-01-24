"""
Batch Processing Orchestrator
Handles batch processing of images and videos with parallel/sequential modes.
"""

import os
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from .config import (
    OUTPUT_DIR, DEVICE, VIDEO_EXTENSIONS, TRANSLATIONS
)
from .state import (
    check_processing_state, update_processing_state,
    frame_pairs, rgba_to_rgb_for_display
)
from .gpu import clear_gpu_memory, clear_gpu_memory_async
from .file_utils import separate_files_by_type
from .models import load_model
from .image_processing import upscale_image, save_image_with_format
from .video_processing import (
    extract_frames, get_video_fps, encode_video,
    analyze_duplicate_frames, save_frame_with_format,
    plan_parallel_video_processing
)


def upscale_image_worker(img_path, model, settings, vram_manager, img_session, output_format, jpeg_quality):
    """
    Worker function for parallel image upscaling.

    Args:
        img_path: Path to image file
        model: Model name string
        settings: Dict with upscaling parameters (preserve_alpha, use_fp16, target_scale, params)
        vram_manager: VRAMManager instance
        img_session: Output directory path
        output_format: Output image format (PNG/JPEG/WebP)
        jpeg_quality: Quality for JPEG/WebP

    Returns:
        Tuple of (output_path, result_array, orig_array, cuda_stream, error_message)
        cuda_stream is the CUDA stream used for GPU operations (for batch-level sync)
        error_message is None on success
    """
    try:
        # Acquire VRAM slot (blocks if all slots busy)
        vram_manager.acquire()

        # CRITICAL: Create a separate CUDA stream for this worker
        # This allows true parallel GPU execution across workers
        cuda_stream = None
        if DEVICE == "cuda":
            import torch
            cuda_stream = torch.cuda.Stream()

        try:
            # Execute GPU operations on dedicated stream
            img = Image.open(img_path)
            result, orig = upscale_image(
                img, model,
                settings['preserve_alpha'],
                output_format, jpeg_quality,
                settings['use_fp16'],
                target_scale=settings['target_scale'],
                target_resolution=0,
                is_video_frame=False,
                cuda_stream=cuda_stream,  # ✅ PASS STREAM TO INFERENCE
                **settings['params']
            )
            # ✅ NO SYNCHRONIZATION HERE - PyTorch auto-syncs during .cpu() inside upscale_image()
            # Results (PIL images) are ready to use - GPU work completed via automatic sync

            # Save output
            img_name = Path(img_path).stem
            output_path = img_session / f"{img_name}_upscaled"
            save_image_with_format(result, output_path, output_format, jpeg_quality)

            # Get final path with extension
            ext_map = {"PNG": ".png", "JPEG": ".jpg", "WebP": ".webp"}
            final_output_path = output_path.with_suffix(ext_map.get(output_format, ".png"))

            # Prepare display arrays
            result_array = rgba_to_rgb_for_display(result)
            orig_resized = orig.resize(result.size, Image.Resampling.LANCZOS)
            orig_array = rgba_to_rgb_for_display(orig_resized)

            # Cleanup
            orig.close()
            result.close()
            del img, result, orig, orig_resized

            return str(final_output_path), result_array, orig_array, cuda_stream, None

        finally:
            # Always release VRAM slot
            vram_manager.release()
            # ✅ Remove clear_gpu_memory_async() call from here
            # GPU cleanup happens AFTER all workers complete (batch-level)

    except Exception as e:
        return None, None, None, None, str(e)


def upscale_video_frame_worker(frame_path, model, settings, vram_manager, frame_format):
    """
    Worker function for parallel video frame upscaling.

    Args:
        frame_path: Path to input frame file
        model: Model name string
        settings: Dict with upscaling parameters (preserve_alpha, use_fp16, target_resolution, params)
        vram_manager: VRAMManager instance
        frame_format: Frame format key for intermediate save (PNG/JPEG)

    Returns:
        Tuple of (frame_path, result_img, orig_img, cuda_stream, error_message)
        - result_img and orig_img are PIL Image objects (NOT saved yet)
        - cuda_stream is the CUDA stream used for GPU operations (for batch-level sync)
        - error_message is None on success
    """
    try:
        # Acquire VRAM slot (blocks if all slots busy)
        vram_manager.acquire()

        # CRITICAL: Create a separate CUDA stream for this worker
        # This allows true parallel GPU execution across workers
        cuda_stream = None
        if DEVICE == "cuda":
            import torch
            cuda_stream = torch.cuda.Stream()

        try:
            # Execute GPU operations on dedicated stream
            img = Image.open(frame_path)
            result, orig = upscale_image(
                img, model,
                settings['preserve_alpha'],
                "PNG",  # Internal format (we'll save with frame_format later)
                95,     # Quality (unused for PNG)
                settings['use_fp16'],
                target_scale=2.0,  # Video frames always use 2x
                target_resolution=settings['target_resolution'],
                is_video_frame=True,
                cuda_stream=cuda_stream,  # ✅ PASS STREAM TO INFERENCE
                **settings['params']
            )
            # ✅ NO SYNCHRONIZATION HERE - PyTorch auto-syncs during .cpu() inside upscale_image()
            # Results (PIL images) are ready to use - GPU work completed via automatic sync

            # Return PIL images WITHOUT saving yet (will be saved in main thread)
            # This allows us to process frames in parallel, then save sequentially
            return str(frame_path), result, orig, cuda_stream, None

        finally:
            # Always release VRAM slot
            vram_manager.release()
            # ✅ Remove clear_gpu_memory_async() call from here
            # GPU cleanup happens AFTER all workers complete (batch-level)

    except Exception as e:
        return str(frame_path), None, None, None, str(e)


def _extract_gradio_value(value, default):
    """
    Extract actual value from Gradio component output.

    Gradio sometimes passes dict like {"value": X} instead of X directly.
    This helper ensures we always get the actual value.
    """
    if isinstance(value, dict):
        return value.get('value', default)
    return value if value is not None else default


def process_batch(files, model, image_scale_radio, video_resolution_dropdown, output_format, jpeg_quality, precision_mode, codec_name, profile_name, fps, preserve_alpha, export_video, keep_audio, frame_format,
                 auto_delete_input_frames, auto_delete_output_frames, auto_delete_frame_mapping, organize_videos_folder, skip_duplicate_frames,
                 use_auto_settings, tile_size, tile_overlap, sharpening, contrast, saturation,
                 video_naming_mode, video_suffix, video_custom_name, enable_parallel=True, vram_manager=None, progress=None):
    """Process multiple files with video export support, auto-cleanup, and duplicate frame detection"""

    # CRITICAL: Extract values from Gradio components (handles dict vs direct value)
    fps = _extract_gradio_value(fps, 0)
    jpeg_quality = _extract_gradio_value(jpeg_quality, 95)
    tile_size = _extract_gradio_value(tile_size, 512)
    tile_overlap = _extract_gradio_value(tile_overlap, 32)
    sharpening = _extract_gradio_value(sharpening, 0.0)
    contrast = _extract_gradio_value(contrast, 1.0)
    saturation = _extract_gradio_value(saturation, 1.0)
    video_target_resolution = _extract_gradio_value(video_resolution_dropdown, 0)

    # Convert precision mode to boolean or None
    if precision_mode == "None":
        use_fp16 = None  # None = No conversion, PyTorch decides
    elif precision_mode == "FP16 (Half Precision)":
        use_fp16 = True
    else:  # FP32 (Full Precision)
        use_fp16 = False

    # Conversion ×1/×2/×4/×8/×16 → float pour images
    scale_mapping = {"×1": 1.0, "×2": 2.0, "×4": 4.0, "×8": 8.0, "×16": 16.0}
    image_target_scale = scale_mapping.get(image_scale_radio, 2.0)

    # Determine parameters based on AUTO mode
    params = {}
    if not use_auto_settings:
        params = {
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "sharpening": sharpening,
            "contrast": contrast,
            "saturation": saturation
        }

    # Initialize processing state (thread-safe)
    update_processing_state("running", True)
    update_processing_state("paused", False)
    update_processing_state("stop", False)

    # Clear frame pairs at module level
    import app_upscale.state as state_module
    state_module.frame_pairs = []

    if not files:
        return None, None, "", "", {"visible": False}, {"visible": False}, ""

    # Separate images and videos
    images, videos = separate_files_by_type(files)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = OUTPUT_DIR / ts
    session.mkdir(exist_ok=True)

    all_results = []
    status_messages = []
    download_files = []

    total_files = len(images) + len(videos)

    # Process images
    if images:
        status_messages.append(f"📸 Processing {len(images)} image(s)...")

        # Smart folder organization: only create "images" subfolder if multiple images
        if len(images) == 1:
            img_session = session
        else:
            img_session = session / "images"
            img_session.mkdir(exist_ok=True)

        # Prepare settings dict for worker
        settings = {
            'preserve_alpha': preserve_alpha,
            'use_fp16': use_fp16,
            'target_scale': image_target_scale,
            'params': params
        }

        # Choose parallel or sequential processing
        if enable_parallel and vram_manager and len(images) > 1:
            # PARALLEL MODE: Process multiple images concurrently
            completed_count = 0
            image_futures = {}

            with ThreadPoolExecutor(max_workers=vram_manager.max_concurrent_jobs) as executor:
                # Submit all images for processing
                for idx, img_path in enumerate(images):
                    if check_processing_state("stop"):
                        break

                    future = executor.submit(
                        upscale_image_worker,
                        img_path, model, settings, vram_manager,
                        img_session, output_format, jpeg_quality
                    )
                    image_futures[future] = (idx, img_path)

                # Track all active CUDA streams for batch-level sync
                active_streams = []

                # Collect results as they complete
                for future in as_completed(image_futures):
                    if check_processing_state("stop"):
                        break

                    while check_processing_state("paused"):
                        time.sleep(0.1)
                        if check_processing_state("stop"):
                            break

                    idx, img_path = image_futures[future]
                    output_path, result_array, orig_array, stream, error = future.result()

                    if error:
                        status_messages.append(f"❌ Error processing {Path(img_path).name}: {error}")
                        continue

                    # Add to results
                    download_files.append(output_path)
                    all_results.append(result_array)
                    state_module.frame_pairs.append((orig_array, result_array))

                    # Track active stream for optional additional sync
                    if stream is not None:
                        active_streams.append(stream)

                    # Update progress
                    completed_count += 1
                    if progress:
                        progress(completed_count / (len(images) + len(videos)),
                                desc=f"Image {completed_count}/{len(images)}")

            status_messages.append(f"✅ {completed_count} image(s) completed (parallel mode)")

            # ✅ OPTIONAL: Additional synchronization for safety (belt-and-suspenders)
            # Note: Not strictly necessary since .cpu() already synced, but adds safety margin
            if DEVICE == "cuda" and active_streams:
                print(f"⏳ Final stream synchronization ({len(active_streams)} streams)...")
                for stream in active_streams:
                    stream.synchronize()
                print(f"✅ All GPU work verified complete")
                clear_gpu_memory()  # Now safe to clean up

        else:
            # SEQUENTIAL MODE: Process images one at a time (fallback)
            for idx, img_path in enumerate(images):
                if check_processing_state("stop"):
                    break

                while check_processing_state("paused"):
                    time.sleep(0.1)
                    if check_processing_state("stop"):
                        break

                if progress:
                    progress((idx + 1) / (len(images) + len(videos)), desc=f"Image {idx+1}/{len(images)}")

                img = Image.open(img_path)
                result, orig = upscale_image(img, model, preserve_alpha,
                                            output_format, jpeg_quality, use_fp16,
                                            target_scale=image_target_scale,
                                            target_resolution=0,
                                            is_video_frame=False, **params)

                img_name = Path(img_path).stem
                output_path = img_session / f"{img_name}_upscaled"
                save_image_with_format(result, output_path, output_format, jpeg_quality)

                # Add to download files list
                ext_map = {"PNG": ".png", "JPEG": ".jpg", "WebP": ".webp"}
                final_output_path = output_path.with_suffix(ext_map.get(output_format, ".png"))
                download_files.append(str(final_output_path))

                all_results.append(rgba_to_rgb_for_display(result))

                # Store for comparison with white background for display
                orig_resized = orig.resize(result.size, Image.Resampling.LANCZOS)
                state_module.frame_pairs.append((rgba_to_rgb_for_display(orig_resized), rgba_to_rgb_for_display(result)))

                # Free memory from PIL images
                # NOTE: img and orig are the same reference, so only close once
                orig.close()
                result.close()
                del img, result, orig, orig_resized

                # Clear GPU cache every 5 images to prevent memory accumulation
                if DEVICE == "cuda" and idx % 5 == 0:
                    clear_gpu_memory()

            status_messages.append(f"✅ {len(images)} image(s) completed")

    # Process videos with PARALLEL PROCESSING
    if videos:
        status_messages.append(f"🎬 Processing {len(videos)} video(s)...")

        for vid_idx, video_path in enumerate(videos):
            if check_processing_state("stop"):
                break

            vid_name = Path(video_path).stem

            # Organize videos based on user preference
            if organize_videos_folder:
                # Put video in dedicated output/videos/ folder (outside session)
                videos_output_dir = OUTPUT_DIR / "videos"
                videos_output_dir.mkdir(exist_ok=True)

                # Temporary processing folder in session
                vid_session = session / "temp_video_processing" / vid_name
            else:
                # Smart folder organization: only create "videos" subfolder if multiple videos
                if len(videos) == 1:
                    vid_session = session / vid_name
                else:
                    vid_session = session / "videos" / vid_name

            frames_in = vid_session / "input"
            frames_out = vid_session / "output"
            vid_session.mkdir(parents=True, exist_ok=True)
            frames_in.mkdir(); frames_out.mkdir()

            # Extract frames with verification
            if progress:
                progress(0.05, desc=f"Extracting frames from {vid_name}...")
            try:
                frames = extract_frames(video_path, str(frames_in))
                total_frames = len(frames)
            except RuntimeError as e:
                status_messages.append(f"{vid_name}: {str(e)}")
                continue

            if not total_frames:
                status_messages.append(f"❌ {vid_name}: No frames extracted")
                continue

            # Get original FPS
            original_fps = get_video_fps(video_path) if fps == 0 else fps

            start_time = time.time()

            # ============================================================
            # NEW PARALLEL PROCESSING SYSTEM
            # ============================================================

            # PHASE 1: Create processing plan (ALWAYS generates JSON mapping)
            if progress:
                progress(0.10, desc=f"{vid_name} - Planning parallel processing...")

            processing_plan = plan_parallel_video_processing(
                str(frames_in),
                detect_duplicates=skip_duplicate_frames,
                progress_callback=progress
            )

            if not processing_plan:
                status_messages.append(f"❌ {vid_name}: Failed to create processing plan")
                continue

            stats = processing_plan["stats"]
            frames_to_process = processing_plan["frames_to_process"]
            frame_output_mapping = processing_plan["frame_output_mapping"]

            # Report statistics with detailed debugging
            status_messages.append(f"📊 {vid_name}: Total frames: {stats['total_frames']}")
            status_messages.append(f"📊 {vid_name}: Unique frames: {stats['unique_frames']}")
            status_messages.append(f"📊 {vid_name}: Duplicate frames: {stats['duplicates']} ({stats['duplicate_percentage']:.1f}%)")
            status_messages.append(f"📊 {vid_name}: Parallel jobs planned: {stats['parallel_jobs']}")
            status_messages.append(f"📊 {vid_name}: frames_to_process length: {len(frames_to_process)}")

            # Debug: Check if parallel is enabled
            if enable_parallel:
                if vram_manager:
                    status_messages.append(f"🔧 {vid_name}: Parallel ENABLED - VRAM manager active ({vram_manager.max_concurrent_jobs} workers)")
                else:
                    status_messages.append(f"⚠️ {vid_name}: Parallel ENABLED but VRAM manager is None!")
            else:
                status_messages.append(f"⚠️ {vid_name}: Parallel processing is DISABLED")

            # PHASE 2: Upscale ONLY unique frames IN PARALLEL (optimized!)
            if progress:
                if skip_duplicate_frames and stats['duplicates'] > 0:
                    progress(0.15, desc=f"{vid_name} - Upscaling {stats['parallel_jobs']} unique frames (skipping {stats['duplicates']} duplicates)...")
                else:
                    progress(0.15, desc=f"{vid_name} - Upscaling {stats['parallel_jobs']} frames in parallel...")

            # Prepare settings for workers
            video_settings = {
                'preserve_alpha': preserve_alpha,
                'use_fp16': use_fp16,
                'target_resolution': video_target_resolution,
                'params': params
            }

            # Dictionary to store upscaled results: {unique_frame_path: (result_img, orig_img)}
            upscaled_results = {}
            frames_completed = 0

            # CRITICAL: frames_to_process now contains ONLY unique frames (duplicates excluded)
            # This is the key optimization: we only upscale what's necessary!
            unique_frame_count = len(frames_to_process)

            # Use ThreadPoolExecutor for parallel upscaling
            status_messages.append(f"🔍 {vid_name}: Checking parallel conditions:")
            status_messages.append(f"   - enable_parallel={enable_parallel}")
            status_messages.append(f"   - vram_manager={'OK' if vram_manager else 'None'}")
            status_messages.append(f"   - unique_frame_count={unique_frame_count} (need >1)")

            if enable_parallel and vram_manager and unique_frame_count > 1:
                # PARALLEL MODE - Upscale ONLY unique frames
                status_messages.append(f"✅ {vid_name}: PARALLEL MODE ACTIVATED - Using {vram_manager.max_concurrent_jobs} workers")
                with ThreadPoolExecutor(max_workers=vram_manager.max_concurrent_jobs) as executor:
                    # Submit ONLY unique frames for processing (duplicates skipped!)
                    future_to_frame = {}
                    for frame_path in frames_to_process:
                        if check_processing_state("stop"):
                            break

                        future = executor.submit(
                            upscale_video_frame_worker,
                            frame_path, model, video_settings, vram_manager, frame_format
                        )
                        future_to_frame[future] = frame_path

                    # Track all active CUDA streams for batch-level sync
                    active_streams = []

                    # Collect results as they complete
                    for future in as_completed(future_to_frame):
                        if check_processing_state("stop"):
                            break

                        while check_processing_state("paused"):
                            time.sleep(0.1)
                            if check_processing_state("stop"):
                                break

                        frame_path_str, result_img, orig_img, stream, error = future.result()

                        if error:
                            status_messages.append(f"❌ Error upscaling frame {Path(frame_path_str).name}: {error}")
                            continue

                        # Store result (PIL images already have synced data via .cpu())
                        # CRITICAL: Normalize path to match keys in frame_output_mapping
                        normalized_path = os.path.normpath(os.path.abspath(frame_path_str))
                        upscaled_results[normalized_path] = (result_img, orig_img)

                        # Track active stream for optional additional sync
                        if stream is not None:
                            active_streams.append(stream)

                        frames_completed += 1

                        # Update progress
                        if progress:
                            progress(0.15 + (frames_completed / unique_frame_count) * 0.7,
                                    desc=f"{vid_name} - Upscaled {frames_completed}/{unique_frame_count} unique frames")

                if skip_duplicate_frames and stats['duplicates'] > 0:
                    status_messages.append(f"⚡ {vid_name}: {frames_completed} unique frames upscaled (saved {stats['duplicates']} duplicate upscales!)")
                else:
                    status_messages.append(f"✅ {vid_name}: {frames_completed} frames upscaled in parallel")

                # ✅ OPTIONAL: Additional synchronization for safety (belt-and-suspenders)
                # Note: Not strictly necessary since .cpu() already synced, but adds safety margin
                if DEVICE == "cuda" and active_streams:
                    print(f"⏳ {vid_name}: Final stream synchronization ({len(active_streams)} streams)...")
                    for stream in active_streams:
                        stream.synchronize()
                    print(f"✅ {vid_name}: All GPU work verified complete")
                    clear_gpu_memory()  # Now safe to clean up

            else:
                # SEQUENTIAL FALLBACK (if parallel disabled or single unique frame)
                if not enable_parallel:
                    status_messages.append(f"⚠️ {vid_name}: SEQUENTIAL MODE - Parallel processing is DISABLED in settings")
                elif not vram_manager:
                    status_messages.append(f"⚠️ {vid_name}: SEQUENTIAL MODE - VRAM manager is None")
                elif unique_frame_count <= 1:
                    status_messages.append(f"⚠️ {vid_name}: SEQUENTIAL MODE - Only {unique_frame_count} unique frame (need >1 for parallel)")
                else:
                    status_messages.append(f"⚠️ {vid_name}: SEQUENTIAL MODE - Unknown reason")

                for frame_path in frames_to_process:
                    if check_processing_state("stop"):
                        break

                    while check_processing_state("paused"):
                        time.sleep(0.1)
                        if check_processing_state("stop"):
                            break

                    frame_path_str, result_img, orig_img, stream, error = upscale_video_frame_worker(
                        frame_path, model, video_settings, vram_manager, frame_format
                    )

                    if error:
                        status_messages.append(f"❌ Error upscaling frame {Path(frame_path_str).name}: {error}")
                        continue

                    # CRITICAL: Normalize path to match keys in frame_output_mapping
                    normalized_path = os.path.normpath(os.path.abspath(frame_path_str))
                    upscaled_results[normalized_path] = (result_img, orig_img)
                    frames_completed += 1

                    if progress:
                        progress(0.15 + (frames_completed / unique_frame_count) * 0.7,
                                desc=f"{vid_name} - Frame {frames_completed}/{unique_frame_count}")

                if skip_duplicate_frames and stats['duplicates'] > 0:
                    status_messages.append(f"⚡ {vid_name}: {frames_completed} unique frames upscaled (saved {stats['duplicates']} duplicate upscales!)")
                else:
                    status_messages.append(f"✅ {vid_name}: {frames_completed} frames upscaled (sequential)")

                # CRITICAL: Synchronize GPU ONLY AFTER all workers finished
                if DEVICE == "cuda":
                    clear_gpu_memory()

            # PHASE 3: Save frames in correct order (reconstruct full sequence)
            if progress:
                progress(0.85, desc=f"{vid_name} - Reconstructing sequence (saving {total_frames} frames)...")

            saved_frames_count = 0
            duplicates_copied = 0
            for frame_idx in range(total_frames):
                if check_processing_state("stop"):
                    break

                frame_info = frame_output_mapping[frame_idx]
                unique_frame = frame_info["unique_frame"]
                output_path = Path(frame_info["output_path"])
                is_duplicate = frame_info["is_duplicate"]

                # Get upscaled result for this unique frame
                if unique_frame not in upscaled_results:
                    # Debug: Show available keys to help diagnose path normalization issues
                    print(f"DEBUG: Looking for '{unique_frame}'")
                    print(f"DEBUG: Available keys: {list(upscaled_results.keys())[:3]}...")  # Show first 3
                    status_messages.append(f"⚠️ {vid_name}: Frame {frame_idx} missing from results")
                    continue

                result_img, orig_img = upscaled_results[unique_frame]

                # Save frame with chosen format
                # CRITICAL: For duplicates, we're reusing the same upscaled image (no re-upscaling!)
                save_frame_with_format(result_img, output_path.with_suffix(''), frame_format)
                saved_frames_count += 1

                # Track duplicates copied
                if is_duplicate:
                    duplicates_copied += 1

                # Progress update - show more frequently when duplicates are involved
                if progress:
                    if skip_duplicate_frames and stats['duplicates'] > 0:
                        # Show progress every frame to visualize duplicate copying
                        if is_duplicate:
                            progress(0.85 + (saved_frames_count / total_frames) * 0.10,
                                    desc=f"{vid_name} - Frame {saved_frames_count}/{total_frames} (copied duplicate)")
                        else:
                            progress(0.85 + (saved_frames_count / total_frames) * 0.10,
                                    desc=f"{vid_name} - Frame {saved_frames_count}/{total_frames} (saved unique)")
                    elif saved_frames_count % 10 == 0:
                        # Standard progress (every 10 frames)
                        progress(0.85 + (saved_frames_count / total_frames) * 0.10,
                                desc=f"{vid_name} - Saved {saved_frames_count}/{total_frames} frames")

                # Update UI with frame pair (only for first 100 frames to avoid memory issues)
                if frame_idx < 100:
                    all_results.append(rgba_to_rgb_for_display(result_img))
                    orig_resized = orig_img.resize(result_img.size, Image.Resampling.LANCZOS)
                    state_module.frame_pairs.append((
                        rgba_to_rgb_for_display(orig_resized),
                        rgba_to_rgb_for_display(result_img)
                    ))
                    orig_resized.close()
                    del orig_resized

            # Confirm all frames saved
            if saved_frames_count < total_frames:
                status_messages.append(f"⚠️ {vid_name}: Only saved {saved_frames_count}/{total_frames} frames (some missing)")
            else:
                if skip_duplicate_frames and duplicates_copied > 0:
                    status_messages.append(f"💾 {vid_name}: Saved {total_frames} frames ({frames_completed} upscaled + {duplicates_copied} copied from duplicates)")
                else:
                    status_messages.append(f"💾 {vid_name}: Successfully saved all {total_frames} frames")

            # Cleanup upscaled results from memory
            for result_img, orig_img in upscaled_results.values():
                result_img.close()
                orig_img.close()
            upscaled_results.clear()
            del upscaled_results

            # Clear GPU memory after processing all frames
            if DEVICE == "cuda":
                clear_gpu_memory()

            # Auto-delete input frames if enabled
            if auto_delete_input_frames:
                try:
                    shutil.rmtree(frames_in)
                    status_messages.append(f"🗑️ {vid_name}: Cleaned up input frames")
                except Exception as e:
                    print(f"⚠️ Failed to delete input frames: {e}")

            # Export video if requested
            if export_video and not check_processing_state("stop"):
                if progress:
                    progress(0.95, desc=f"Encoding {vid_name}...")

                # Determine extension based on codec
                ext_map = {
                    "H.264 (AVC)": ".mp4",
                    "H.265 (HEVC)": ".mp4",
                    "ProRes": ".mov",
                    "DNxHD/DNxHR": ".mov"
                }
                ext = ext_map.get(codec_name, ".mp4")

                # Determine output video name based on naming mode
                # Handle both English and French naming modes
                t_fr = TRANSLATIONS["fr"]
                t_en = TRANSLATIONS["en"]
                if video_naming_mode in ["Same as input", t_en.get("naming_same"), t_fr.get("naming_same"), "Même nom que l'original"]:
                    output_video_name = vid_name
                elif video_naming_mode in ["Add suffix", t_en.get("naming_suffix"), t_fr.get("naming_suffix"), "Ajouter un suffixe"]:
                    output_video_name = f"{vid_name}{video_suffix}"
                else:  # Custom name
                    # Use custom name if provided, otherwise fallback to original name
                    output_video_name = video_custom_name.strip() if video_custom_name.strip() else vid_name

                # Determine final output path
                if organize_videos_folder:
                    # Export directly to output/videos/
                    video_output = videos_output_dir / f"{output_video_name}{ext}"
                else:
                    # Export to session folder
                    video_output = vid_session / f"{output_video_name}{ext}"

                success, result_msg = encode_video(
                    str(frames_out),
                    str(video_output),
                    codec_name,
                    profile_name,
                    original_fps,
                    preserve_alpha,
                    video_path,
                    keep_audio
                )

                if success:
                    status_messages.append(f"✅ {vid_name}: Video exported to {video_output.parent.name}/{video_output.name} ({codec_name} - {profile_name})")
                    download_files.append(str(video_output))

                    # Auto-delete upscaled frames after successful encoding (if enabled)
                    if auto_delete_output_frames:
                        try:
                            shutil.rmtree(frames_out)
                            status_messages.append(f"🗑️ {vid_name}: Cleaned up upscaled frames")
                        except Exception as e:
                            status_messages.append(f"⚠️ {vid_name}: Failed to delete upscaled frames: {e}")
                else:
                    status_messages.append(f"⚠️ {vid_name}: {result_msg}")

            # Auto-delete parallel_processing_plan.json if enabled
            if auto_delete_frame_mapping:
                plan_file = vid_session / "parallel_processing_plan.json"
                if plan_file.exists():
                    try:
                        os.remove(plan_file)
                        status_messages.append(f"🗑️ {vid_name}: Deleted parallel_processing_plan.json")
                    except Exception as e:
                        print(f"⚠️ Failed to delete parallel_processing_plan.json: {e}")

            # Clean up empty video session folder if all contents were deleted
            if auto_delete_input_frames and auto_delete_output_frames and auto_delete_frame_mapping:
                try:
                    # Check if vid_session is empty or only contains the video file
                    remaining_items = list(vid_session.iterdir())
                    # Filter out the video file itself
                    remaining_items = [item for item in remaining_items if not item.is_file() or not str(item).endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))]

                    if len(remaining_items) == 0 or (len(remaining_items) == 0 and export_video):
                        # Folder is empty (except possibly the video), we can delete it
                        # But only if video was exported to videos/ folder (organize_videos_folder=True)
                        if organize_videos_folder and export_video:
                            # Video is in output/videos/, so we can safely delete vid_session
                            shutil.rmtree(vid_session)
                            status_messages.append(f"🗑️ {vid_name}: Cleaned up empty video processing folder")
                except Exception as e:
                    print(f"⚠️ Failed to clean up video session folder: {e}")

            # Add performance statistics with optimization details
            processing_time = time.time() - start_time
            if skip_duplicate_frames and stats["duplicates"] > 0:
                duplicates = stats["duplicates"]
                unique = stats["unique_frames"]
                percentage = stats["duplicate_percentage"]

                # Calculate theoretical speedup
                # Base: If we upscaled all frames sequentially
                # Optimized: Upscale only unique frames in parallel + copy duplicates
                if enable_parallel and vram_manager:
                    workers = vram_manager.max_concurrent_jobs
                    theoretical_speedup = (total_frames / unique) * workers
                    status_messages.append(
                        f"⚡ {vid_name}: OPTIMIZED - {duplicates} duplicates skipped ({percentage:.1f}%), "
                        f"{unique} unique frames upscaled with {workers} workers"
                    )
                else:
                    status_messages.append(
                        f"⚡ {vid_name}: {duplicates} duplicates skipped ({percentage:.1f}%), "
                        f"{unique} unique frames upscaled (sequential)"
                    )

                status_messages.append(f"⏱️ {vid_name}: Processing time: {processing_time:.1f}s")
            else:
                if enable_parallel and vram_manager:
                    workers = vram_manager.max_concurrent_jobs
                    status_messages.append(f"⏱️ {vid_name}: Processing time: {processing_time:.1f}s ({total_frames} frames, {workers} parallel workers)")
                else:
                    status_messages.append(f"⏱️ {vid_name}: Processing time: {processing_time:.1f}s ({total_frames} frames, sequential)")

            status_messages.append(f"✅ {vid_name}: {total_frames} frames processed")

    update_processing_state("running", False)
    if progress:
        progress(1.0)

    # Prepare outputs
    first_pair = state_module.frame_pairs[0] if state_module.frame_pairs else (None, None)
    final_status = "\n".join(status_messages)

    # Create download links text with full paths
    download_text = ""
    if download_files:
        file_list = []
        for f in download_files:
            file_path = Path(f)
            # Get file size
            try:
                size_bytes = file_path.stat().st_size
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024*1024:
                    size_str = f"{size_bytes/1024:.1f} KB"
                elif size_bytes < 1024*1024*1024:
                    size_str = f"{size_bytes/(1024*1024):.1f} MB"
                else:
                    size_str = f"{size_bytes/(1024*1024*1024):.2f} GB"

                file_list.append(f"• {file_path.name} ({size_str})\n  📁 {f}")
            except:
                file_list.append(f"• {file_path.name}\n  📁 {f}")

        download_text = f"📥 {len(download_files)} file(s) ready:\n\n" + "\n\n".join(file_list)

    # CRITICAL: Return simple values for Gradio components, not dicts
    # Gradio will handle the update internally based on component type
    frame_slider_value = 1
    frame_slider_maximum = max(1, len(state_module.frame_pairs))
    frame_slider_visible = len(state_module.frame_pairs) > 1

    frame_label_value = f"Frame 1/{len(state_module.frame_pairs)}"
    frame_label_visible = len(state_module.frame_pairs) > 1

    # For Gradio updates, we need to return a dict, but ensure it's properly formatted
    # The issue is that Gradio expects certain formats for slider updates
    frame_updates = {
        "maximum": frame_slider_maximum,
        "value": frame_slider_value,
        "visible": frame_slider_visible,
        "__type__": "update"  # Explicit type hint for Gradio
    }
    frame_label_update = {
        "value": frame_label_value,
        "visible": frame_label_visible,
        "__type__": "update"  # Explicit type hint for Gradio
    }

    return (first_pair, all_results, final_status, str(session),
            frame_updates, frame_label_update, download_text)
