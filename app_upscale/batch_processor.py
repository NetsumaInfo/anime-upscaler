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
from .gpu import clear_gpu_memory, clear_gpu_memory_async, get_model_dtype
from .file_utils import separate_files_by_type
from .models import load_model
from .image_processing import upscale_image, save_image_with_format, upscale_batch
from .video_processing import (
    extract_frames, get_video_fps, encode_video,
    analyze_duplicate_frames, save_frame_with_format,
    plan_parallel_video_processing
)


def _apply_pre_downscale(img, pre_downscale_height):
    """
    Apply pre-downscale to image if enabled and needed.

    Args:
        img: PIL Image to downscale
        pre_downscale_height: Target height (0 = no downscale)

    Returns:
        Downscaled image (or original if not applicable)
    """
    if pre_downscale_height <= 0:
        return img

    orig_width, orig_height = img.size

    # Only downscale if image is larger than target
    if orig_height > pre_downscale_height:
        # Calculate new width (preserve aspect ratio)
        new_width = int(orig_width * (pre_downscale_height / orig_height))
        new_height = pre_downscale_height

        # Apply downscale with LANCZOS (high quality)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img


def upscale_image_worker(img_path, model, settings, vram_manager, img_session, output_format, jpeg_quality, custom_name=None):
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
        custom_name: Optional custom name for output file (without extension)

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

            # PRÉ-DOWNSCALE (v2.8+): Apply before upscaling if enabled
            img = _apply_pre_downscale(img, settings.get('pre_downscale_height', 0))

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
            # ✅ CUDA stream is synchronized inside upscale_image() before .cpu() transfer
            # Results (PIL images) are ready to use - GPU work completed

            # Save output (use custom name if provided)
            img_name = custom_name if custom_name else Path(img_path).stem
            output_path = img_session / img_name
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
        import threading
        thread_id = threading.current_thread().ident
        frame_name = Path(frame_path).name if isinstance(frame_path, (str, Path)) else 'unknown'

        t_start = time.time()
        print(f"[Thread {thread_id:05d}] START {frame_name}")

        # Acquire VRAM slot (blocks if all slots busy)
        vram_manager.acquire()
        t_acquired = time.time()
        print(f"[Thread {thread_id:05d}] ACQUIRED VRAM after {t_acquired - t_start:.3f}s wait")

        # CRITICAL: Create a separate CUDA stream for this worker
        # This allows true parallel GPU execution across workers
        cuda_stream = None
        if DEVICE == "cuda":
            import torch
            cuda_stream = torch.cuda.Stream()

        try:
            # Execute GPU operations on dedicated stream
            img = Image.open(frame_path)

            # PRÉ-DOWNSCALE (v2.8+): Apply before upscaling if enabled
            img = _apply_pre_downscale(img, settings.get('pre_downscale_height', 0))

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
            # ✅ CUDA stream is synchronized inside upscale_image() before .cpu() transfer
            # Results (PIL images) are ready to use - GPU work completed

            t_upscaled = time.time()
            print(f"[Thread {thread_id:05d}] UPSCALED {frame_name} in {t_upscaled - t_acquired:.3f}s (total {t_upscaled - t_start:.3f}s)")

            # Return PIL images WITHOUT saving yet (will be saved in main thread)
            # This allows us to process frames in parallel, then save sequentially
            return str(frame_path), result, orig, cuda_stream, None

        finally:
            # Always release VRAM slot
            vram_manager.release()
            t_released = time.time()
            print(f"[Thread {thread_id:05d}] RELEASED VRAM for {frame_name} (total {t_released - t_start:.3f}s)")
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


def process_batch(files, model, image_scale_radio, video_resolution_dropdown, pre_downscale_dropdown, output_format, jpeg_quality, precision_mode, codec_name, profile_name, fps, preserve_alpha, export_video, keep_audio, frame_format,
                 auto_delete_input_frames, auto_delete_output_frames, auto_delete_extraction_folder, auto_delete_frame_mapping, organize_videos_folder, organize_images_folder, skip_duplicate_frames,
                 use_auto_settings, tile_size, tile_overlap, sharpening, contrast, saturation,
                 video_naming_mode, video_suffix, video_custom_name, enable_parallel=True, parallel_workers=2, file_rename_textbox="", vram_manager=None, progress=None):
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

    # ============================================================================
    # CONVERSION PRÉ-DOWNSCALE : UI Label → Height Value (v2.8+)
    # ============================================================================
    pre_downscale_height = 0  # Default: no downscale
    pre_downscale_value = _extract_gradio_value(pre_downscale_dropdown, "Original")
    if pre_downscale_value and isinstance(pre_downscale_value, str):
        pre_downscale_lower = pre_downscale_value.lower()
        if "480" in pre_downscale_lower:
            pre_downscale_height = 480
        elif "720" in pre_downscale_lower:
            pre_downscale_height = 720
        elif "1080" in pre_downscale_lower:
            pre_downscale_height = 1080
        # Sinon reste 0 (Original ou valeur inconnue)

    # Parse custom file names from textbox (one name per line)
    custom_file_names = []
    if file_rename_textbox and file_rename_textbox.strip():
        custom_file_names = [line.strip() for line in file_rename_textbox.strip().split('\n') if line.strip()]
    
    # CRITICAL: Handle parallel_workers with extra validation
    # This parameter can sometimes come as a dict from Gradio
    try:
        pw_value = _extract_gradio_value(parallel_workers, 2)
        parallel_workers = int(pw_value) if pw_value is not None else 2
    except (TypeError, ValueError) as e:
        print(f"⚠️ Warning: Failed to extract parallel_workers value ({type(parallel_workers)}): {e}")
        print(f"   Raw value: {parallel_workers}")
        parallel_workers = 2  # Safe fallback

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

    # Update VRAMManager with user-configured parallel workers count
    if vram_manager is not None and enable_parallel:
        vram_manager.update_max_jobs(parallel_workers)
        print(f"👷 Parallel workers set to: {parallel_workers}")

    if not files:
        return None, None, "", "", {"visible": False}, {"visible": False}, ""

    # Separate images and videos
    images, videos = separate_files_by_type(files)

    # Create mapping: file_path -> custom_name
    file_custom_names = {}
    if custom_file_names:
        # Map custom names to files in order
        for idx, file_path in enumerate(files):
            if idx < len(custom_file_names):
                file_custom_names[str(Path(file_path).absolute())] = custom_file_names[idx]

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

        # Smart folder organization based on organize_images_folder setting
        if organize_images_folder:
            # Export to dedicated images/ folder (like videos/)
            img_session = OUTPUT_DIR / "images"
            img_session.mkdir(exist_ok=True)
        else:
            # Original behavior: only create "images" subfolder if multiple images
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
            'pre_downscale_height': pre_downscale_height,  # v2.8+
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

                    # Get custom name if available
                    custom_name = file_custom_names.get(str(Path(img_path).absolute()))

                    future = executor.submit(
                        upscale_image_worker,
                        img_path, model, settings, vram_manager,
                        img_session, output_format, jpeg_quality, custom_name
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

            # ✅ BATCH-LEVEL CLEANUP: All streams already synchronized inside workers
            # This is redundant with per-worker sync but ensures all GPU ops complete before cleanup
            if DEVICE == "cuda" and active_streams:
                print(f"⏳ Final cleanup ({len(active_streams)} streams)...")
                for stream in active_streams:
                    stream.synchronize()  # Redundant but safe
                print(f"✅ All GPU work verified complete")
                clear_gpu_memory()  # Now safe to clean up

        else:
            # SEQUENTIAL MODE: Process images one at a time (fallback)
            # OPTIMIZATION: Async preloading + async saving (Phase 2.1 + 2.3)
            from .config import ENABLE_ASYNC_PRELOAD, ENABLE_ASYNC_SAVE

            if (ENABLE_ASYNC_PRELOAD or ENABLE_ASYNC_SAVE) and len(images) > 1:
                # Use async preloading and/or async saving for better pipeline utilization
                prefetch_pool = None
                save_pool = None
                future_images = {}
                save_futures = []

                if ENABLE_ASYNC_PRELOAD:
                    prefetch_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ImagePrefetch")
                    # Preload first image
                    if len(images) > 0:
                        future_images[0] = prefetch_pool.submit(Image.open, images[0])

                if ENABLE_ASYNC_SAVE:
                    save_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ImageSaver")

                try:
                    for idx, img_path in enumerate(images):
                        if check_processing_state("stop"):
                            break

                        while check_processing_state("paused"):
                            time.sleep(0.1)
                            if check_processing_state("stop"):
                                break

                        if progress:
                            progress((idx + 1) / (len(images) + len(videos)), desc=f"Image {idx+1}/{len(images)}")

                        # Get current image (from prefetch if available)
                        if ENABLE_ASYNC_PRELOAD and idx in future_images:
                            img = future_images[idx].result()
                        else:
                            img = Image.open(img_path)

                        # Launch prefetch for NEXT image BEFORE upscaling current one
                        if ENABLE_ASYNC_PRELOAD and idx + 1 < len(images):
                            future_images[idx+1] = prefetch_pool.submit(Image.open, images[idx+1])

                        # PRÉ-DOWNSCALE (v2.8+): Apply before upscaling if enabled
                        img = _apply_pre_downscale(img, pre_downscale_height)

                        # GPU upscales current image WHILE CPU loads next image in background
                        result, orig = upscale_image(img, model, preserve_alpha,
                                                    output_format, jpeg_quality, use_fp16,
                                                    target_scale=image_target_scale,
                                                    target_resolution=0,
                                                    is_video_frame=False, **params)

                        # Use custom name if available
                        custom_name = file_custom_names.get(str(Path(img_path).absolute()))
                        img_name = custom_name if custom_name else Path(img_path).stem
                        output_path = img_session / f"{img_name}_upscaled"

                        # PHASE 2.3: Save asynchronously in background (if enabled)
                        if ENABLE_ASYNC_SAVE:
                            # Submit save job to background thread
                            save_future = save_pool.submit(
                                save_image_with_format,
                                result.copy(),  # Create copy for async save
                                output_path,
                                output_format,
                                jpeg_quality
                            )
                            save_futures.append(save_future)
                        else:
                            # Synchronous save (blocks until done)
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
                finally:
                    # Wait for all async saves to complete
                    if ENABLE_ASYNC_SAVE and save_pool:
                        for save_future in save_futures:
                            save_future.result()
                        save_pool.shutdown(wait=True)

                    if ENABLE_ASYNC_PRELOAD and prefetch_pool:
                        prefetch_pool.shutdown(wait=True)

                # Build status message
                optimizations = []
                if ENABLE_ASYNC_PRELOAD:
                    optimizations.append("async preload")
                if ENABLE_ASYNC_SAVE:
                    optimizations.append("async save")
                opt_str = " + ".join(optimizations) if optimizations else "sequential"
                status_messages.append(f"✅ {len(images)} image(s) completed ({opt_str})")

            else:
                # Standard sequential mode (no preloading)
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

                    # PRÉ-DOWNSCALE (v2.8+): Apply before upscaling if enabled
                    img = _apply_pre_downscale(img, pre_downscale_height)

                    result, orig = upscale_image(img, model, preserve_alpha,
                                                output_format, jpeg_quality, use_fp16,
                                                target_scale=image_target_scale,
                                                target_resolution=0,
                                                is_video_frame=False, **params)

                    # Use custom name if available
                    custom_name = file_custom_names.get(str(Path(img_path).absolute()))
                    img_name = custom_name if custom_name else Path(img_path).stem
                    output_path = img_session / img_name
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

            # Get custom name if available (for folder naming)
            custom_name = file_custom_names.get(str(Path(video_path).absolute()))
            folder_name = custom_name if custom_name else vid_name

            # Organize videos based on user preference
            if organize_videos_folder:
                # Put video in dedicated output/videos/ folder (outside session)
                videos_output_dir = OUTPUT_DIR / "videos"
                videos_output_dir.mkdir(exist_ok=True)

                # Temporary processing folder in session
                vid_session = session / "temp_video_processing" / folder_name
            else:
                # Smart folder organization: only create "videos" subfolder if multiple videos
                if len(videos) == 1:
                    vid_session = session / folder_name
                else:
                    vid_session = session / "videos" / folder_name

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

            # Initialize stats with default values (will be populated by pipeline or sequential mode)
            stats = {
                "total_frames": total_frames,
                "unique_frames": total_frames,
                "duplicates": 0,
                "duplicate_percentage": 0.0
            }

            # ============================================================
            # CHOOSE PROCESSING MODE: GPU Pipeline vs Sequential
            # ============================================================
            from .config import ENABLE_GPU_PIPELINE, PIPELINE_MIN_FRAMES

            # Determine if GPU-first pipeline should be used
            use_gpu_pipeline = (
                ENABLE_GPU_PIPELINE and
                total_frames >= PIPELINE_MIN_FRAMES and
                enable_parallel and
                vram_manager is not None
            )

            if use_gpu_pipeline:
                # ============================================================
                # NEW ASYNC PIPELINE + BATCHING (v3.0 - OPTIMIZED)
                # ============================================================
                from .pipeline import ConcurrentPipeline
                import torch

                # Prepare upscale parameters
                upscale_params = {
                    "preserve_alpha": preserve_alpha,
                    "use_fp16": use_fp16,
                    "tile_size": params.get("tile_size", 512),
                    "tile_overlap": params.get("tile_overlap", 32),
                    "target_scale": 2.0,
                    "target_resolution": video_target_resolution,
                    "sharpening": params.get("sharpening", 0),
                    "contrast": params.get("contrast", 1.0),
                    "saturation": params.get("saturation", 1.0),
                    "pre_downscale_height": pre_downscale_height,  # v2.8+
                    "is_video_frame": True
                }

                status_messages.append(f"🚀 {vid_name}: ASYNC PIPELINE + BATCHING MODE (3 parallel threads)")

                # Run new concurrent pipeline with async batching
                pipeline = ConcurrentPipeline(
                    video_path=video_path,
                    output_dir=str(frames_out),
                    model_name=model,  # Now passes model NAME for upscale_batch
                    vram_manager=vram_manager,
                    upscale_params=upscale_params,
                    detect_duplicates=skip_duplicate_frames,
                    frame_format=frame_format,
                    progress_callback=progress
                )

                success, result_path, pipeline_stats = pipeline.run()

                if not success:
                    status_messages.append(f"❌ {vid_name}: Pipeline failed - {result_path}")
                    continue

                # Report pipeline statistics
                elapsed = pipeline_stats["total_time"]
                status_messages.append(f"⏱️ {vid_name}: Pipeline completed in {elapsed:.2f}s ({pipeline_stats['fps']:.2f} fps)")
                status_messages.append(f"📊 {vid_name}: Total: {pipeline_stats['total_frames']} | Unique: {pipeline_stats['unique_frames']} | Duplicates: {pipeline_stats['duplicate_frames']} ({pipeline_stats['duplicate_percentage']:.1f}%)")

                # Create stats dict compatible with later code
                stats = {
                    "total_frames": pipeline_stats["total_frames"],
                    "unique_frames": pipeline_stats["unique_frames"],
                    "duplicates": pipeline_stats["duplicate_frames"],
                    "duplicate_percentage": pipeline_stats["duplicate_percentage"]
                }

                # Frames already saved by pipeline, skip to encoding
                frames_out_populated = True

            else:
                # ============================================================
                # SEQUENTIAL PARALLEL PROCESSING (Original System)
                # ============================================================

                # Reason for not using pipeline
                if not ENABLE_GPU_PIPELINE:
                    status_messages.append(f"ℹ️ {vid_name}: GPU pipeline disabled in config")
                elif total_frames < PIPELINE_MIN_FRAMES:
                    status_messages.append(f"ℹ️ {vid_name}: Video too short for GPU pipeline ({total_frames} < {PIPELINE_MIN_FRAMES} frames)")
                elif not enable_parallel or vram_manager is None:
                    status_messages.append(f"ℹ️ {vid_name}: Parallel processing not available")

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

                # ✅ VERIFICATION: Check plan consistency
                if len(frames_to_process) != stats['unique_frames']:
                    status_messages.append(f"⚠️ WARNING: frames_to_process ({len(frames_to_process)}) != unique_frames ({stats['unique_frames']}) - Plan may be incorrect!")
                elif skip_duplicate_frames and stats['duplicates'] > 0:
                    status_messages.append(f"✅ OPTIMIZATION CONFIRMED: Will upscale ONLY {len(frames_to_process)} unique frames, skip {stats['duplicates']} duplicates")

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
                    'pre_downscale_height': pre_downscale_height,  # v2.8+
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
                    # BATCH MODE with ASYNC PREFETCH - Load N+1 while GPU processes N
                    batch_size = vram_manager.max_concurrent_jobs
                    status_messages.append(f"🚀 {vid_name}: ASYNC BATCH MODE - {batch_size} frames/batch with prefetching")
                    
                    # Group frames into batches
                    batches = []
                    for i in range(0, len(frames_to_process), batch_size):
                        batch = frames_to_process[i:i + batch_size]
                        batches.append(batch)
                    
                    status_messages.append(f"📦 {vid_name}: Created {len(batches)} batches from {unique_frame_count} unique frames")
                    
                    # Function to load a batch of images
                    def load_batch(batch_paths):
                        """Load batch of images in background thread"""
                        loaded_frames = []
                        loaded_orig = []
                        loaded_paths = []
                        for frame_path in batch_paths:
                            try:
                                img = Image.open(frame_path)

                                # PRÉ-DOWNSCALE (v2.8+): Apply before upscaling if enabled
                                img = _apply_pre_downscale(img, pre_downscale_height)

                                loaded_frames.append(img.copy())
                                loaded_orig.append(img.copy())
                                loaded_paths.append(frame_path)
                                img.close()
                            except Exception as e:
                                print(f"⚠️ Error loading {Path(frame_path).name}: {e}")
                        return loaded_frames, loaded_orig, loaded_paths
                    
                    # Async prefetch executor
                    with ThreadPoolExecutor(max_workers=1) as prefetch_executor:
                        # Start loading first batch
                        next_batch_future = prefetch_executor.submit(load_batch, batches[0]) if batches else None
                        
                        for batch_idx, batch_paths in enumerate(batches):
                            if check_processing_state("stop"):
                                break
                            
                            while check_processing_state("paused"):
                                time.sleep(0.1)
                                if check_processing_state("stop"):
                                    break
                            
                            # Get preloaded batch (already loaded in background)
                            batch_frames, batch_orig, valid_paths = next_batch_future.result()
                            
                            # Start prefetching NEXT batch while GPU works on current
                            if batch_idx + 1 < len(batches):
                                next_batch_future = prefetch_executor.submit(load_batch, batches[batch_idx + 1])
                            
                            if not batch_frames:
                                continue
                            
                            print(f"🔄 Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_frames)} frames)...")
                            
                            # GPU: Upscale entire batch in single call
                            try:
                                batch_results = upscale_batch(
                                    batch_frames,
                                    model,
                                    video_settings['use_fp16'],
                                    video_settings['target_resolution']
                                )
                                
                                # IMMEDIATE SAVE: Save each frame right after upscaling
                                for i, (frame_path, result_img, orig_img) in enumerate(zip(valid_paths, batch_results, batch_orig)):
                                    normalized_input = os.path.normpath(os.path.abspath(frame_path))
                                    
                                    for frame_idx, frame_info in frame_output_mapping.items():
                                        frame_unique = os.path.normpath(os.path.abspath(frame_info["unique_frame"]))
                                        if frame_unique == normalized_input and not frame_info["is_duplicate"]:
                                            output_path = Path(frame_info["output_path"])
                                            
                                            # Save immediately
                                            save_frame_with_format(result_img, output_path.with_suffix(''), frame_format)
                                            
                                            # Track for duplicate copying
                                            upscaled_results[normalized_input] = output_path
                                            
                                            # Update UI (first 100 frames only)
                                            if frames_completed < 100:
                                                all_results.append(rgba_to_rgb_for_display(result_img))
                                                orig_resized = orig_img.resize(result_img.size, Image.Resampling.LANCZOS)
                                                state_module.frame_pairs.append((
                                                    rgba_to_rgb_for_display(orig_resized),
                                                    rgba_to_rgb_for_display(result_img)
                                                ))
                                                orig_resized.close()
                                            
                                            frames_completed += 1
                                            break
                                    
                                    # Close images to free memory
                                    result_img.close()
                                    orig_img.close()
                                
                                print(f"✅ Batch {batch_idx + 1}: {len(batch_results)} frames upscaled and SAVED")
                                
                            except Exception as e:
                                status_messages.append(f"❌ Batch {batch_idx + 1} failed: {e}")
                                print(f"❌ Batch error: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Close source images to free memory
                            for img in batch_frames:
                                try:
                                    img.close()
                                except:
                                    pass
                            
                            # Update progress
                            if progress:
                                progress(0.15 + (frames_completed / unique_frame_count) * 0.8,
                                        desc=f"{vid_name} - Batch {batch_idx + 1}/{len(batches)} ({frames_completed}/{unique_frame_count} frames saved)")
                    
                    if skip_duplicate_frames and stats['duplicates'] > 0:
                        status_messages.append(f"⚡ {vid_name}: {frames_completed} unique frames upscaled (saved {stats['duplicates']} duplicate upscales!)")
                    else:
                        status_messages.append(f"✅ {vid_name}: {frames_completed} frames upscaled and saved")
                    
                    # Cleanup GPU memory after all batches
                    if DEVICE == "cuda":
                        clear_gpu_memory()

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
                # ✅ CRITICAL FIX: Use 2-pass approach to avoid order dependency
                # PASS 1: Save ALL unique frames first
                # PASS 2: Copy files for all duplicates

                if progress:
                    progress(0.85, desc=f"{vid_name} - Saving unique frames...")

                saved_unique_frames = {}  # Track saved unique frames: {unique_frame_path: saved_output_path}

                # ============================================================
                # PASS 1: Save ALL unique frames from upscaled_results
                # For BATCH MODE: Frames already saved, upscaled_results contains output paths
                # For SEQUENTIAL MODE: upscaled_results contains (result_img, orig_img) tuples
                # ============================================================
                print(f"🔵 PHASE 3: Processing save phase. upscaled_results has {len(upscaled_results)} entries, total_frames={total_frames}")
                
                for frame_idx in range(total_frames):
                    if check_processing_state("stop"):
                        break

                    frame_info = frame_output_mapping[frame_idx]
                    is_duplicate = frame_info["is_duplicate"]

                    if not is_duplicate:
                        # This is a unique frame
                        unique_frame_raw = frame_info["unique_frame"]
                        unique_frame = os.path.normpath(os.path.abspath(unique_frame_raw))
                        output_path = Path(frame_info["output_path"])

                        if unique_frame not in upscaled_results:
                            print(f"DEBUG: Looking for normalized path '{unique_frame}'")
                            print(f"DEBUG: Available keys sample: {list(upscaled_results.keys())[:3]}...")
                            status_messages.append(f"⚠️ {vid_name}: Unique frame {frame_idx} missing from upscaled results")
                            continue

                        result_data = upscaled_results[unique_frame]
                        
                        # Check if it's already a Path (batch mode - already saved)
                        if isinstance(result_data, Path):
                            # Batch mode: frame already saved, just track for duplicates
                            saved_unique_frames[unique_frame] = result_data
                        else:
                            # Sequential mode: result_data is (result_img, orig_img) tuple
                            result_img, orig_img = result_data
                            
                            # Save with format
                            save_frame_with_format(result_img, output_path.with_suffix(''), frame_format)
                            saved_unique_frames[unique_frame] = output_path

                            # Update UI (first 100 frames)
                            if frame_idx < 100:
                                all_results.append(rgba_to_rgb_for_display(result_img))
                                orig_resized = orig_img.resize(result_img.size, Image.Resampling.LANCZOS)
                                state_module.frame_pairs.append((
                                    rgba_to_rgb_for_display(orig_resized),
                                    rgba_to_rgb_for_display(result_img)
                                ))
                                orig_resized.close()

                unique_saved = len(saved_unique_frames)

                # ============================================================
                # PASS 2: Copy files for ALL duplicates (now all sources exist!)
                # ============================================================
                if progress:
                    progress(0.90, desc=f"{vid_name} - Copying duplicate frames...")

                duplicates_copied = 0
                for frame_idx in range(total_frames):
                    if check_processing_state("stop"):
                        break

                    frame_info = frame_output_mapping[frame_idx]
                    is_duplicate = frame_info["is_duplicate"]

                    if is_duplicate:
                        # This is a duplicate - copy from saved unique frame
                        unique_frame_raw = frame_info["unique_frame"]
                        unique_frame = os.path.normpath(os.path.abspath(unique_frame_raw))
                        output_path = Path(frame_info["output_path"])

                        if unique_frame in saved_unique_frames:
                            source_path = saved_unique_frames[unique_frame]
                            try:
                                shutil.copy2(source_path, output_path)
                                duplicates_copied += 1
                            except Exception as e:
                                status_messages.append(f"⚠️ {vid_name}: Failed to copy duplicate frame {frame_idx}: {e}")
                        else:
                            status_messages.append(f"⚠️ {vid_name}: Unique frame not found for duplicate {frame_idx}")

                saved_frames_count = unique_saved + duplicates_copied

                # ============================================================
                # CRITICAL VALIDATION (Bug #8): Verify frame sequence integrity
                # ============================================================
                expected_frames = set(range(total_frames))
                actual_frames = set()

                # Scan output directory for saved frames
                for frame_file in sorted(Path(frames_out).glob("frame_*.png")):
                    try:
                        # Extract frame number from filename (frame_XXXXX.png)
                        frame_num = int(frame_file.stem.split('_')[1])
                        actual_frames.add(frame_num)
                    except (IndexError, ValueError):
                        status_messages.append(f"⚠️ {vid_name}: Invalid frame filename: {frame_file.name}")

                missing_frames = expected_frames - actual_frames
                extra_frames = actual_frames - expected_frames

                if missing_frames or extra_frames:
                    status_messages.append(f"❌ {vid_name}: CRITICAL - Frame sequence corrupted!")
                    if missing_frames:
                        missing_list = sorted(missing_frames)[:10]
                        status_messages.append(f"   Missing frames: {missing_list}" + (f"... ({len(missing_frames) - 10} more)" if len(missing_frames) > 10 else ""))
                    if extra_frames:
                        extra_list = sorted(extra_frames)[:10]
                        status_messages.append(f"   Extra frames: {extra_list}" + (f"... ({len(extra_frames) - 10} more)" if len(extra_frames) > 10 else ""))
                    status_messages.append(f"   Expected {len(expected_frames)} frames, found {len(actual_frames)} frames")
                else:
                    # Confirm all frames saved in correct sequence
                    if saved_frames_count < total_frames:
                        status_messages.append(f"⚠️ {vid_name}: Only saved {saved_frames_count}/{total_frames} frames (some missing)")
                    else:
                        status_messages.append(f"✅ {vid_name}: Frame sequence validated - all {total_frames} frames in correct order")
                        if skip_duplicate_frames and duplicates_copied > 0:
                            status_messages.append(f"💾 {vid_name}: Saved {total_frames} frames ({frames_completed} upscaled + {duplicates_copied} copied from duplicates)")
                            time_saved_pct = (duplicates_copied / total_frames * 100)
                            status_messages.append(f"⚡ {vid_name}: OPTIMIZATION SUCCESS - Skipped {duplicates_copied} upscales ({time_saved_pct:.1f}% time saved)")
                        else:
                            status_messages.append(f"💾 {vid_name}: Successfully saved all {total_frames} frames")

                # Cleanup upscaled results from memory
                # Batch mode: values are Path objects (already closed during save)
                # Sequential mode: values are (result_img, orig_img) tuples
                for result_data in upscaled_results.values():
                    if not isinstance(result_data, Path):
                        result_img, orig_img = result_data
                        try:
                            result_img.close()
                            orig_img.close()
                        except:
                            pass
                upscaled_results.clear()
                del upscaled_results

                # Clear GPU memory after processing all frames
                if DEVICE == "cuda":
                    clear_gpu_memory()

                # Auto-delete input frames if enabled (only if not deleting entire extraction folder)
                if auto_delete_input_frames and not auto_delete_extraction_folder:
                    try:
                        shutil.rmtree(frames_in)
                        status_messages.append(f"🗑️ {vid_name}: Cleaned up input frames")
                    except Exception as e:
                        print(f"⚠️ Failed to delete input frames: {e}")

        # Export video if requested
        if export_video and not check_processing_state("stop"):
            # Show encoding details
            encoding_start_time = time.time()
            if progress:
                progress(0.95, desc=f"🎬 Encoding {vid_name} with {codec_name} ({profile_name})...")

            status_messages.append(f"🎬 {vid_name}: Starting video encoding with {codec_name} ({profile_name})...")

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
                # Use file-specific custom name if available, otherwise global custom name
                if custom_name:
                    output_video_name = custom_name
                else:
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
                keep_audio,
                frame_format  # ✅ CRITICAL FIX: Pass frame_format to use correct file extension
            )

            encoding_time = time.time() - encoding_start_time

            if success:
                status_messages.append(f"✅ {vid_name}: Video exported to {video_output.parent.name}/{video_output.name} ({codec_name} - {profile_name})")
                status_messages.append(f"⏱️ {vid_name}: Encoding time: {encoding_time:.1f}s ({total_frames / encoding_time:.2f} fps)")
                download_files.append(str(video_output))

                # Auto-delete upscaled frames after successful encoding (only if not deleting entire extraction folder)
                if auto_delete_output_frames and not auto_delete_extraction_folder:
                    try:
                        shutil.rmtree(frames_out)
                        status_messages.append(f"🗑️ {vid_name}: Cleaned up upscaled frames")
                    except Exception as e:
                        status_messages.append(f"⚠️ {vid_name}: Failed to delete upscaled frames: {e}")
            else:
                status_messages.append(f"⚠️ {vid_name}: {result_msg}")

        # Auto-delete JSON file if enabled
        if auto_delete_frame_mapping:
            plan_file = vid_session / "parallel_processing_plan.json"
            if plan_file.exists():
                try:
                    os.remove(plan_file)
                    status_messages.append(f"🗑️ {vid_name}: Deleted parallel_processing_plan.json")
                except Exception as e:
                    print(f"⚠️ Failed to delete parallel_processing_plan.json: {e}")

        # Auto-delete extraction folder when video is exported
        if auto_delete_extraction_folder and organize_videos_folder and export_video:
            # Delete the complete extraction folder (vid_session)
            try:
                if vid_session.exists():
                    shutil.rmtree(vid_session)
                    status_messages.append(f"🗑️ {vid_name}: Deleted extraction folder")
            except Exception as e:
                status_messages.append(f"⚠️ {vid_name}: Failed to delete extraction folder: {e}")
            
            # Also clean up temp_video_processing parent folder if empty
            try:
                temp_parent = vid_session.parent
                if temp_parent.name == "temp_video_processing" and temp_parent.exists():
                    remaining = list(temp_parent.iterdir())
                    if len(remaining) == 0:
                        temp_parent.rmdir()
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

    # Clean up empty session folder if all content was moved/deleted
    # This removes the timestamp folder (e.g., 20260129_193320) if it's empty
    try:
        if session.exists():
            # Check if session folder is empty or only contains empty subfolders
            remaining_items = list(session.rglob('*'))
            # Filter out directories, only count actual files
            remaining_files = [item for item in remaining_items if item.is_file()]

            if len(remaining_files) == 0:
                # Session folder is empty or only has empty directories
                shutil.rmtree(session)
                status_messages.append(f"🗑️ Cleaned up empty session folder: {session.name}")
    except Exception as e:
        print(f"⚠️ Failed to clean up session folder: {e}")

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
