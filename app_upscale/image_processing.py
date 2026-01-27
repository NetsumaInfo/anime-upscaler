"""
Image Processing Pipeline for Anime Upscaler

This module handles the complete image upscaling pipeline including:
- Post-processing (sharpening, contrast, saturation)
- Resizing utilities
- Tile-based upscaling with weighted blending
- Multi-pass upscaling for high-scale factors
- Image format conversion and saving
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
from typing import Tuple, Optional

from .config import DEVICE, DEFAULT_UPSCALING_SETTINGS
from .models import load_model
from .gpu import get_model_dtype


# ============================================================================
# Post-Processing Functions
# ============================================================================

def apply_post_processing_gpu(
    img_tensor: torch.Tensor,
    sharpening: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    cuda_stream=None
) -> torch.Tensor:
    """
    Apply post-processing enhancements directly on GPU using PyTorch.

    This is MUCH faster than CPU PIL operations, especially for large images.
    Operates on tensor in CHW format (channels, height, width).

    Args:
        img_tensor: Tensor on GPU with shape (C, H, W) in [0, 1] range
        sharpening: Sharpening factor (0 = none, 0.5-1.0 = moderate, 1.5-2.0 = strong)
        contrast: Contrast multiplier (<1.0 = decrease, 1.0 = original, >1.0 = increase)
        saturation: Saturation multiplier (<1.0 = decrease, 1.0 = original, >1.0 = increase)
        cuda_stream: Optional CUDA stream for parallel execution

    Returns:
        Post-processed tensor on GPU, same shape and dtype as input
    """
    # Skip if all parameters are default (no processing needed)
    if sharpening == 0.0 and contrast == 1.0 and saturation == 1.0:
        return img_tensor

    # Work with the tensor directly (no CPU transfer)
    processed = img_tensor

    with torch.inference_mode():
        # Use provided stream if available (for parallel execution)
        if cuda_stream is not None:
            with torch.cuda.stream(cuda_stream):
                processed = _apply_post_processing_ops(processed, sharpening, contrast, saturation)
        else:
            processed = _apply_post_processing_ops(processed, sharpening, contrast, saturation)

    return processed


def _apply_post_processing_ops(
    img_tensor: torch.Tensor,
    sharpening: float,
    contrast: float,
    saturation: float
) -> torch.Tensor:
    """
    Internal function to apply post-processing operations.
    Separated for cleaner stream handling.
    """
    processed = img_tensor

    # Sharpening via unsharp mask (GPU convolution)
    if sharpening > 0:
        # Unsharp mask kernel (emphasizes edges)
        # Center value = 9 + sharpening_strength, edges = -1
        strength = sharpening
        kernel_center = 9.0 + strength * 8.0
        kernel = torch.tensor([
            [-strength, -strength, -strength],
            [-strength, kernel_center, -strength],
            [-strength, -strength, -strength]
        ], dtype=processed.dtype, device=processed.device)

        # Reshape kernel for conv2d: (out_channels, in_channels, H, W)
        # Apply same kernel to each RGB channel
        kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

        # Add batch dimension if needed (conv2d expects NCHW)
        if processed.dim() == 3:
            processed = processed.unsqueeze(0)  # CHW -> NCHW
            remove_batch = True
        else:
            remove_batch = False

        # Apply convolution with padding to preserve size
        processed = F.conv2d(processed, kernel, padding=1, groups=3)

        # Clamp to [0, 1] range after sharpening
        processed = torch.clamp(processed, 0.0, 1.0)

        if remove_batch:
            processed = processed.squeeze(0)  # NCHW -> CHW

    # Contrast adjustment (around midpoint 0.5)
    if contrast != 1.0:
        # Add batch dim if needed for easier calculation
        if processed.dim() == 3:
            processed = processed.unsqueeze(0)
            remove_batch = True
        else:
            remove_batch = False

        # Calculate per-image mean (gray level midpoint)
        mean = processed.mean(dim=[1, 2, 3], keepdim=True)

        # Apply contrast: new = (old - mean) * contrast + mean
        processed = (processed - mean) * contrast + mean

        # Clamp to valid range
        processed = torch.clamp(processed, 0.0, 1.0)

        if remove_batch:
            processed = processed.squeeze(0)

    # Saturation adjustment (convert to grayscale, then blend)
    if saturation != 1.0:
        # Add batch dim if needed
        if processed.dim() == 3:
            processed = processed.unsqueeze(0)
            remove_batch = True
        else:
            remove_batch = False

        # Convert to grayscale using standard luminance weights (ITU-R BT.601)
        # L = 0.299*R + 0.587*G + 0.114*B
        gray_weights = torch.tensor([0.299, 0.587, 0.114], dtype=processed.dtype, device=processed.device)
        gray_weights = gray_weights.view(1, 3, 1, 1)  # Shape for broadcasting

        # Compute grayscale (weighted average across RGB channels)
        gray = (processed * gray_weights).sum(dim=1, keepdim=True)  # (N, 1, H, W)

        # Blend: result = gray * (1 - saturation) + color * saturation
        processed = gray * (1 - saturation) + processed * saturation

        # Clamp to valid range
        processed = torch.clamp(processed, 0.0, 1.0)

        if remove_batch:
            processed = processed.squeeze(0)

    return processed


def apply_post_processing(
    img: Image.Image,
    sharpening: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0
) -> Image.Image:
    """
    Apply post-processing enhancements to an image.

    Args:
        img: PIL Image to process
        sharpening: Sharpening factor (0 = none, 0.5-1.0 = moderate, 1.5-2.0 = strong)
        contrast: Contrast multiplier (<1.0 = decrease, 1.0 = original, >1.0 = increase)
        saturation: Saturation multiplier (<1.0 = decrease, 1.0 = original, >1.0 = increase)

    Returns:
        Processed PIL Image
    """
    # Sharpening
    if sharpening > 0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.0 + sharpening)

    # Contrast
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

    # Saturation
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(saturation)

    return img


# ============================================================================
# Image Resizing Utilities
# ============================================================================

def resize_to_1080p(img: Image.Image) -> Image.Image:
    """
    Resize image to 1080p max height while preserving aspect ratio.

    Args:
        img: PIL Image

    Returns:
        Resized image (or original if height <= 1080px)
    """
    width, height = img.size

    # Only resize if height exceeds 1080px
    if height > 1080:
        # Calculate new dimensions while preserving aspect ratio
        new_height = 1080
        new_width = int(width * (1080 / height))

        # Use LANCZOS for high-quality downscaling
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img


def calculate_upscale_passes(
    original_height: int,
    target_height: int,
    scale: int = 2
) -> int:
    """
    Calculate the number of upscaling passes needed to reach or exceed target resolution.

    Args:
        original_height: Source image height
        target_height: Desired target height
        scale: Model upscale factor (2x, 4x, etc.)

    Returns:
        Number of passes to perform (minimum 1)

    Examples (2x model):
        480p → 1080p: 480 * 2 = 960 (< 1080), 960 * 2 = 1920 (> 1080) → 2 passes
        1080p → 1080p: 1080 * 2 = 2160 (> 1080) → 1 pass
        720p → 4K (2160p): 720 * 2 = 1440 (< 2160), 1440 * 2 = 2880 (> 2160) → 2 passes

    Examples (4x model):
        480p → 1080p: 480 * 4 = 1920 (> 1080) → 1 pass
        720p → 4K (2160p): 720 * 4 = 2880 (> 2160) → 1 pass
    """
    if target_height == 0:
        return 1  # Auto mode: 1 pass only

    current_height = original_height
    passes = 0

    # Calculate how many passes to exceed target
    while current_height < target_height:
        current_height *= scale
        passes += 1

    # If already above target, at least 1 pass
    if passes == 0:
        passes = 1

    return passes


def resize_to_target_resolution(
    img: Image.Image,
    target_height: int,
    original_aspect_ratio: float
) -> Image.Image:
    """
    Resize image to target height while preserving aspect ratio.

    Args:
        img: Image to resize
        target_height: Target height (0 = no resize)
        original_aspect_ratio: Original width/height ratio

    Returns:
        Resized image with LANCZOS

    Examples:
        Image 3840×2160 (16:9) → target 1080p → 1920×1080 (preserves 16:9)
        Image 2880×2160 (4:3) → target 1080p → 1440×1080 (preserves 4:3)
    """
    if target_height == 0:
        return img  # Auto mode: no resize

    current_width, current_height = img.size

    if current_height == target_height:
        return img  # Already at target resolution

    # Calculate proportional width to preserve aspect ratio
    target_width = int(target_height * original_aspect_ratio)

    # LANCZOS for high quality (downscale OR upscale)
    return img.resize((target_width, target_height), Image.Resampling.LANCZOS)


def resize_to_target_scale(
    img: Image.Image,
    target_scale: float,
    original_size: Tuple[int, int]
) -> Image.Image:
    """
    Resize image to target scale factor.

    Args:
        img: Upscaled image (may have been upscaled multiple times)
        target_scale: Desired final scale factor (2.0, 3.0, 4.0, etc.)
        original_size: (width, height) of original image

    Returns:
        Resized image with LANCZOS

    Examples:
        Original 1000×1000, upscaled 2x → 2000×2000, target_scale=2.0 → 2000×2000 (no resize)
        Original 1000×1000, upscaled 4x → 4000×4000, target_scale=3.0 → 3000×3000 (downscale)
    """
    orig_width, orig_height = original_size
    target_width = int(orig_width * target_scale)
    target_height = int(orig_height * target_scale)

    current_width, current_height = img.size

    if current_width == target_width and current_height == target_height:
        return img  # Already at target size

    return img.resize((target_width, target_height), Image.Resampling.LANCZOS)


# ============================================================================
# Dithering and Quantization
# ============================================================================

def apply_dithering(img_float: np.ndarray, enable: bool = True) -> np.ndarray:
    """
    Apply dithering to reduce banding artifacts during float->uint8 conversion.

    Banding occurs when converting high-precision float (millions of values) to 8-bit (256 values).
    Dithering adds controlled noise that makes the quantization error less visible.

    Uses triangular dithering (better perceptual quality than uniform noise).

    Args:
        img_float: Image array in float [0, 1] range, shape (H, W, 3)
        enable: Whether to apply dithering

    Returns:
        Dithered image in uint8 [0, 255] range
    """
    # SAFETY: Check for NaN or Inf values first (can happen with some models)
    if not np.isfinite(img_float).all():
        print("⚠️ Warning: NaN/Inf detected in image, cleaning before conversion")
        img_float = np.nan_to_num(img_float, nan=0.0, posinf=1.0, neginf=0.0)

    # Ensure values are in [0, 1] range
    img_float = np.clip(img_float, 0.0, 1.0)

    # Scale to [0, 255]
    img_scaled = img_float * 255.0

    # Apply dithering only if enabled
    if enable:
        # Add triangular dithering noise
        # Triangular distribution has better perceptual quality than uniform
        # noise1 + noise2 creates triangular distribution (mean=0, smoother than single uniform)
        noise1 = np.random.uniform(-0.5, 0.5, img_scaled.shape)
        noise2 = np.random.uniform(-0.5, 0.5, img_scaled.shape)
        dither_noise = noise1 + noise2

        # Add dither and round
        img_dithered = img_scaled + dither_noise
    else:
        # No dithering: just round normally
        img_dithered = img_scaled

    # Clip and convert to uint8 (with extra safety check)
    img_dithered = np.clip(img_dithered, 0, 255)

    # Final safety: ensure no NaN before cast
    if not np.isfinite(img_dithered).all():
        print("⚠️ Warning: NaN/Inf after processing, forcing cleanup")
        img_dithered = np.nan_to_num(img_dithered, nan=0.0, posinf=255.0, neginf=0.0)

    img_dithered = img_dithered.astype(np.uint8)

    return img_dithered


# ============================================================================
# Tile-Based Upscaling with Weighted Blending
# ============================================================================

def create_gaussian_weight_map(
    tile_h: int,
    tile_w: int,
    overlap: int
) -> np.ndarray:
    """
    Create linear weight map for smooth tile blending.

    Uses linear feathering in overlap regions to eliminate visible tile boundaries.
    CRITICAL: This creates smooth transitions at tile edges so they blend seamlessly.

    Args:
        tile_h: Tile height (in upscaled coordinates)
        tile_w: Tile width (in upscaled coordinates)
        overlap: Overlap size in pixels (in upscaled coordinates)

    Returns:
        Weight map array (tile_h, tile_w, 3) with linear falloff in overlap regions
    """
    # Start with all 1.0 (full weight in center)
    weight = np.ones((tile_h, tile_w), dtype=np.float32)

    if overlap > 0 and overlap <= min(tile_h, tile_w):
        # Create linear ramp from 0 to 1
        ramp = np.linspace(0, 1, overlap, dtype=np.float32)

        # Left edge: fade in from 0 to 1
        weight[:, :overlap] = ramp[np.newaxis, :]

        # Right edge: fade out from 1 to 0
        weight[:, -overlap:] = ramp[np.newaxis, ::-1]

        # Top edge: multiply by fade in
        weight[:overlap, :] *= ramp[:, np.newaxis]

        # Bottom edge: multiply by fade out
        weight[-overlap:, :] *= ramp[::-1, np.newaxis]

    # Expand to 3 channels (RGB)
    weight = np.stack([weight, weight, weight], axis=2)

    return weight


def _upscale_single_pass(
    img: Image.Image,
    model,
    scale: int,
    tile_size: int,
    tile_overlap: int,
    use_fp16: bool,
    cuda_stream=None,
    sharpening: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0
) -> Image.Image:
    """
    Perform a SINGLE upscaling pass with optional CUDA stream.

    Internal function used by upscale_image() for multi-pass processing.

    Args:
        img: PIL Image in RGB mode
        model: Loaded Spandrel model
        scale: Model scale factor (always 2 for 2x models)
        tile_size: Tile size in pixels
        tile_overlap: Tile overlap in pixels
        use_fp16: Use FP16 for acceleration
        cuda_stream: Optional torch.cuda.Stream for parallel GPU execution.
                     If None, uses default stream (backward compatible).

    Returns:
        Upscaled image (scale×) in RGB mode
    """
    # Get numpy array - ensure no color space conversion
    img_np = np.array(img, dtype=np.float32) / 255.0

    # Determine target dtype based on FP16 setting and model dtype
    # OPTIMIZATION: Get model dtype once to avoid repeated calls
    model_dtype = get_model_dtype(model)

    # Create tensor with correct dtype from the start (no conversion needed)
    if use_fp16 is None or model_dtype == torch.float16:
        target_dtype = model_dtype
    else:
        target_dtype = torch.float16 if (DEVICE == "cuda" and use_fp16) else torch.float32

    h, w = img_np.shape[:2]

    # For small images, process directly (no tiling needed)
    if h * w <= tile_size * tile_size:
        # CRITICAL FIX: Wrap ALL GPU operations in stream context to prevent race conditions
        # in parallel mode. Previously only model() was in context, causing artifacts.
        if cuda_stream is not None:
            with torch.cuda.stream(cuda_stream):
                # Transfer to GPU on this stream
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(
                    dtype=target_dtype, device=DEVICE
                )
                with torch.inference_mode():
                    output = model(img_tensor)
                
                # GPU post-processing (if enabled)
                from .config import ENABLE_GPU_POST_PROCESSING
                if ENABLE_GPU_POST_PROCESSING:
                    output_chw = output.squeeze(0)
                    output_chw = apply_post_processing_gpu(output_chw, sharpening, contrast, saturation, cuda_stream)
                    output = output_chw.unsqueeze(0)
                
                # Synchronize before CPU transfer
                cuda_stream.synchronize()
                output = output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        else:
            # Default stream (CPU or non-parallel CUDA)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(
                dtype=target_dtype, device=DEVICE
            )
            with torch.inference_mode():
                output = model(img_tensor)
            
            from .config import ENABLE_GPU_POST_PROCESSING
            if ENABLE_GPU_POST_PROCESSING and DEVICE == "cuda":
                output_chw = output.squeeze(0)
                output_chw = apply_post_processing_gpu(output_chw, sharpening, contrast, saturation, None)
                output = output_chw.unsqueeze(0)
            
            output = output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()

        # CRITICAL: Clean NaN/Inf IMMEDIATELY after model output
        if not np.isfinite(output).all():
            print(f"⚠️ Warning: NaN/Inf in model output, cleaning (model may be unstable)")
            output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)

        # Clip to [0,1] BEFORE scaling
        output = np.clip(output, 0.0, 1.0)

        # Convert float [0,1] to uint8 [0,255]
        output_uint8 = (output * 255.0).round().astype(np.uint8)
        result_img = Image.fromarray(output_uint8, mode='RGB')

        # Clean up GPU memory
        del img_tensor, output
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return result_img

    # Tile-based processing for large images
    result = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    weight = np.zeros((h * scale, w * scale, 3), dtype=np.float32)

    # OPTIMIZATION: Pre-compute weight maps for standard tiles (cache them)
    weight_cache = {}
    overlap_scaled = tile_overlap * scale

    # Process tiles with memory optimization and weighted blending
    for y in range(0, h, tile_size - tile_overlap):
        for x in range(0, w, tile_size - tile_overlap):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = img_np[y:y_end, x:x_end]

            # CRITICAL FIX: Wrap ALL tile GPU operations in stream context
            if cuda_stream is not None:
                with torch.cuda.stream(cuda_stream):
                    tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(
                        dtype=target_dtype, device=DEVICE
                    )
                    with torch.inference_mode():
                        tile_output = model(tile_tensor)
                    cuda_stream.synchronize()
                    tile_output = tile_output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
            else:
                tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(
                    dtype=target_dtype, device=DEVICE
                )
                with torch.inference_mode():
                    tile_output = model(tile_tensor)
                tile_output = tile_output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()

            # CRITICAL: Clean NaN/Inf IMMEDIATELY after model output
            if not np.isfinite(tile_output).all():
                print(f"⚠️ Warning: NaN/Inf in tile output, cleaning (model may be unstable)")
                tile_output = np.nan_to_num(tile_output, nan=0.0, posinf=1.0, neginf=0.0)

            # Clip tile output to [0,1] IMMEDIATELY
            tile_output = np.clip(tile_output, 0.0, 1.0)

            y_out, x_out = y * scale, x * scale
            th, tw = tile_output.shape[:2]

            # OPTIMIZATION: Use cached weight map if available
            weight_key = (th, tw)
            if weight_key not in weight_cache:
                weight_cache[weight_key] = create_gaussian_weight_map(th, tw, overlap_scaled)
            tile_weight = weight_cache[weight_key]

            # Apply weighted blending
            result[y_out:y_out+th, x_out:x_out+tw] += tile_output * tile_weight
            weight[y_out:y_out+th, x_out:x_out+tw] += tile_weight

            # Clean up tile tensors to free VRAM (keep tile_weight in cache)
            del tile_tensor, tile_output

    # Clear CUDA cache after all tiles processed
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Normalize by weight
    result = result / np.maximum(weight, 1e-8)

    # FINAL clipping: ensure [0,1] range before conversion
    result = np.clip(result, 0.0, 1.0)

    # OPTIMIZATION: Apply GPU post-processing on assembled result (Phase 2.2)
    from .config import ENABLE_GPU_POST_PROCESSING
    if ENABLE_GPU_POST_PROCESSING and DEVICE == "cuda" and (sharpening != 0.0 or contrast != 1.0 or saturation != 1.0):
        # Convert assembled result to GPU tensor
        result_tensor = torch.from_numpy(result).permute(2, 0, 1).to(dtype=torch.float32, device=DEVICE)

        # Apply post-processing on GPU
        with torch.inference_mode():
            if cuda_stream is not None:
                with torch.cuda.stream(cuda_stream):
                    result_tensor = apply_post_processing_gpu(result_tensor, sharpening, contrast, saturation, cuda_stream)
            else:
                result_tensor = apply_post_processing_gpu(result_tensor, sharpening, contrast, saturation, None)

        # CRITICAL: Synchronize CUDA stream BEFORE .cpu() transfer
        # Without this, we copy data BEFORE GPU computation finishes → corruption/artifacts
        if cuda_stream is not None:
            cuda_stream.synchronize()

        # Transfer back to CPU
        result = result_tensor.permute(1, 2, 0).float().cpu().numpy()
        result = np.clip(result, 0.0, 1.0)
        del result_tensor

    # Convert float [0,1] to uint8 [0,255]
    result_uint8 = (result * 255.0).round().astype(np.uint8)
    result_img = Image.fromarray(result_uint8, mode='RGB')

    return result_img


# ============================================================================
# Main Upscaling Function
# ============================================================================

def upscale_image(
    img: Image.Image,
    model_name: str,
    preserve_alpha: bool = False,
    output_format: str = "PNG",
    jpeg_quality: int = 95,
    use_fp16: bool = True,
    downscale_to_1080p: bool = False,
    tile_size: Optional[int] = None,
    tile_overlap: Optional[int] = None,
    sharpening: Optional[float] = None,
    contrast: Optional[float] = None,
    saturation: Optional[float] = None,
    target_scale: float = 2.0,
    target_resolution: int = 0,
    is_video_frame: bool = False,
    cuda_stream=None
) -> Tuple[Image.Image, Image.Image]:
    """
    Upscale image with tile-based processing and multi-pass support.

    Supports AUTO settings (model decides quality) or custom parameters.

    Args:
        img: Input PIL Image
        model_name: Model display name
        preserve_alpha: Keep alpha channel if present
        output_format: Output format ("PNG", "JPEG", "WebP")
        jpeg_quality: Quality for JPEG/WebP (80-100)
        use_fp16: Use FP16 precision (50% VRAM reduction on CUDA)
        downscale_to_1080p: Legacy parameter (unused)
        tile_size: Tile size in pixels (None = AUTO based on model scale)
        tile_overlap: Tile overlap in pixels (None = AUTO 32px)
        sharpening: Post-processing sharpening (0-2.0)
        contrast: Post-processing contrast (0.8-1.2)
        saturation: Post-processing saturation (0.8-1.2)
        target_scale: Target scale factor for images (2.0, 4.0, 8.0, 16.0)
        target_resolution: Target height for videos (0 = auto)
        is_video_frame: Whether this is a video frame
        cuda_stream: Optional torch.cuda.Stream for parallel GPU execution.
                     Propagated to _upscale_single_pass for all inference operations.

    Returns:
        Tuple of (upscaled_image, original_image)
    """
    # Load model first to get scale factor
    model, actual_fp16, scale = load_model(model_name, use_fp16)
    # Update use_fp16 to reflect actual model dtype
    use_fp16 = actual_fp16

    # Use AUTO settings if not provided
    # OPTIMIZATION: Adjust tile size automatically for x8 and x16 models
    if tile_size is None:
        base_tile_size = DEFAULT_UPSCALING_SETTINGS["tile_size"]
        if scale >= 16:
            # x16 models: reduce tile size significantly to avoid OOM
            tile_size = max(128, base_tile_size // 4)
            print(f"🔧 Auto tile size for x{scale} model: {tile_size}px (optimized for high-scale)")
        elif scale >= 8:
            # x8 models: reduce tile size moderately
            tile_size = max(192, base_tile_size // 2)
            print(f"🔧 Auto tile size for x{scale} model: {tile_size}px (optimized for high-scale)")
        else:
            # x1, x2, x4 models: use default
            tile_size = base_tile_size

    tile_overlap = tile_overlap if tile_overlap is not None else DEFAULT_UPSCALING_SETTINGS["tile_overlap"]
    sharpening = sharpening if sharpening is not None else DEFAULT_UPSCALING_SETTINGS["sharpening"]
    contrast = contrast if contrast is not None else DEFAULT_UPSCALING_SETTINGS["contrast"]
    saturation = saturation if saturation is not None else DEFAULT_UPSCALING_SETTINGS["saturation"]

    # Store original dimensions and aspect ratio (BEFORE any transformation)
    original_width, original_height = img.size
    original_aspect_ratio = original_width / original_height

    # Calculate number of passes needed (based on model scale)
    if scale == 1:
        # x1 models: no upscaling, just processing (e.g., NES_Composite_To_RGB)
        num_passes = 1
        if target_scale != 1.0:
            print(f"⚠️ x1 model detected: target scale {target_scale}x ignored (x1 models don't upscale)")
    elif is_video_frame and target_resolution > 0:
        # For videos: calculate passes to reach target resolution
        num_passes = calculate_upscale_passes(original_height, target_resolution, scale)
    elif not is_video_frame and target_scale > scale:
        # For images: calculate passes for target scale
        # Example: scale=2x, target=4x → 2 passes (2x → 2x)
        # Example: scale=4x, target=8x → 2 passes (4x → 4x then downscale)
        num_passes = int(np.ceil(np.log(target_scale) / np.log(scale)))
    else:
        # Default: 1 pass (current behavior)
        num_passes = 1

    # CRITICAL: Remove ICC profile to prevent color shifts (like chaiNNer does)
    if 'icc_profile' in img.info:
        img_data = img.tobytes()
        img = Image.frombytes(img.mode, img.size, img_data)

    # Store original alpha channel if present
    original_alpha = None
    if img.mode in ('RGBA', 'LA') and preserve_alpha:
        original_alpha = img.getchannel('A')

    # --- MULTI-PASS UPSCALING LOOP ---
    current_img = img

    for pass_num in range(num_passes):
        # Convert to RGB if needed
        # CRITICAL FIX: Use PIL's convert() for proper alpha blending
        # Direct numpy slicing [:, :, :3] can cause line artifacts
        if current_img.mode != 'RGB':
            current_img = current_img.convert('RGB')

        # ONE upscaling pass
        current_img = _upscale_single_pass(
            current_img,
            model,
            scale,
            tile_size,
            tile_overlap,
            use_fp16,
            cuda_stream,
            sharpening,
            contrast,
            saturation
        )

    result_img = current_img

    # Apply post-processing (ONCE after all passes)
    # OPTIMIZATION: Skip PIL post-processing if GPU version was already applied (Phase 2.2)
    from .config import ENABLE_GPU_POST_PROCESSING
    if not (ENABLE_GPU_POST_PROCESSING and DEVICE == "cuda"):
        # Use CPU PIL post-processing (fallback or when GPU disabled)
        result_img = apply_post_processing(result_img, sharpening, contrast, saturation)

    # Smart resizing based on type
    if scale == 1:
        # x1 models: no resize needed, output = input size
        pass
    elif is_video_frame and target_resolution > 0:
        # For videos: resize to target resolution with preserved aspect ratio
        result_img = resize_to_target_resolution(result_img, target_resolution, original_aspect_ratio)
    elif not is_video_frame and target_scale != 2.0:
        # For images: resize to target scale
        result_img = resize_to_target_scale(result_img, target_scale, (original_width, original_height))

    # Apply preserved alpha channel
    if original_alpha is not None:
        upscaled_alpha = original_alpha.resize(
            (result_img.width, result_img.height),
            Image.Resampling.LANCZOS
        )
        result_img = result_img.convert('RGBA')
        result_img.putalpha(upscaled_alpha)

    return result_img, img


# ============================================================================
# Image Saving with Format Support
# ============================================================================

def save_image_with_format(
    img: Image.Image,
    path: Path,
    output_format: str,
    jpeg_quality: int = 95
) -> None:
    """
    Save image with specified format (without ICC profile to preserve exact colors).

    Args:
        img: PIL Image to save
        path: Output path (extension will be adjusted based on format)
        output_format: Output format ("PNG", "JPEG", "WebP")
        jpeg_quality: Quality for JPEG/WebP (80-100)
    """
    if output_format == "JPEG":
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
            quality=jpeg_quality,
            optimize=True,
            icc_profile=None
        )

    elif output_format == "WebP":
        img.save(
            path.with_suffix('.webp'),
            'WebP',
            quality=jpeg_quality,
            method=6,
            icc_profile=None
        )

    else:  # PNG (default)
        img.save(
            path.with_suffix('.png'),
            'PNG',
            optimize=True,
            icc_profile=None
        )
