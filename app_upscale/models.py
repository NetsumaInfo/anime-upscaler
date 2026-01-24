"""
Model Management for Anime Upscaler

This module handles AI model scanning, downloading, loading, caching, and VRAM management.
"""

import torch
import threading
import requests
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Dict, Optional, Any

from .config import MODELS_DIR, DEFAULT_MODELS, DEVICE
from .gpu import get_model_dtype


# ============================================================================
# Global Model Cache
# ============================================================================

# Cache for loaded models: {cache_key: {"model": model, "scale": int}}
loaded_models: Dict[str, Dict[str, Any]] = {}

# Available models configuration: {display_name: {"file": filename, "url": url}}
MODELS: Dict[str, Dict[str, str]] = {}

# Mapping from display names to filenames: {display_name: filename}
MODEL_DISPLAY_TO_FILE: Dict[str, str] = {}


# ============================================================================
# VRAM Manager for Parallel Processing
# ============================================================================

class VRAMManager:
    """
    Manages concurrent access to GPU VRAM to prevent OOM errors
    when processing multiple images in parallel.

    Uses a semaphore to limit the number of concurrent GPU operations.
    """

    def __init__(self, max_concurrent_jobs: int = 2):
        """
        Initialize VRAM manager.

        Args:
            max_concurrent_jobs: Maximum number of parallel GPU operations allowed
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.semaphore = threading.Semaphore(max_concurrent_jobs)
        self.lock = threading.Lock()

    def acquire(self) -> None:
        """
        Wait for a VRAM slot to become available.

        This blocks until a slot is free if all slots are currently occupied.
        """
        self.semaphore.acquire()

    def release(self) -> None:
        """Release a VRAM slot, allowing another operation to proceed."""
        self.semaphore.release()

    @staticmethod
    def auto_calculate_slots(total_vram_gb: float) -> int:
        """
        Calculate optimal number of parallel jobs based on available VRAM.

        OPTIMIZED FORMULA (v2.6.2):
        - More aggressive worker allocation based on real-world testing
        - Removes synchronization bottlenecks (torch.cuda.synchronize moved out of workers)
        - Allows VRAM to be fully utilized without artificial limits

        Args:
            total_vram_gb: Total VRAM in GB

        Returns:
            Number of recommended parallel jobs:
            - <5GB (4GB): 1 worker (sequential fallback)
            - 5-7GB (6GB): 3 workers (was 2)
            - 7-10GB (8GB): 5 workers (was 3)
            - 10-12GB: 6 workers (was 4)
            - ≥12GB: 8 workers (was 4)
        """
        if total_vram_gb < 5:
            return 1  # 4GB: Sequential only
        elif total_vram_gb < 7:
            return 3  # 6GB: 3 parallel jobs (50% increase)
        elif total_vram_gb < 10:
            return 5  # 8GB: 5 parallel jobs (67% increase)
        elif total_vram_gb < 12:
            return 6  # 10GB: 6 parallel jobs
        else:
            return 8  # 12GB+: 8 parallel jobs (100% increase)

    def update_max_jobs(self, new_max: int) -> None:
        """
        Update maximum concurrent jobs (recreates semaphore).

        Args:
            new_max: New maximum number of concurrent jobs
        """
        with self.lock:
            self.max_concurrent_jobs = new_max
            self.semaphore = threading.Semaphore(new_max)


# ============================================================================
# Model Discovery and Configuration
# ============================================================================

def scan_models() -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """
    Scan models directory and build model configuration with display names.

    Combines existing models in the models/ folder with default models
    that will be auto-downloaded.

    Returns:
        Tuple of (models_dict, display_to_file_mapping):
        - models_dict: {display_name: {"file": filename, "url": url}}
        - display_to_file_mapping: {display_name: filename}

    Note:
        Scale factor is now auto-detected by Spandrel when loading models,
        so it's no longer stored in the configuration.
    """
    models = {}
    display_to_file = {}

    # Scan for existing models in models folder
    if MODELS_DIR.exists():
        for model_file in MODELS_DIR.iterdir():
            if model_file.suffix.lower() in ['.pth', '.safetensors']:
                model_name = model_file.stem

                # Get display name from DEFAULT_MODELS if available
                display_name = DEFAULT_MODELS.get(model_file.name, {}).get(
                    "display_name", model_name
                )

                models[display_name] = {
                    "file": model_file.name,
                    "url": DEFAULT_MODELS.get(model_file.name, {}).get("url", None)
                }
                display_to_file[display_name] = model_file.name

    # Add default models if not found (they'll be auto-downloaded)
    for default_file, default_config in DEFAULT_MODELS.items():
        model_path = MODELS_DIR / default_file
        display_name = default_config.get("display_name", Path(default_file).stem)

        if not model_path.exists():
            if display_name not in models:
                models[display_name] = {
                    "file": default_file,
                    "url": default_config["url"]
                }
                display_to_file[display_name] = default_file

    return models, display_to_file


def initialize_models() -> Tuple[Dict, Dict]:
    """
    Initialize global MODELS and MODEL_DISPLAY_TO_FILE by scanning the models directory.

    Returns:
        Tuple of (MODELS, MODEL_DISPLAY_TO_FILE)
    """
    global MODELS, MODEL_DISPLAY_TO_FILE
    MODELS, MODEL_DISPLAY_TO_FILE = scan_models()
    return MODELS, MODEL_DISPLAY_TO_FILE


# ============================================================================
# Model Download
# ============================================================================

def download_model(model_name: str) -> Path:
    """
    Download model from configured URL if not present locally.

    Args:
        model_name: Display name of the model

    Returns:
        Path to the model file

    Raises:
        FileNotFoundError: If model file not found and no URL provided
    """
    config = MODELS[model_name]
    model_path = MODELS_DIR / config["file"]

    # If model exists, return immediately
    if model_path.exists():
        return model_path

    # If no URL is provided (manually added model), raise error
    if not config.get("url"):
        raise FileNotFoundError(
            f"Model file '{config['file']}' not found in {MODELS_DIR}"
        )

    print(f"📥 Downloading {model_name}...")
    url = config["url"]

    # Download with progress bar
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))

    with open(model_path, 'wb') as f, tqdm(
        desc=config["file"],
        total=total,
        unit='B',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(1024):
            f.write(data)
            pbar.update(len(data))

    return model_path


# ============================================================================
# Model Loading with Spandrel
# ============================================================================

def load_model(
    model_name: str,
    use_fp16: Optional[bool] = True
) -> Tuple[Any, bool, int]:
    """
    Load model using Spandrel with precision optimizations.

    Args:
        model_name: Display name of the model
        use_fp16:
            - True: Force FP16 (50% VRAM reduction on CUDA)
            - False: Force FP32 (maximum precision)
            - None: No conversion (PyTorch decides automatically)

    Returns:
        Tuple of (model, actual_fp16_enabled, scale):
        - model: Loaded Spandrel model instance
        - actual_fp16_enabled: True if model is using FP16
        - scale: Model upscale factor (1, 2, 4, 8, 16, etc.)

    Raises:
        ImportError: If Spandrel not installed
        FileNotFoundError: If model file not found and no download URL
    """
    global loaded_models

    # Create cache key based on model name and precision
    if use_fp16 is None:
        cache_key = f"{model_name}_none"
    elif use_fp16 and DEVICE == "cuda":
        cache_key = f"{model_name}_fp16"
    else:
        cache_key = f"{model_name}_fp32"

    # Return cached model if available
    if cache_key in loaded_models:
        model_data = loaded_models[cache_key]
        model = model_data["model"]
        scale = model_data["scale"]
        actual_fp16 = (get_model_dtype(model) == torch.float16)
        print(f"♻️ Using cached model: {model_name} ({'FP16' if actual_fp16 else 'FP32'})")
        return model, actual_fp16, scale

    # Import Spandrel
    try:
        from spandrel import ImageModelDescriptor, ModelLoader
    except ImportError:
        raise ImportError("Spandrel not installed. Run: pip install spandrel")

    # Download model if needed
    model_path = download_model(model_name)
    print(f"⏳ Loading {model_name}...")

    # Load model via Spandrel
    model_descriptor = ModelLoader().load_from_file(str(model_path))

    # Extract scale from Spandrel's auto-detection
    if isinstance(model_descriptor, ImageModelDescriptor):
        scale = model_descriptor.scale
        model = model_descriptor.model
    else:
        # Fallback for non-ImageModelDescriptor types
        scale = 2
        model = model_descriptor

    # Move model to device and set to eval mode
    model = model.to(DEVICE).eval()

    # Track actual FP16 status
    actual_fp16_enabled = False

    # Apply precision conversion if requested (None = skip conversion)
    if use_fp16 is None:
        print(f"ℹ️ Precision mode: None (PyTorch default, no conversion)")

    elif DEVICE == "cuda" and use_fp16 is True:
        try:
            # Check if this is a DAT model - they have FP16 compatibility issues
            model_arch = str(type(model).__module__)
            is_dat_model = 'DAT' in model_arch or 'dat' in model_arch.lower()

            if is_dat_model:
                # DAT models have internal dtype mismatches with FP16 - force FP32
                print(f"⚠️ DAT architecture detected - FP16 disabled (incompatible)")
                print(f"   Using FP32 for stability")
                model = model.float()
                actual_fp16_enabled = False
            else:
                # Convert model parameters to FP16
                model = model.half()

                # Convert buffers ONLY if they are floating point (skip integer indices)
                for buffer in model.buffers():
                    # Only convert float/double buffers, skip integer indices (long, int, etc.)
                    if buffer.dtype in [torch.float32, torch.float64]:
                        buffer.data = buffer.data.half()

                actual_fp16_enabled = True
                print(f"✅ FP16 enabled (VRAM usage reduced by ~50%)")

        except Exception as e:
            print(f"⚠️ FP16 conversion failed: {e}, using FP32")
            actual_fp16_enabled = False

    elif use_fp16 is False:
        # Explicitly ensure FP32 (though models load as FP32 by default)
        model = model.float()

        # Convert buffers ONLY if they are floating point (skip integer indices)
        for buffer in model.buffers():
            if buffer.dtype in [torch.float16]:
                buffer.data = buffer.data.float()

        print(f"ℹ️ FP32 enforced (maximum precision)")

    # Cache model with scale information
    loaded_models[cache_key] = {"model": model, "scale": scale}

    # Verify actual dtype after all transformations
    final_dtype = get_model_dtype(model)
    actual_fp16_enabled = (final_dtype == torch.float16)

    print(f"✅ {model_name} loaded on {DEVICE} ({'FP16' if actual_fp16_enabled else 'FP32'}) - {scale}x upscale")

    return model, actual_fp16_enabled, scale


# ============================================================================
# VRAM Utilities
# ============================================================================

def get_gpu_vram_gb() -> float:
    """
    Get total GPU VRAM in GB.

    Returns:
        Total VRAM in GB, or 0.0 if CPU mode, or 8.0 as fallback
    """
    if DEVICE == "cuda":
        try:
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return round(total_vram, 1)
        except Exception:
            return 8.0  # Default fallback
    return 0.0  # CPU mode


# Initialize models on module import
MODELS, MODEL_DISPLAY_TO_FILE = initialize_models()
