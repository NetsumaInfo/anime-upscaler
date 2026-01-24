"""
GPU and VRAM Management for Anime Upscaler

This module handles GPU memory optimization, VRAM monitoring, and precision mode (FP16/FP32).
"""

import torch
from .config import DEVICE


# ============================================================================
# GPU Memory Management
# ============================================================================

def clear_gpu_memory() -> None:
    """
    Clear GPU memory cache and synchronize (blocks until all GPU operations complete).

    This should ONLY be called AFTER all parallel workers have finished,
    NOT inside individual workers (would destroy parallelism).
    """
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def clear_gpu_memory_async() -> None:
    """
    Clear GPU memory cache WITHOUT synchronization (non-blocking).

    Safe to use inside parallel workers. Frees cached memory but allows
    GPU operations to continue in parallel without blocking other threads.
    """
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def get_gpu_memory_info() -> str:
    """
    Get current GPU memory usage information.

    Returns:
        String describing allocated and reserved VRAM, or "CPU mode" if CUDA unavailable
    """
    if DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert bytes to GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # Convert bytes to GB
        return f"VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "CPU mode"


# ============================================================================
# Model Precision Utilities
# ============================================================================

def get_model_dtype(model) -> torch.dtype:
    """
    Detect the actual dtype of model weights.

    Handles both regular models and torch.compile wrapped models.

    Args:
        model: PyTorch model or Spandrel model

    Returns:
        torch.dtype of the model's parameters (float16 or float32)
    """
    try:
        # Try to access _orig_mod for torch.compile wrapped models
        if hasattr(model, '_orig_mod'):
            param = next(model._orig_mod.parameters())
        else:
            param = next(model.parameters())
        return param.dtype
    except StopIteration:
        return torch.float32  # Fallback if no parameters found


# ============================================================================
# torch.compile Support Detection
# ============================================================================

def check_torch_compile_support() -> tuple[bool, str]:
    """
    Check if torch.compile is available and supported on this platform.

    torch.compile requires:
    - PyTorch 2.0+
    - Triton library (Linux only, not available on Windows)

    Returns:
        Tuple of (is_supported, message)
    """
    if not hasattr(torch, 'compile'):
        return False, "torch.compile not available (requires PyTorch 2.0+)"

    # Check for Triton availability (required for torch.compile)
    try:
        import platform
        if platform.system() == "Windows":
            return False, "torch.compile available but Triton not supported on Windows"

        # Try importing triton
        import triton
        return True, "torch.compile available with Triton support"
    except ImportError:
        return False, "torch.compile available but Triton library not installed"


def apply_torch_compile(model, mode: str = "default"):
    """
    Apply torch.compile to a model if supported.

    Provides 20-30% speedup on Linux with Triton. Gracefully falls back
    to original model if not supported.

    Args:
        model: Model to compile
        mode: Compilation mode ("default", "reduce-overhead", or "max-autotune")

    Returns:
        Compiled model, or original model if compilation not supported
    """
    is_supported, message = check_torch_compile_support()

    if not is_supported:
        print(f"ℹ️ {message} - using uncompiled model")
        return model

    try:
        # Enable error suppression to avoid cryptic error messages
        if hasattr(torch, '_dynamo'):
            torch._dynamo.config.suppress_errors = True

        compiled_model = torch.compile(model, mode=mode)
        print(f"✅ torch.compile applied successfully (mode: {mode})")
        return compiled_model

    except Exception as e:
        print(f"⚠️ torch.compile failed: {e} - using uncompiled model")
        return model


# ============================================================================
# VRAM Information
# ============================================================================

def get_gpu_vram_gb() -> float:
    """
    Get total GPU VRAM in GB.

    Returns:
        Total VRAM in GB, or 0.0 if CPU mode, or 8.0 as fallback if detection fails
    """
    if DEVICE == "cuda":
        try:
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return round(total_vram, 1)
        except Exception:
            return 8.0  # Default fallback for CUDA available but detection failed
    return 0.0  # CPU mode


def get_gpu_name() -> str:
    """
    Get GPU device name.

    Returns:
        GPU name string, or "CPU" if CUDA unavailable
    """
    if DEVICE == "cuda":
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "Unknown CUDA Device"
    return "CPU"


def get_cuda_version() -> str:
    """
    Get CUDA version.

    Returns:
        CUDA version string, or "N/A" if unavailable
    """
    if DEVICE == "cuda":
        try:
            return torch.version.cuda
        except Exception:
            return "N/A"
    return "N/A"


def print_gpu_diagnostics() -> None:
    """
    Print comprehensive GPU diagnostics at startup.

    Displays:
    - GPU name and VRAM
    - CUDA version
    - PyTorch version
    - torch.compile availability
    """
    print("=" * 60)
    print("🎮 GPU Diagnostics")
    print("=" * 60)

    if DEVICE == "cuda":
        print(f"✅ CUDA Available: Yes")
        print(f"📟 GPU: {get_gpu_name()}")
        print(f"💾 Total VRAM: {get_gpu_vram_gb()}GB")
        print(f"🔧 CUDA Version: {get_cuda_version()}")
    else:
        print(f"❌ CUDA Available: No (running on CPU)")

    print(f"🐍 PyTorch Version: {torch.__version__}")

    # Check torch.compile
    is_supported, message = check_torch_compile_support()
    if is_supported:
        print(f"⚡ torch.compile: ✅ {message}")
    else:
        print(f"⚡ torch.compile: ⚠️ {message}")

    print("=" * 60)
