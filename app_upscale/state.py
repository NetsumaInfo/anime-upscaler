"""
State Management for Anime Upscaler

This module provides thread-safe state management for processing control and frame navigation.
"""

import threading
import numpy as np
from PIL import Image
from typing import List, Tuple

# ============================================================================
# Global State Variables
# ============================================================================

# Thread-safe lock for processing state
processing_state_lock = threading.Lock()

# Processing state dictionary (running, paused, stop flags)
processing_state = {
    "running": False,
    "paused": False,
    "stop": False
}

# Frame pairs for navigation UI: [(original_img, upscaled_img), ...]
frame_pairs: List[Tuple[Image.Image, Image.Image]] = []

# Current language for multilingual UI
current_language = "fr"  # Default, will be set from config

# ============================================================================
# Thread-Safe State Access Functions
# ============================================================================

def check_processing_state(key: str) -> bool:
    """
    Thread-safe read of processing state.

    Args:
        key: State key to check ("running", "paused", or "stop")

    Returns:
        Boolean value of the requested state
    """
    with processing_state_lock:
        return processing_state.get(key, False)


def update_processing_state(key: str, value: bool) -> None:
    """
    Thread-safe update of processing state.

    Args:
        key: State key to update ("running", "paused", or "stop")
        value: New boolean value for the state
    """
    with processing_state_lock:
        processing_state[key] = value


# ============================================================================
# Processing Control Functions
# ============================================================================

def stop_processing() -> str:
    """
    Stop processing and reset to initial state.

    Returns:
        Status message string
    """
    update_processing_state("stop", True)
    update_processing_state("running", False)
    update_processing_state("paused", False)
    return "⏹️ Stopped"


def pause_processing():
    """
    Toggle pause/resume state.

    Returns:
        Tuple of (button_update, status_message) for UI update

    Note: This function returns gradio updates, so it requires gradio to be imported
    in the calling context (ui.py).
    """
    import gradio as gr

    current_paused = check_processing_state("paused")
    update_processing_state("paused", not current_paused)

    if not current_paused:  # Now paused
        return gr.update(value="▶️ Resume"), "⏸️ Paused"
    else:  # Now resumed
        return gr.update(value="⏸️ Pause"), "▶️ Resumed"


# ============================================================================
# Image Conversion Utilities
# ============================================================================

def rgba_to_rgb_for_display(img: Image.Image) -> np.ndarray:
    """
    Convert RGBA image to RGB with checkerboard background for display.

    This is useful for displaying images with transparency in the UI,
    where a checkerboard pattern indicates transparent areas.

    Args:
        img: PIL Image (can be RGBA, RGB, or other mode)

    Returns:
        Numpy array in RGB format suitable for display
    """
    if img.mode == 'RGBA':
        # Create checkerboard background (16x16 pixel tiles, light gray and white)
        checker_size = 16  # Size of each checker square
        light_color = (200, 200, 200)  # Light gray
        dark_color = (170, 170, 170)   # Darker gray

        # Create checkerboard pattern
        width, height = img.size
        background = Image.new('RGB', img.size, light_color)

        # Draw dark squares
        from PIL import ImageDraw
        draw = ImageDraw.Draw(background)
        for y in range(0, height, checker_size):
            for x in range(0, width, checker_size):
                # Alternate pattern: dark square if (x//checker_size + y//checker_size) is odd
                if ((x // checker_size) + (y // checker_size)) % 2 == 1:
                    draw.rectangle(
                        [x, y, x + checker_size, y + checker_size],
                        fill=dark_color
                    )

        # Paste image on top with alpha blending
        background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        return np.array(background)

    elif img.mode == 'RGB':
        return np.array(img)

    else:
        # Convert to RGB for other modes (Grayscale, etc.)
        return np.array(img.convert('RGB'))


# ============================================================================
# Frame Storage Management
# ============================================================================

def clear_frame_pairs() -> None:
    """Clear the frame pairs list (useful for starting new batch)."""
    global frame_pairs
    frame_pairs = []


def add_frame_pair(original: Image.Image, upscaled: Image.Image) -> None:
    """
    Add a frame pair to the navigation list.

    Args:
        original: Original image before upscaling
        upscaled: Upscaled result image
    """
    global frame_pairs
    frame_pairs.append((original, upscaled))


def get_frame_pair(index: int) -> Tuple[Image.Image, Image.Image]:
    """
    Get a specific frame pair by index.

    Args:
        index: Frame pair index

    Returns:
        Tuple of (original, upscaled) images

    Raises:
        IndexError: If index is out of range
    """
    return frame_pairs[index]


def get_frame_pairs_count() -> int:
    """
    Get the total number of frame pairs.

    Returns:
        Number of stored frame pairs
    """
    return len(frame_pairs)
