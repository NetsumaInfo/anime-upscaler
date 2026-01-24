"""
File Type Detection and Utilities for Anime Upscaler

This module handles file type detection (image vs video) and file separation.
"""

from pathlib import Path
from typing import List, Tuple, Optional
from .config import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


# ============================================================================
# File Type Detection
# ============================================================================

def detect_type(path: str) -> Optional[str]:
    """
    Detect if a file is an image or video based on extension.

    Args:
        path: File path string

    Returns:
        "image" if file is an image,
        "video" if file is a video,
        None if unknown/unsupported type
    """
    ext = Path(path).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"

    return None


def separate_files_by_type(files) -> Tuple[List[str], List[str]]:
    """
    Separate uploaded files into images and videos.

    Handles both single files and lists of files from Gradio uploads.

    Args:
        files: Single file or list of files (Gradio File objects or strings)

    Returns:
        Tuple of (images_list, videos_list) with file paths
    """
    images = []
    videos = []

    if not files:
        return images, videos

    # Normalize to list
    file_list = files if isinstance(files, list) else [files]

    for f in file_list:
        # Extract path from Gradio File object or use string directly
        path = f.name if hasattr(f, 'name') else f
        ftype = detect_type(path)

        if ftype == "image":
            images.append(path)
        elif ftype == "video":
            videos.append(path)
        # Ignore unknown types silently

    return images, videos


def is_image(path: str) -> bool:
    """
    Check if a file is an image.

    Args:
        path: File path string

    Returns:
        True if image, False otherwise
    """
    return detect_type(path) == "image"


def is_video(path: str) -> bool:
    """
    Check if a file is a video.

    Args:
        path: File path string

    Returns:
        True if video, False otherwise
    """
    return detect_type(path) == "video"


def get_file_extension(path: str) -> str:
    """
    Get the file extension (lowercase, with dot).

    Args:
        path: File path string

    Returns:
        File extension (e.g., ".png", ".mp4")
    """
    return Path(path).suffix.lower()


def get_file_stem(path: str) -> str:
    """
    Get the file name without extension.

    Args:
        path: File path string

    Returns:
        Filename stem (e.g., "video" from "video.mp4")
    """
    return Path(path).stem
