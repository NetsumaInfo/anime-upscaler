"""
Configuration and Constants for Anime Upscaler

This module contains all global configuration, constants, translations, and default settings.
"""

import torch
import locale
from pathlib import Path

# ============================================================================
# Device Configuration
# ============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# File Extensions
# ============================================================================
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv'}

# ============================================================================
# Directory Paths
# ============================================================================
BASE_DIR = Path(__file__).parent.parent  # Go up from app_upscale/ to project root
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================================
# Performance Configuration (CPU/GPU Parallelization)
# ============================================================================
import multiprocessing

# ============================================================================
# Phase 1: Parallel Hash Computation for Duplicate Frame Detection
# ============================================================================
ENABLE_PARALLEL_HASH_DETECTION = True  # Set to False to use sequential (fallback)
HASH_WORKERS = min(multiprocessing.cpu_count(), 8)  # Max 8 workers for hash computation
# Expected gain: 5-7x faster duplicate detection (75s → 10-15s for 1000 frames)

# Default number of parallel workers for GPU upscaling (user-configurable in UI)
DEFAULT_PARALLEL_WORKERS = 3  # Default value, user can adjust based on VRAM

# ============================================================================
# Phase 2: CPU+GPU Hybrid Optimizations
# ============================================================================

# Phase 2.1: Async Image Pre-loading (Sequential Mode)
ENABLE_ASYNC_PRELOAD = False  # DISABLED - Testing to isolate corruption issue
# Load image N+1 while GPU upscales image N
# Expected gain: 5-10% faster sequential processing (eliminates I/O wait)

# Phase 2.2: GPU Post-Processing (Sharpening/Contrast/Saturation on GPU)
ENABLE_GPU_POST_PROCESSING = False  # TEMPORARILY DISABLED - Debugging visual artifacts
# Use PyTorch instead of PIL for post-processing
# Expected gain: 10-15% faster (post-processing on GPU instead of CPU)
# NOTE: Only applies when post-processing parameters are non-default

# Phase 2.3: Async Image Saving (Sequential Mode)
ENABLE_ASYNC_SAVE = False  # DISABLED - Causes memory corruption (PIL copy() issue)
# Save image N while GPU upscales image N+1
# Expected gain: 5-10% faster sequential processing (eliminates I/O wait)

# ============================================================================
# Combined Expected Performance (Phase 1 + Phase 2)
# ============================================================================
# Sequential images: 20-35% faster (Phase 2 optimizations)
# Parallel images: 10-15% faster (Phase 2.2 GPU post-processing only)
# Video duplicate detection: 5-7x faster (Phase 1)
# Overall video processing: 35-50% faster (Phase 1 + Phase 2 combined)

# ============================================================================
# Phase 3: GPU-First Optimized Pipeline (REPLACES v2.7 Concurrent Pipeline)
# ============================================================================

# Enable GPU-first pipeline for video processing (IMPROVED v2.8)
ENABLE_GPU_PIPELINE = False  # TEMPORARILY DISABLED - debugging async pipeline issue
# Minimum frames required to use pipeline (overhead not worth it for short videos)
PIPELINE_MIN_FRAMES = 20  # Lowered from 50 (GPU pipeline activates for videos ≥20 frames)

# GPU Pipeline Features:
# - FFmpeg CUDA/NVDEC extraction (3-5x faster than CPU)
# - GPU perceptual hashing with PyTorch tensors (instant detection)
# - Intelligent pre-loading (load N+1 while upscaling N)
# - Minimal CPU usage (only I/O and duplicate copying)
# - CUDA streams for true parallel GPU utilization

# Expected Performance Gains (v2.8 GPU Pipeline):
# - Extraction: 3-5x faster (GPU decode vs CPU)
# - Detection: 10-20x faster (GPU tensors vs CPU hashing)
# - Upscaling: Same as v2.6.2 (CUDA streams, parallel workers)
# - Pre-loading: Eliminates frame load bottleneck (zero idle time)
#
# Total Expected Speedup:
#   Without duplicates: 2-3x faster than v2.6.2 (GPU extraction + pre-load)
#   With duplicates (40%): 6-12x faster (GPU everything + skip duplicates)
#   Best case (70% duplicates): 15-25x faster
#
# Fallback: If GPU extraction unavailable, automatically uses CPU extraction
#   (still faster than v2.7 due to pre-loading and simpler architecture)

# ============================================================================
# Multilingual Translations (FR/EN)
# ============================================================================
TRANSLATIONS = {
    "fr": {
        "title": "🎨 Anime Upscaler",
        "subtitle": "",
        "upload_title": "📁 Fichiers",
        "upload_label": "Déposez les fichiers ici",
        "files_summary": "Fichiers",
        "no_files": "Aucun fichier",
        "ai_model_title": "🤖 Modèle",
        "select_model": "Modèle",
        "output_format_title": "⚙️ Image Format & Échelle",
        "image_scale_label": "Échelle",
        "image_scale_info": "",
        "pre_downscale_label": "Pré-réduction résolution",
        "pre_downscale_info": "⚠️ Réduit la résolution AVANT l'upscaling AI pour accélérer le traitement et économiser la VRAM (préserve le ratio)",
        "pre_downscale_original": "Original (aucune réduction)",
        "pre_downscale_480p": "480p",
        "pre_downscale_720p": "720p",
        "pre_downscale_1080p": "1080p",
        "final_output_format": "Format",
        "jpeg_quality": "Qualité",
        "video_frame_format": "Format frames",
        "video_frame_format_info": "",
        "upscaling_params_title": "🎨 Paramètres",
        "use_auto": "🤖 Auto (Recommandé)",
        "use_auto_info": "",
        "tile_settings": "Tuiles",
        "tile_size": "Taille tuile",
        "tile_size_info": "",
        "tile_overlap": "Chevauchement",
        "tile_overlap_info": "",
        "post_processing": "Post-Traitement",
        "sharpening": "Netteté",
        "sharpening_info": "",
        "contrast": "Contraste",
        "contrast_info": "",
        "saturation": "Saturation",
        "saturation_info": "",
        "advanced_title": "⚡ Avancé",
        "precision_mode": "Précision",
        "precision_info": "",
        "video_export_title": "🎬 Vidéo",
        "video_resolution_label": "Résolution",
        "video_resolution_info": "",
        "export_videos": "Exporter vidéos",
        "video_codec": "Codec",
        "codec_profile": "Profil",
        "fps_label": "FPS (0 = auto)",
        "preserve_transparency": "Transparence",
        "keep_audio": "🔊 Conserver audio",
        "keep_audio_info": "",
        "skip_duplicates": "⚡ Ignorer doublons",
        "skip_duplicates_info": "",
        "video_naming_title": "📝 Nommage",
        "video_naming_label": "Mode",
        "video_naming_info": "",
        "naming_same": "Nom original",
        "naming_suffix": "Suffixe",
        "naming_custom": "Personnalisé",
        "suffix_label": "Suffixe",
        "suffix_placeholder": "_upscale",
        "suffix_info": "",
        "custom_name_label": "Nom",
        "custom_name_placeholder": "ma_video",
        "custom_name_info": "",
        "auto_cleanup_title": "",
        "auto_cleanup_accordion": "Nettoyage Auto",
        "delete_input_frames": "Supprimer frames entrée",
        "delete_input_frames_info": "",
        "delete_output_frames": "Supprimer frames sortie",
        "delete_output_frames_info": "",
        "delete_extraction_folder": "🗑️ Supprimer TOUS les dossiers",
        "delete_extraction_folder_info": "Supprime le dossier d'extraction complet (frames entrée + sortie + tout)",
        "delete_mapping": "Supprimer JSON",
        "delete_mapping_info": "",
        "organize_videos": "📁 Dossier videos/ dédié",
        "organize_videos_info": "",
        "organize_images": "📁 Dossier images/ dédié",
        "organize_images_info": "",
        "test_btn": "🧪 Test",
        "test_frame_number": "🎞️ Frame à tester",
        "test_frame_number_info": "Choisissez le numéro de frame à tester pour les vidéos (la première frame est souvent peu intéressante)",
        "run_batch_btn": "▶️ Lancer",
        "test_status": "Test",
        "pause_btn": "⏸️ Pause",
        "stop_btn": "⏹️ Stop",
        "compare_tab": "⚖️ Comparer",
        "before_after": "Avant / Après",
        "frame_navigation": "Navigation",
        "gallery_tab": "🖼️ Galerie",
        "all_results": "Résultats",
        "status_label": "État",
        "download_info": "Fichiers",
        "output_folder": "📂 Dossier",
        "info_help_title": "",
        "enable_parallel": "⚡ Traitement parallèle",
        "enable_parallel_info": "",
        "batch_size": "📦 Batch size",
        "batch_size_info": ""
    },
    "en": {
        "title": "🎨 Anime Upscaler",
        "subtitle": "",
        "upload_title": "📁 Files",
        "upload_label": "Drop files here",
        "files_summary": "Files",
        "no_files": "No files",
        "ai_model_title": "🤖 Model",
        "select_model": "Model",
        "output_format_title": "⚙️ Image Format & Scale",
        "image_scale_label": "Scale",
        "image_scale_info": "",
        "pre_downscale_label": "Pre-Downscale Resolution",
        "pre_downscale_info": "⚠️ Reduces resolution BEFORE AI upscaling to speed up processing and save VRAM (preserves aspect ratio)",
        "pre_downscale_original": "Original (no reduction)",
        "pre_downscale_480p": "480p",
        "pre_downscale_720p": "720p",
        "pre_downscale_1080p": "1080p",
        "final_output_format": "Format",
        "jpeg_quality": "Quality",
        "video_frame_format": "Frame format",
        "video_frame_format_info": "",
        "upscaling_params_title": "🎨 Parameters",
        "use_auto": "🤖 Auto (Recommended)",
        "use_auto_info": "",
        "tile_settings": "Tiles",
        "tile_size": "Tile size",
        "tile_size_info": "",
        "tile_overlap": "Overlap",
        "tile_overlap_info": "",
        "post_processing": "Post-Processing",
        "sharpening": "Sharpening",
        "sharpening_info": "",
        "contrast": "Contrast",
        "contrast_info": "",
        "saturation": "Saturation",
        "saturation_info": "",
        "advanced_title": "⚡ Advanced",
        "precision_mode": "Precision",
        "precision_info": "",
        "video_export_title": "🎬 Video",
        "video_resolution_label": "Resolution",
        "video_resolution_info": "",
        "export_videos": "Export videos",
        "video_codec": "Codec",
        "codec_profile": "Profile",
        "fps_label": "FPS (0 = auto)",
        "preserve_transparency": "Transparency",
        "keep_audio": "🔊 Keep audio",
        "keep_audio_info": "",
        "skip_duplicates": "⚡ Skip duplicates",
        "skip_duplicates_info": "",
        "video_naming_title": "📝 Naming",
        "video_naming_label": "Mode",
        "video_naming_info": "",
        "naming_same": "Original name",
        "naming_suffix": "Suffix",
        "naming_custom": "Custom",
        "suffix_label": "Suffix",
        "suffix_placeholder": "_upscaled",
        "suffix_info": "",
        "custom_name_label": "Name",
        "custom_name_placeholder": "my_video",
        "custom_name_info": "",
        "auto_cleanup_title": "🗑️ Cleanup",
        "auto_cleanup_accordion": "Auto Cleanup",
        "delete_input_frames": "Delete input frames",
        "delete_input_frames_info": "",
        "delete_output_frames": "Delete output frames",
        "delete_output_frames_info": "",
        "delete_extraction_folder": "🗑️ Delete ALL folders",
        "delete_extraction_folder_info": "Deletes the entire extraction folder (input + output frames + everything)",
        "delete_mapping": "Delete JSON",
        "delete_mapping_info": "",
        "organize_videos": "📁 Dedicated videos/ folder",
        "organize_videos_info": "",
        "organize_images": "📁 Dedicated images/ folder",
        "organize_images_info": "",
        "test_btn": "🧪 Test",
        "test_frame_number": "🎞️ Test frame number",
        "test_frame_number_info": "Choose which frame to test for videos (the first frame is often not interesting)",
        "run_batch_btn": "▶️ Run",
        "test_status": "Test",
        "pause_btn": "⏸️ Pause",
        "stop_btn": "⏹️ Stop",
        "compare_tab": "⚖️ Compare",
        "before_after": "Before / After",
        "frame_navigation": "Navigation",
        "gallery_tab": "🖼️ Gallery",
        "all_results": "Results",
        "status_label": "Status",
        "download_info": "Files",
        "output_folder": "📂 Folder",
        "info_help_title": "",
        "enable_parallel": "⚡ Parallel processing",
        "enable_parallel_info": "",
        "batch_size": "📦 Batch size",
        "batch_size_info": ""
    }
}

# ============================================================================
# Current Language Detection
# ============================================================================
try:
    system_locale = locale.getdefaultlocale()[0]
    if system_locale and system_locale.startswith('en'):
        current_language = "en"
    else:
        current_language = "fr"
except:
    current_language = "fr"  # Fallback to French

# ============================================================================
# Video Codec Configurations
# ============================================================================
VIDEO_CODECS = {
    "H.264 (AVC)": {
        "codec": "h264_nvenc",  # GPU encoding with NVENC (2-5x faster than libx264)
        "alpha_support": False,
        "profiles": {
            "Main (Good)": {"preset": "p4", "cq": 20, "profile": "main"},
            "High (Better)": {"preset": "p7", "cq": 17, "profile": "high"},
            "Lossless (Best)": {"preset": "p4", "cq": 0, "profile": "high"},
        }
    },
    "H.265 (HEVC)": {
        "codec": "hevc_nvenc",  # GPU encoding with NVENC (3-7x faster than libx265)
        "alpha_support": False,
        "profiles": {
            "Main (8-bit)": {"preset": "p4", "cq": 20, "profile": "main", "pix_fmt": "yuv420p"},
            "Main10 (10-bit)": {"preset": "p4", "cq": 20, "profile": "main10", "pix_fmt": "yuv420p10le"},
            "Main10 High Quality": {"preset": "p7", "cq": 16, "profile": "main10", "pix_fmt": "yuv420p10le"},
            "Main10 Fast": {"preset": "p1", "cq": 24, "profile": "main10", "pix_fmt": "yuv420p10le"},
        }
    },
    "ProRes": {
        "codec": "prores_ks",
        "alpha_support": True,
        "profiles": {
            "ProRes 422 Proxy": {"profile": "0", "pix_fmt": "yuv422p10le"},
            "ProRes 422 LT": {"profile": "1", "pix_fmt": "yuv422p10le"},
            "ProRes 422": {"profile": "2", "pix_fmt": "yuv422p10le"},
            "ProRes 422 HQ": {"profile": "3", "pix_fmt": "yuv422p10le"},
            "ProRes 4444": {"profile": "4", "pix_fmt": "yuva444p10le"},
            "ProRes 4444 XQ": {"profile": "5", "pix_fmt": "yuva444p10le"},
        }
    },
    "DNxHD/DNxHR": {
        "codec": "dnxhd",
        "alpha_support": True,
        "profiles": {
            "DNxHD 36": {"profile": "dnxhd", "bitrate": "36M"},
            "DNxHD 115": {"profile": "dnxhd", "bitrate": "115M"},
            "DNxHD 175": {"profile": "dnxhd", "bitrate": "175M"},
            "DNxHD 220": {"profile": "dnxhd", "bitrate": "220M"},
            "DNxHR LB": {"profile": "dnxhr_lb"},
            "DNxHR SQ": {"profile": "dnxhr_sq"},
            "DNxHR HQ": {"profile": "dnxhr_hq"},
            "DNxHR HQX": {"profile": "dnxhr_hqx"},
            "DNxHR 444": {"profile": "dnxhr_444"},
        }
    }
}

# ============================================================================
# Default Export Settings
# ============================================================================
DEFAULT_EXPORT_SETTINGS = {
    "codec": "H.264 (AVC)",
    "profile": "High (Better)",
    "fps": 0,  # 0 = use original FPS
    "preserve_alpha": True,
    "target_resolution": 0  # 0 = Auto (no resize), preserves current behavior
}

# ============================================================================
# Frame Intermediate Format Options
# ============================================================================
FRAME_FORMAT_OPTIONS = {
    "PNG - Uncompressed (16-bit)": {"format": "PNG", "compress_level": 0, "bits": 16},
    "PNG - Normal (8-bit)": {"format": "PNG", "compress_level": 6, "bits": 8},
    "PNG - High Compression (8-bit)": {"format": "PNG", "compress_level": 9, "bits": 8},
    "JPEG - Quality 100%": {"format": "JPEG", "quality": 100},
    "JPEG - Quality 95%": {"format": "JPEG", "quality": 95},
}

# ============================================================================
# Default Upscaling Settings (AUTO: optimized like chaiNNer)
# ============================================================================
DEFAULT_UPSCALING_SETTINGS = {
    "tile_size": 512,  # AUTO: 512px balanced for most GPUs (4-8GB VRAM)
    "tile_overlap": 32,  # AUTO: 32px for smooth blending without slowdown
    "output_format": "PNG",
    "jpeg_quality": 95,
    "sharpening": 0.0,  # AUTO: No sharpening (model decides quality)
    "contrast": 1.0,    # AUTO: No contrast adjustment
    "saturation": 1.0,  # AUTO: No saturation adjustment
    "use_fp16": True,
    "image_target_scale": 2.0  # ×2 by default (preserves current behavior)
}

# ============================================================================
# Pre-Downscale Configuration (v2.8+)
# ============================================================================

# Pré-downscale resolutions (height values, 0 = Original/no downscale)
# Allows reducing resolution BEFORE AI upscaling to save VRAM and processing time
# Example: 4K (3840×2160) → pre-downscale 1080p → upscale ×2 → 4K
#          (instead of: 4K → upscale ×2 → 8K which is much slower)
PRE_DOWNSCALE_OPTIONS = {
    "Original": 0,      # No downscale (default)
    "480p": 480,        # Downscale to 480px height (~854×480 for 16:9)
    "720p": 720,        # Downscale to 720px height (1280×720 for 16:9)
    "1080p": 1080       # Downscale to 1080px height (1920×1080 for 16:9)
}

DEFAULT_PRE_DOWNSCALE = "Original"  # No downscale by default (backward compatible)

# ============================================================================
# Default Models with Download URLs from Upscale-Hub
# ============================================================================
DEFAULT_MODELS = {
    # AniToon - Best for 90s/2000s cartoons and low-quality anime
    "2x_AniToon_RPLKSRS_242500.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/AniToon/2x_AniToon_RPLKSRS_242500.pth",
        "description": "AniToon Small - Fast, for old/low-quality anime",
        "display_name": "AniToon Small"
    },
    "2x_AniToon_RPLKSR_197500.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/AniToon/2x_AniToon_RPLKSR_197500.pth",
        "description": "AniToon - Balanced, for old/low-quality anime",
        "display_name": "AniToon"
    },
    "2x_AniToon_RPLKSRL_280K.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/AniToon/2x_AniToon_RPLKSRL_280K.pth",
        "description": "AniToon Large - Best quality, for old/low-quality anime",
        "display_name": "AniToon Large"
    },
    # Ani4K v2 - Best for modern anime (Bluray to WEB)
    "2x_Ani4Kv2_G6i2_UltraCompact_105K.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/Ani4K-v2/2x_Ani4Kv2_G6i2_UltraCompact_105K.pth",
        "description": "Ani4K v2 UltraCompact - Very fast, for modern anime",
        "display_name": "Ani4K v2 UltraCompact"
    },
    "2x_Ani4Kv2_G6i2_Compact_107500.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/Ani4K-v2/2x_Ani4Kv2_G6i2_Compact_107500.pth",
        "description": "Ani4K v2 Compact - RECOMMENDED - Balanced speed/quality",
        "display_name": "Ani4K v2 Compact (Recommended)"
    },
    # AniSD - For old anime
    "2x_AniSD_AC_RealPLKSR_127500.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/AniSD-RealPLKSR/2x_AniSD_AC_RealPLKSR_127500.pth",
        "description": "AniSD AC - For old anime (AC variant)",
        "display_name": "AniSD AC"
    },
    "2x_AniSD_RealPLKSR_140K.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/AniSD-RealPLKSR/2x_AniSD_RealPLKSR_140K.pth",
        "description": "AniSD - For old anime (general)",
        "display_name": "AniSD"
    },
    # OpenProteus - Alternative to Topaz Proteus
    "2x_OpenProteus_Compact_i2_70K.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/OpenProteus/2x_OpenProteus_Compact_i2_70K.pth",
        "description": "OpenProteus - Free alternative to Topaz Proteus",
        "display_name": "OpenProteus Compact"
    },
    # AniScale2 - Various options
    "2x_AniScale2S_Compact_i8_60K.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/AniScale2/2x_AniScale2S_Compact_i8_60K.pth",
        "description": "AniScale2 Compact - Fast general purpose",
        "display_name": "AniScale2 Compact"
    },
}
