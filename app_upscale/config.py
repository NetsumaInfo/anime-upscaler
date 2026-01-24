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

# Parallel hash computation for duplicate frame detection
ENABLE_PARALLEL_HASH_DETECTION = True  # Set to False to use sequential (fallback)
HASH_WORKERS = min(multiprocessing.cpu_count(), 8)  # Max 8 workers for hash computation

# Async image pre-loading (Phase 2.1 - to be implemented)
ENABLE_ASYNC_PRELOAD = False  # Will be enabled after Phase 2.1 implementation

# Async image saving (Phase 2.3 - to be implemented)
ENABLE_ASYNC_SAVE = False  # Will be enabled after Phase 2.3 implementation

# GPU post-processing (Phase 2.2 - to be implemented)
ENABLE_GPU_POST_PROCESSING = False  # Will be enabled after Phase 2.2 implementation

# ============================================================================
# Multilingual Translations (FR/EN)
# ============================================================================
TRANSLATIONS = {
    "fr": {
        "title": "🎨 Anime Upscaler - Traitement Batch & Export Vidéo",
        "subtitle": "*Upscaling 2x pour anime et dessins animés — Traitement batch multi-fichiers avec export vidéo professionnel*",
        "upload_title": "📁 Télécharger Images/Vidéos (Fichiers multiples supportés)",
        "upload_label": "Déposez les fichiers ici ou cliquez pour parcourir (sélectionnez plusieurs fichiers à la fois)",
        "files_summary": "Résumé des Fichiers",
        "no_files": "Aucun fichier téléchargé",
        "ai_model_title": "🤖 Modèle IA",
        "select_model": "Sélectionner le Modèle",
        "output_format_title": "⚙️ Format de Sortie & Taille",
        "image_scale_label": "🔢 Échelle finale de l'image",
        "image_scale_info": "×1 = dimension native | ×2 = 1 passe | ×4 = 2 passes | ×8 = optimisé | ×16 = optimisé haute qualité",
        "final_output_format": "Format de Sortie Final",
        "jpeg_quality": "Qualité JPEG/WebP",
        "video_frame_format": "🎬 Format Intermédiaire des Frames Vidéo",
        "video_frame_format_info": "Format utilisé lors de la sauvegarde des frames upscalées avant l'encodage vidéo",
        "upscaling_params_title": "🎨 Paramètres d'Upscaling",
        "use_auto": "🤖 Utiliser les paramètres AUTO (Le modèle décide - Recommandé)",
        "use_auto_info": "Laisser le modèle IA utiliser des paramètres optimisés. Décochez pour personnaliser manuellement.",
        "tile_settings": "Paramètres de Tuiles",
        "tile_size": "Taille de Tuile (px)",
        "tile_size_info": "Plus petit = Moins de VRAM, Plus lent | Plus grand = Plus de VRAM, Plus rapide",
        "tile_overlap": "Chevauchement de Tuile (px)",
        "tile_overlap_info": "Plus grand = Meilleur mélange, Plus lent | Plus petit = Plus rapide, Artifacts possibles",
        "post_processing": "Post-Traitement",
        "sharpening": "Netteté",
        "sharpening_info": "0 = Aucune | 0.5-1.0 = Modéré | 1.5-2.0 = Fort",
        "contrast": "Contraste",
        "contrast_info": "<1.0 = Diminuer | 1.0 = Original | >1.0 = Augmenter",
        "saturation": "Saturation",
        "saturation_info": "<1.0 = Diminuer | 1.0 = Original | >1.0 = Augmenter",
        "advanced_title": "⚡ Avancé",
        "precision_mode": "Mode de Précision",
        "precision_info": "None: Désactivé (PyTorch décide automatiquement) | FP16: Force FP16, plus rapide avec CUDA, 50% moins de VRAM | FP32: Force FP32, plus lent, plus de VRAM, précision maximale",
        "video_export_title": "🎬 Paramètres d'Export Vidéo",
        "video_resolution_label": "📐 Résolution vidéo cible",
        "video_resolution_info": "Upscale intelligent multi-passes puis redimensionnement (Auto = 1 passe 2x, pas de resize)",
        "export_videos": "Exporter les vidéos (pas seulement les frames)",
        "video_codec": "Codec Vidéo",
        "codec_profile": "Profil du Codec",
        "fps_label": "FPS (0 = utiliser l'original)",
        "preserve_transparency": "Préserver la transparence (si supporté)",
        "keep_audio": "🔊 Conserver l'audio de la vidéo originale",
        "keep_audio_info": "Copier la piste audio de la vidéo source vers la sortie upscalée",
        "skip_duplicates": "⚡ Ignorer les frames dupliquées (traitement plus rapide)",
        "skip_duplicates_info": "Détecter les frames identiques et réutiliser le résultat upscalé (accélération importante pour les scènes statiques)",
        "video_naming_title": "📝 Nommage de la Sortie Vidéo",
        "video_naming_label": "Mode de nommage",
        "video_naming_info": "Choisir comment nommer le fichier vidéo de sortie",
        "naming_same": "Même nom que l'original",
        "naming_suffix": "Ajouter un suffixe",
        "naming_custom": "Nom personnalisé",
        "suffix_label": "Suffixe à ajouter",
        "suffix_placeholder": "_upscale",
        "suffix_info": "Sera ajouté avant l'extension (ex: video_upscale.mp4)",
        "custom_name_label": "Nom personnalisé",
        "custom_name_placeholder": "ma_video",
        "custom_name_info": "Nom personnalisé (l'extension sera ajoutée automatiquement)",
        "auto_cleanup_title": "🗑️ Nettoyage Automatique",
        "auto_cleanup_accordion": "Paramètres de Suppression Automatique",
        "delete_input_frames": "Supprimer les frames d'entrée après traitement",
        "delete_input_frames_info": "Supprimer automatiquement les frames originales extraites",
        "delete_output_frames": "Supprimer les frames upscalées après encodage",
        "delete_output_frames_info": "Supprimer automatiquement les frames upscalées après encodage réussi",
        "delete_mapping": "Supprimer le fichier de détection des doublons",
        "delete_mapping_info": "Supprimer automatiquement frame_mapping.json après traitement",
        "organize_videos": "📁 Exporter les vidéos vers un dossier videos/ dédié",
        "organize_videos_info": "Placer les vidéos exportées dans output/videos/ au lieu de output/session/",
        "test_btn": "🧪 Test",
        "run_batch_btn": "▶️ Lancer le Batch",
        "test_status": "État du Test",
        "pause_btn": "⏸️ Pause",
        "stop_btn": "⏹️ Stop",
        "compare_tab": "⚖️ Comparer",
        "before_after": "Avant / Après",
        "frame_navigation": "🎞️ Navigation des Frames",
        "gallery_tab": "🖼️ Galerie",
        "all_results": "Tous les Résultats",
        "status_label": "État",
        "download_info": "Informations de Téléchargement",
        "output_folder": "📂 Dossier de Sortie",
        "info_help_title": "ℹ️ Info & Aide",
        "enable_parallel": "⚡ Activer le traitement parallèle des images",
        "enable_parallel_info": "Traiter plusieurs images simultanément (détection VRAM automatique)"
    },
    "en": {
        "title": "🎨 Anime Upscaler - Batch & Video Export",
        "subtitle": "*2x upscaling for anime and cartoons — Multi-file batch processing with professional video export*",
        "upload_title": "📁 Upload Images/Videos (Multiple files supported)",
        "upload_label": "Drop files here or click to browse (select multiple files at once)",
        "files_summary": "Files Summary",
        "no_files": "No files uploaded",
        "ai_model_title": "🤖 AI Model",
        "select_model": "Select Model",
        "output_format_title": "⚙️ Output Format & Size",
        "image_scale_label": "🔢 Final Image Scale",
        "image_scale_info": "×1 = native dimension | ×2 = 1 pass | ×4 = 2 passes | ×8 = optimized | ×16 = optimized high quality",
        "final_output_format": "Final Output Format",
        "jpeg_quality": "JPEG/WebP Quality",
        "video_frame_format": "🎬 Video Frame Intermediate Format",
        "video_frame_format_info": "Format used when saving upscaled frames before video encoding",
        "upscaling_params_title": "🎨 Upscaling Parameters",
        "use_auto": "🤖 Use AUTO settings (Model decides - Recommended)",
        "use_auto_info": "Let the AI model use optimized parameters. Uncheck to customize manually.",
        "tile_settings": "Tile Settings",
        "tile_size": "Tile Size (px)",
        "tile_size_info": "Smaller = Less VRAM, Slower | Larger = More VRAM, Faster",
        "tile_overlap": "Tile Overlap (px)",
        "tile_overlap_info": "Larger = Better blending, Slower | Smaller = Faster, Possible artifacts",
        "post_processing": "Post-Processing",
        "sharpening": "Sharpening",
        "sharpening_info": "0 = None | 0.5-1.0 = Moderate | 1.5-2.0 = Strong",
        "contrast": "Contrast",
        "contrast_info": "<1.0 = Decrease | 1.0 = Original | >1.0 = Increase",
        "saturation": "Saturation",
        "saturation_info": "<1.0 = Decrease | 1.0 = Original | >1.0 = Increase",
        "advanced_title": "⚡ Advanced",
        "precision_mode": "Precision Mode",
        "precision_info": "None: Disabled (PyTorch decides automatically) | FP16: Force FP16, faster on CUDA, 50% less VRAM | FP32: Force FP32, slower, more VRAM, maximum precision",
        "video_export_title": "🎬 Video Export Settings",
        "video_resolution_label": "📐 Target Video Resolution",
        "video_resolution_info": "Smart multi-pass upscale then resize (Auto = 1 pass 2x, no resize)",
        "export_videos": "Export videos (not just frames)",
        "video_codec": "Video Codec",
        "codec_profile": "Codec Profile",
        "fps_label": "FPS (0 = use original)",
        "preserve_transparency": "Preserve transparency (if supported)",
        "keep_audio": "🔊 Keep audio from original video",
        "keep_audio_info": "Copy audio track from source video to upscaled output",
        "skip_duplicates": "⚡ Skip duplicate frames (faster processing)",
        "skip_duplicates_info": "Detect identical frames and reuse upscaled result (huge speedup for static scenes)",
        "video_naming_title": "📝 Video Output Naming",
        "video_naming_label": "Output video naming",
        "video_naming_info": "Choose how to name the output video file",
        "naming_same": "Same as input",
        "naming_suffix": "Add suffix",
        "naming_custom": "Custom name",
        "suffix_label": "Suffix to add",
        "suffix_placeholder": "_upscaled",
        "suffix_info": "Will be added before file extension (e.g., video_upscaled.mp4)",
        "custom_name_label": "Custom video name",
        "custom_name_placeholder": "my_video",
        "custom_name_info": "Custom name (extension will be added automatically)",
        "auto_cleanup_title": "🗑️ Auto-Cleanup (Save Space)",
        "auto_cleanup_accordion": "Auto-Deletion Settings",
        "delete_input_frames": "🗑️ Delete input frames after processing",
        "delete_input_frames_info": "Automatically delete extracted input frames after video is upscaled",
        "delete_output_frames": "🗑️ Delete upscaled frames after encoding",
        "delete_output_frames_info": "Automatically delete upscaled frames after video is successfully encoded",
        "delete_mapping": "🗑️ Delete frame mapping file after processing",
        "delete_mapping_info": "Automatically delete frame_mapping.json (duplicate detection data) after processing",
        "organize_videos": "📁 Export videos to dedicated videos/ folder",
        "organize_videos_info": "Put exported videos in output/videos/ instead of output/session/",
        "test_btn": "🧪 Test",
        "run_batch_btn": "▶️ Run Batch",
        "test_status": "Test Status",
        "pause_btn": "⏸️ Pause",
        "stop_btn": "⏹️ Stop",
        "compare_tab": "⚖️ Compare",
        "before_after": "Before / After",
        "frame_navigation": "🎞️ Frame Navigation",
        "gallery_tab": "🖼️ Gallery",
        "all_results": "All Results",
        "status_label": "Status",
        "download_info": "Download Info",
        "output_folder": "📂 Output Folder",
        "info_help_title": "ℹ️ Info & Help",
        "enable_parallel": "⚡ Enable parallel image processing",
        "enable_parallel_info": "Process multiple images simultaneously (automatic VRAM detection)"
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
        "codec": "libx264",
        "alpha_support": False,
        "profiles": {
            "Main (Good)": {"preset": "medium", "crf": 20, "profile": "main"},
            "High (Better)": {"preset": "slow", "crf": 17, "profile": "high"},
            "Lossless (Best)": {"preset": "medium", "crf": 0, "profile": "high"},
        }
    },
    "H.265 (HEVC)": {
        "codec": "libx265",
        "alpha_support": False,
        "profiles": {
            "Main (8-bit)": {"preset": "medium", "crf": 20, "profile": "main", "pix_fmt": "yuv420p"},
            "Main10 (10-bit)": {"preset": "medium", "crf": 20, "profile": "main10", "pix_fmt": "yuv420p10le"},
            "Main12 (12-bit)": {"preset": "medium", "crf": 20, "profile": "main12", "pix_fmt": "yuv420p12le"},
            "Main 4:4:4 10-bit": {"preset": "medium", "crf": 20, "profile": "main444-10", "pix_fmt": "yuv444p10le"},
            "Main10 High Quality": {"preset": "slow", "crf": 16, "profile": "main10", "pix_fmt": "yuv420p10le"},
            "Main10 Fast": {"preset": "fast", "crf": 24, "profile": "main10", "pix_fmt": "yuv420p10le"},
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
