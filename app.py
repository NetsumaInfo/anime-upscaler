"""
🎨 Anime Upscaler - Batch Processing & Professional Video Export
- Multi-file batch processing (images and videos)
- Professional video export with H.264, H.265, ProRes, DNxHD/HR
- Transparency support for ProRes 4444/XQ and DNxHR 444
- Optimized for NVIDIA CUDA with Spandrel model loading
"""

import torch
from PIL import Image, ImageFilter, ImageEnhance
import gradio as gr
import os
import numpy as np
import subprocess
from pathlib import Path
from datetime import datetime
import requests
from tqdm import tqdm
import hashlib
import shutil

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv'}

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Translations system
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
        "image_scale_info": "×2 = 1 passe | ×3 = 2 passes (2x→4x) puis downscale | ×4 = 2 passes (2x→2x)",
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
        "precision_info": "FP16: Plus rapide avec CUDA, 50% moins de VRAM, précision légèrement réduite | FP32: Plus lent, plus de VRAM, précision maximale",
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
        "info_help_title": "ℹ️ Info & Aide"
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
        "image_scale_info": "×2 = 1 pass | ×3 = 2 passes (2x→4x) then downscale | ×4 = 2 passes (2x→2x)",
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
        "precision_info": "FP16: Faster on CUDA, 50% less VRAM, slightly lower precision | FP32: Slower, more VRAM, maximum precision",
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
        "info_help_title": "ℹ️ Info & Help"
    }
}

# Current language - detect from system locale, default to French
import locale
try:
    system_locale = locale.getdefaultlocale()[0]
    if system_locale and system_locale.startswith('en'):
        current_language = "en"
    else:
        current_language = "fr"
except:
    current_language = "fr"  # Fallback to French

# Video export codec configurations
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

# Default export settings
DEFAULT_EXPORT_SETTINGS = {
    "codec": "H.264 (AVC)",
    "profile": "High (Better)",
    "fps": 0,  # 0 = use original FPS
    "preserve_alpha": True,
    "target_resolution": 0  # 0 = Auto (pas de resize), comportement actuel préservé
}

# Frame intermediate format settings
FRAME_FORMAT_OPTIONS = {
    "PNG - Uncompressed (16-bit)": {"format": "PNG", "compress_level": 0, "bits": 16},
    "PNG - Normal (8-bit)": {"format": "PNG", "compress_level": 6, "bits": 8},
    "PNG - High Compression (8-bit)": {"format": "PNG", "compress_level": 9, "bits": 8},
    "JPEG - Quality 100%": {"format": "JPEG", "quality": 100},
    "JPEG - Quality 95%": {"format": "JPEG", "quality": 95},
}

# Default upscaling settings (AUTO: optimized like chaiNNer)
DEFAULT_UPSCALING_SETTINGS = {
    "tile_size": 512,  # AUTO: 512px balanced for most GPUs (4-8GB VRAM)
    "tile_overlap": 32,  # AUTO: 32px for smooth blending without slowdown
    "output_format": "PNG",
    "jpeg_quality": 95,
    "sharpening": 0.0,  # AUTO: No sharpening (model decides quality)
    "contrast": 1.0,    # AUTO: No contrast adjustment
    "saturation": 1.0,  # AUTO: No saturation adjustment
    "use_fp16": True,
    "image_target_scale": 2.0  # ×2 par défaut (comportement actuel préservé)
}

# Default models with download URLs from Upscale-Hub (https://github.com/Sirosky/Upscale-Hub)
# Display names map technical filenames to user-friendly names
DEFAULT_MODELS = {
    # AniToon - Best for 90s/2000s cartoons and low-quality anime
    "2x_AniToon_RPLKSRS_242500.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/AniToon/2x_AniToon_RPLKSRS_242500.pth",
        "scale": 2,
        "description": "AniToon Small - Fast, for old/low-quality anime",
        "display_name": "AniToon Small"
    },
    "2x_AniToon_RPLKSR_197500.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/AniToon/2x_AniToon_RPLKSR_197500.pth",
        "scale": 2,
        "description": "AniToon - Balanced, for old/low-quality anime",
        "display_name": "AniToon"
    },
    "2x_AniToon_RPLKSRL_280K.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/AniToon/2x_AniToon_RPLKSRL_280K.pth",
        "scale": 2,
        "description": "AniToon Large - Best quality, for old/low-quality anime",
        "display_name": "AniToon Large"
    },
    # Ani4K v2 - Best for modern anime (Bluray to WEB)
    "2x_Ani4Kv2_G6i2_UltraCompact_105K.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/Ani4K-v2/2x_Ani4Kv2_G6i2_UltraCompact_105K.pth",
        "scale": 2,
        "description": "Ani4K v2 UltraCompact - Very fast, for modern anime",
        "display_name": "Ani4K v2 UltraCompact"
    },
    "2x_Ani4Kv2_G6i2_Compact_107500.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/Ani4K-v2/2x_Ani4Kv2_G6i2_Compact_107500.pth",
        "scale": 2,
        "description": "Ani4K v2 Compact - RECOMMENDED - Balanced speed/quality",
        "display_name": "Ani4K v2 Compact (Recommended)"
    },
    # AniSD - For old anime
    "2x_AniSD_AC_RealPLKSR_127500.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/AniSD-RealPLKSR/2x_AniSD_AC_RealPLKSR_127500.pth",
        "scale": 2,
        "description": "AniSD AC - For old anime (AC variant)",
        "display_name": "AniSD AC"
    },
    "2x_AniSD_RealPLKSR_140K.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/AniSD-RealPLKSR/2x_AniSD_RealPLKSR_140K.pth",
        "scale": 2,
        "description": "AniSD - For old anime (general)",
        "display_name": "AniSD"
    },
    # OpenProteus - Alternative to Topaz Proteus
    "2x_OpenProteus_Compact_i2_70K.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/OpenProteus/2x_OpenProteus_Compact_i2_70K.pth",
        "scale": 2,
        "description": "OpenProteus - Free alternative to Topaz Proteus",
        "display_name": "OpenProteus Compact"
    },
    # AniScale2 - Various options
    "2x_AniScale2S_Compact_i8_60K.pth": {
        "url": "https://github.com/Sirosky/Upscale-Hub/releases/download/AniScale2/2x_AniScale2S_Compact_i8_60K.pth",
        "scale": 2,
        "description": "AniScale2 Compact - Fast general purpose",
        "display_name": "AniScale2 Compact"
    },
}

def extract_scale_from_filename(filename: str) -> int:
    """Extract scale factor from model filename (e.g., '2x', '4x')"""
    import re
    match = re.search(r'(\d+)x', filename.lower())
    if match:
        return int(match.group(1))
    return 2  # Default scale

def scan_models() -> tuple[dict, dict]:
    """Scan models directory and build model configuration with display names
    Returns: (models_dict, display_to_file_mapping)
    """
    models = {}
    display_to_file = {}  # Maps display name -> filename

    # Scan for existing models in models folder
    if MODELS_DIR.exists():
        for model_file in MODELS_DIR.iterdir():
            if model_file.suffix.lower() in ['.pth', '.safetensors']:
                model_name = model_file.stem
                scale = extract_scale_from_filename(model_file.name)

                # Get display name from DEFAULT_MODELS if available
                display_name = DEFAULT_MODELS.get(model_file.name, {}).get("display_name", model_name)

                models[display_name] = {
                    "file": model_file.name,
                    "url": DEFAULT_MODELS.get(model_file.name, {}).get("url", None),
                    "scale": scale
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
                    "url": default_config["url"],
                    "scale": default_config["scale"]
                }
                display_to_file[display_name] = default_file

    return models, display_to_file

# Model configurations (auto-detected + defaults)
MODELS, MODEL_DISPLAY_TO_FILE = scan_models()

# Cached models
loaded_models = {}

def clear_gpu_memory():
    """Clear GPU memory and cache"""
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return f"VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "CPU mode"

def download_model(model_name: str):
    """Download model if not present"""
    config = MODELS[model_name]
    model_path = MODELS_DIR / config["file"]

    if model_path.exists():
        return model_path

    # If no URL is provided (manually added model), return path anyway
    if not config.get("url"):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file '{config['file']}' not found in {MODELS_DIR}")
        return model_path

    print(f"📥 Downloading {model_name}...")
    url = config["url"]

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

def load_model(model_name: str, use_fp16: bool = True):
    """Load model using Spandrel with optimizations"""
    global loaded_models

    # Create cache key based on model name and precision
    cache_key = f"{model_name}_fp16" if use_fp16 and DEVICE == "cuda" else model_name

    if cache_key in loaded_models:
        return loaded_models[cache_key]

    try:
        from spandrel import ImageModelDescriptor, ModelLoader
    except ImportError:
        raise ImportError("Spandrel not installed. Run: pip install spandrel")

    model_path = download_model(model_name)
    print(f"⏳ Loading {model_name}...")

    model = ModelLoader().load_from_file(str(model_path))

    if isinstance(model, ImageModelDescriptor):
        model = model.model

    model = model.to(DEVICE).eval()

    # Apply FP16 conversion if CUDA available and enabled
    if DEVICE == "cuda" and use_fp16:
        try:
            model = model.half()
            print(f"✅ FP16 enabled (VRAM usage reduced by ~50%)")
        except Exception as e:
            print(f"⚠️ FP16 conversion failed: {e}, using FP32")
            use_fp16 = False

    # Try to compile model with torch.compile for ~20-30% speedup (PyTorch 2.0+)
    # Note: Requires Triton on Linux, may not work on Windows
    try:
        if DEVICE == "cuda" and hasattr(torch, 'compile'):
            # Suppress triton errors and fall back to eager mode if compilation fails
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="reduce-overhead")
            print(f"✅ Model compiled with torch.compile (faster inference)")
    except Exception as e:
        print(f"⚠️ torch.compile not available ({type(e).__name__}), using uncompiled model")
        # This is expected on Windows or without Triton

    loaded_models[cache_key] = model
    print(f"✅ {model_name} loaded on {DEVICE} ({'FP16' if use_fp16 and DEVICE == 'cuda' else 'FP32'})")

    return model

def apply_post_processing(img: Image.Image, sharpening: float = 0.0, contrast: float = 1.0, saturation: float = 1.0):
    """Apply post-processing enhancements"""
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

def resize_to_1080p(img: Image.Image) -> Image.Image:
    """Resize image to 1080p max height while preserving aspect ratio"""
    width, height = img.size

    # Only resize if height exceeds 1080px
    if height > 1080:
        # Calculate new dimensions while preserving aspect ratio
        new_height = 1080
        new_width = int(width * (1080 / height))

        # Use LANCZOS for high-quality downscaling
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img

def calculate_upscale_passes(original_height: int, target_height: int) -> int:
    """
    Calcule le nombre de passes 2x nécessaires pour atteindre ou dépasser la résolution cible.

    Args:
        original_height: Hauteur de l'image source
        target_height: Hauteur cible souhaitée

    Returns:
        Nombre de passes 2x à effectuer (minimum 1)

    Exemples:
        480p → 1080p: 480 * 2 = 960 (< 1080), 960 * 2 = 1920 (> 1080) → 2 passes
        1080p → 1080p: 1080 * 2 = 2160 (> 1080) → 1 passe
        720p → 4K (2160p): 720 * 2 = 1440 (< 2160), 1440 * 2 = 2880 (> 2160) → 2 passes
    """
    if target_height == 0:
        return 1  # Auto mode: 1 passe seulement

    current_height = original_height
    passes = 0

    # Calculer combien de passes 2x pour dépasser la cible
    while current_height < target_height:
        current_height *= 2
        passes += 1

    # Si déjà au-dessus de la cible, au moins 1 passe
    if passes == 0:
        passes = 1

    return passes

def resize_to_target_resolution(img: Image.Image, target_height: int, original_aspect_ratio: float) -> Image.Image:
    """
    Redimensionne l'image vers une hauteur cible en préservant l'aspect ratio.

    Args:
        img: Image à redimensionner
        target_height: Hauteur cible (0 = pas de resize)
        original_aspect_ratio: Ratio largeur/hauteur de l'image originale (width/height)

    Returns:
        Image redimensionnée avec LANCZOS

    Exemples:
        Image 3840×2160 (16:9) → cible 1080p → 1920×1080 (préserve 16:9)
        Image 2880×2160 (4:3) → cible 1080p → 1440×1080 (préserve 4:3)
    """
    if target_height == 0:
        return img  # Auto mode: pas de redimensionnement

    current_width, current_height = img.size

    if current_height == target_height:
        return img  # Déjà à la bonne résolution

    # Calculer largeur proportionnelle pour préserver aspect ratio original
    target_width = int(target_height * original_aspect_ratio)

    # LANCZOS pour haute qualité (downscale OU upscale)
    return img.resize((target_width, target_height), Image.Resampling.LANCZOS)

def resize_to_target_scale(img: Image.Image, target_scale: float, original_size: tuple) -> Image.Image:
    """
    Redimensionne l'image vers un facteur d'échelle cible.

    Args:
        img: Image upscalée (peut avoir été upscalée plusieurs fois)
        target_scale: Facteur d'échelle final souhaité (2.0, 3.0, 4.0)
        original_size: (width, height) de l'image originale

    Returns:
        Image redimensionnée avec LANCZOS

    Exemples:
        Original 1000×1000, upscalé 2x → 2000×2000, target_scale=2.0 → 2000×2000 (aucun resize)
        Original 1000×1000, upscalé 4x → 4000×4000, target_scale=3.0 → 3000×3000 (downscale)
    """
    orig_width, orig_height = original_size
    target_width = int(orig_width * target_scale)
    target_height = int(orig_height * target_scale)

    current_width, current_height = img.size

    if current_width == target_width and current_height == target_height:
        return img  # Déjà à la bonne taille

    return img.resize((target_width, target_height), Image.Resampling.LANCZOS)

def apply_dithering(img_float: np.ndarray, enable: bool = True) -> np.ndarray:
    """Apply dithering to reduce banding artifacts during float->uint8 conversion

    Banding occurs when converting high-precision float (millions of values) to 8-bit (256 values).
    Dithering adds controlled noise that makes the quantization error less visible.

    Uses simple random dithering (faster than Floyd-Steinberg for large images).

    Args:
        img_float: Image array in float [0, 1] range, shape (H, W, 3)

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

def create_gaussian_weight_map(tile_h: int, tile_w: int, overlap: int) -> np.ndarray:
    """Create linear weight map for smooth tile blending

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
    use_fp16: bool
) -> Image.Image:
    """
    Effectue UNE SEULE passe d'upscaling 2x.
    Fonction interne utilisée par upscale_image() pour le multi-passes.

    Args:
        img: Image PIL en mode RGB
        model: Modèle Spandrel chargé
        scale: Facteur d'échelle du modèle (toujours 2)
        tile_size: Taille des tuiles
        tile_overlap: Chevauchement des tuiles
        use_fp16: Utiliser FP16 pour accélérer

    Returns:
        Image upscalée 2x en mode RGB
    """
    # Get numpy array - ensure no color space conversion
    img_np = np.array(img, dtype=np.float32) / 255.0

    # Convert to tensor (FIXED: separate dtype conversion from device transfer to prevent artifacts)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    # Apply FP16 conversion AFTER tensor creation
    if DEVICE == "cuda" and use_fp16:
        try:
            img_tensor = img_tensor.half()
        except:
            pass  # Fallback to FP32 if conversion fails

    img_tensor = img_tensor.to(DEVICE)

    h, w = img_np.shape[:2]

    # For small images, process directly
    if h * w <= tile_size * tile_size:
        with torch.no_grad():
            output = model(img_tensor)
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

    # Process tiles with memory optimization and Gaussian blending
    for y in range(0, h, tile_size - tile_overlap):
        for x in range(0, w, tile_size - tile_overlap):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = img_np[y:y_end, x:x_end]

            # Convert tile using FIXED method
            tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0)

            # Apply FP16 conversion AFTER tensor creation
            if DEVICE == "cuda" and use_fp16:
                try:
                    tile_tensor = tile_tensor.half()
                except:
                    pass

            tile_tensor = tile_tensor.to(DEVICE)

            with torch.no_grad():
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

            # Create Gaussian weight map for this tile
            tile_weight = create_gaussian_weight_map(th, tw, tile_overlap * scale)

            # Apply weighted blending
            result[y_out:y_out+th, x_out:x_out+tw] += tile_output * tile_weight
            weight[y_out:y_out+th, x_out:x_out+tw] += tile_weight

            # Clean up tile tensors to free VRAM
            del tile_tensor, tile_output, tile_weight

    # Clear CUDA cache after all tiles processed
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Normalize by weight
    result = result / np.maximum(weight, 1e-8)

    # FINAL clipping: ensure [0,1] range before conversion
    result = np.clip(result, 0.0, 1.0)

    # Convert float [0,1] to uint8 [0,255]
    result_uint8 = (result * 255.0).round().astype(np.uint8)
    result_img = Image.fromarray(result_uint8, mode='RGB')

    return result_img

def upscale_image(img: Image.Image, model_name: str,
                 preserve_alpha: bool = False, output_format: str = "PNG", jpeg_quality: int = 95,
                 use_fp16: bool = True, downscale_to_1080p: bool = False,
                 tile_size: int = None, tile_overlap: int = None,
                 sharpening: float = None, contrast: float = None, saturation: float = None,
                 target_scale: float = 2.0, target_resolution: int = 0, is_video_frame: bool = False):
    """Upscale image with tile-based processing and multi-pass support (AUTO or custom settings)"""
    # Use AUTO settings if not provided (model decides quality, not user)
    tile_size = tile_size if tile_size is not None else DEFAULT_UPSCALING_SETTINGS["tile_size"]
    tile_overlap = tile_overlap if tile_overlap is not None else DEFAULT_UPSCALING_SETTINGS["tile_overlap"]
    sharpening = sharpening if sharpening is not None else DEFAULT_UPSCALING_SETTINGS["sharpening"]
    contrast = contrast if contrast is not None else DEFAULT_UPSCALING_SETTINGS["contrast"]
    saturation = saturation if saturation is not None else DEFAULT_UPSCALING_SETTINGS["saturation"]
    model = load_model(model_name, use_fp16)
    scale = MODELS[model_name]["scale"]

    # Stocker dimensions et aspect ratio originaux (AVANT toute transformation)
    original_width, original_height = img.size
    original_aspect_ratio = original_width / original_height

    # Calculer le nombre de passes nécessaires
    if is_video_frame and target_resolution > 0:
        # Pour vidéos: calculer passes pour atteindre résolution cible
        num_passes = calculate_upscale_passes(original_height, target_resolution)
    elif not is_video_frame and target_scale > 2.0:
        # Pour images: calculer passes pour scale cible
        # ×2 = 1 passe, ×3 = 2 passes (2x → 4x puis downscale), ×4 = 2 passes (2x → 2x)
        if target_scale <= 2.0:
            num_passes = 1
        elif target_scale <= 4.0:
            num_passes = 2
        else:
            num_passes = 3  # Pour ×8 si on veut l'ajouter plus tard
    else:
        # Par défaut: 1 passe (comportement actuel)
        num_passes = 1

    # CRITICAL: Remove ICC profile to prevent color shifts (like chaiNNer does)
    if 'icc_profile' in img.info:
        img_data = img.tobytes()
        img = Image.frombytes(img.mode, img.size, img_data)

    # Store original alpha channel if present
    original_alpha = None
    if img.mode in ('RGBA', 'LA') and preserve_alpha:
        original_alpha = img.getchannel('A')

    # --- BOUCLE MULTI-PASSES D'UPSCALING ---
    current_img = img

    for pass_num in range(num_passes):
        # Convert to RGB si nécessaire
        if current_img.mode != 'RGB':
            if current_img.mode == 'RGBA':
                current_img = Image.fromarray(np.array(current_img)[:, :, :3], mode='RGB')
            else:
                current_img = current_img.convert('RGB')

        # UNE passe d'upscaling 2x
        current_img = _upscale_single_pass(
            current_img,
            model,
            scale,
            tile_size,
            tile_overlap,
            use_fp16
        )

    result_img = current_img

    # Apply post-processing (UNE SEULE FOIS après toutes les passes)
    result_img = apply_post_processing(result_img, sharpening, contrast, saturation)

    # NOUVEAU : Redimensionnement intelligent selon le type
    if is_video_frame and target_resolution > 0:
        # Pour vidéos: redimensionner vers résolution cible avec aspect ratio préservé
        result_img = resize_to_target_resolution(result_img, target_resolution, original_aspect_ratio)
    elif not is_video_frame and target_scale != 2.0:
        # Pour images: redimensionner vers scale cible
        result_img = resize_to_target_scale(result_img, target_scale, (original_width, original_height))

    # Apply preserved alpha channel
    if original_alpha is not None:
        upscaled_alpha = original_alpha.resize((result_img.width, result_img.height), Image.Resampling.LANCZOS)
        result_img = result_img.convert('RGBA')
        result_img.putalpha(upscaled_alpha)

    return result_img, img

def compute_frame_hash(img_path: str) -> str:
    """Compute perceptual hash of frame for duplicate detection (tolerant to compression noise)"""
    # Use perceptual hashing to detect visually identical frames
    # even if they have minor pixel differences from compression artifacts
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

def analyze_duplicate_frames(frames_dir: str, progress_callback=None) -> dict:
    """Analyze all frames and create a mapping of unique frames to their duplicates

    Returns:
        dict: {
            "unique_frames": {hash: [list of frame paths with this hash]},
            "frame_to_unique": {frame_path: unique_frame_path},
            "stats": {total, unique, duplicates}
        }
    """
    import json
    from pathlib import Path

    frames_path = Path(frames_dir)
    frames = sorted([f for f in frames_path.iterdir() if f.suffix.lower() == '.png'])

    if not frames:
        return None

    # Phase 1: Compute hashes for all frames
    hash_to_frames = {}  # {hash: [frame_paths]}
    frame_to_hash = {}   # {frame_path: hash}

    import os 

    for i, frame_path in enumerate(frames):
        if progress_callback:
            progress_callback((i + 1) / len(frames), desc=f"Analyzing frames {i+1}/{len(frames)}")

        # Normalize path to ensure consistency (resolve absolute path)
        norm_path = os.path.normpath(os.path.abspath(str(frame_path)))
        
        frame_hash = compute_frame_hash(norm_path)
        frame_to_hash[norm_path] = frame_hash

        if frame_hash not in hash_to_frames:
            hash_to_frames[frame_hash] = []
        hash_to_frames[frame_hash].append(norm_path)

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
            "duplicate_percentage": (duplicate_count / total_frames * 100) if total_frames > 0 else 0
        }
    }

    # Save mapping to JSON file for debugging/inspection
    mapping_file = frames_path.parent / "frame_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)

    return mapping

def save_frame_with_format(img: Image.Image, path: Path, frame_format_name: str):
    """Save frame with specified intermediate format"""
    config = FRAME_FORMAT_OPTIONS[frame_format_name]

    if config["format"] == "PNG":
        # Handle bit depth for PNG
        if config["bits"] == 16:
            # Convert to 16-bit PNG
            if img.mode == 'RGB':
                img_16bit = img.convert('I;16')  # 16-bit grayscale
                # For RGB, we need to save each channel
                img.save(path.with_suffix('.png'), 'PNG', compress_level=config["compress_level"], bits=16)
            else:
                img.save(path.with_suffix('.png'), 'PNG', compress_level=config["compress_level"])
        else:
            # 8-bit PNG
            img.save(path.with_suffix('.png'), 'PNG', compress_level=config["compress_level"])

    elif config["format"] == "JPEG":
        # Convert RGBA to RGB for JPEG
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else img.split()[1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(path.with_suffix('.jpg'), 'JPEG', quality=config["quality"], optimize=True)

def get_video_frame_count(video_path: str):
    """Get total frame count of video using FFprobe"""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            video_path
        ], capture_output=True, text=True)
        frame_count = int(result.stdout.strip())
        return frame_count
    except:
        # Fallback method using duration and fps
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

def extract_frames(video: str, out_dir: str):
    """Extract frames from video with alpha channel support and verification"""
    os.makedirs(out_dir, exist_ok=True)

    # Get expected frame count from video
    expected_frames = get_video_frame_count(video)

    # Extract frames
    result = subprocess.run([
        "ffmpeg", "-i", video,
        "-pix_fmt", "rgba",  # Support alpha channel
        "-start_number", "0",
        os.path.join(out_dir, "frame_%05d.png"), "-y"
    ], capture_output=True, text=True)

    # Verify extraction
    extracted_frames = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.png')])
    extracted_count = len(extracted_frames)

    # Check if extraction was complete
    if expected_frames is not None:
        if extracted_count == 0:
            raise RuntimeError(f"❌ Frame extraction failed: No frames extracted from video. FFmpeg error: {result.stderr}")
        elif extracted_count < expected_frames:
            raise RuntimeError(f"❌ Incomplete frame extraction: Expected {expected_frames} frames but only extracted {extracted_count} frames. FFmpeg may have encountered an error.")
        elif extracted_count > expected_frames:
            # This shouldn't happen but log it as a warning
            print(f"⚠️ Warning: Extracted {extracted_count} frames but expected {expected_frames}")
    else:
        # Couldn't detect expected count, just verify we got something
        if extracted_count == 0:
            raise RuntimeError(f"❌ Frame extraction failed: No frames extracted from video. FFmpeg error: {result.stderr}")

    print(f"✅ Successfully extracted {extracted_count} frames{f' (expected: {expected_frames})' if expected_frames else ''}")
    return extracted_frames

def get_video_fps(video_path: str):
    """Get FPS of input video"""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ], capture_output=True, text=True)
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            return num / den
        return float(fps_str)
    except:
        return 24.0  # Default fallback

def encode_video(frames_dir: str, output_path: str, codec_name: str, profile_name: str, fps: float, preserve_alpha: bool = True, original_video_path: str = None, keep_audio: bool = False):
    """Encode video from frames with specified codec and profile, optionally keeping audio from original"""
    codec_config = VIDEO_CODECS[codec_name]
    profile_config = codec_config["profiles"][profile_name]
    codec = codec_config["codec"]

    # Check if alpha should be preserved
    has_alpha_support = codec_config["alpha_support"] and preserve_alpha

    # Build FFmpeg command
    # CRITICAL: Address "Washed Out" colors (PC vs TV Range mismatch)
    # 1. Input: PNGs are Full Range (0-255). We tell FFmpeg this explicitely.
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
        "-color_range", "tv" # TV Range (Limited) is standard for video compatibility
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
        # ProRes usually likes 'unknown' or auto, but we can try to tag it if needed.
        # Often better to stick to standard tags.
        cmd.extend(color_metadata)

    elif codec_name == "DNxHD/DNxHR":
        profile = profile_config["profile"]
        
        # DNxHR often works with full range, but standardizing on TV range resolves the "washed out" 
        # issue for most common players unless using specialized NLEs.
        
        if "dnxhr" in profile:
            cmd.extend(["-c:v", profile])
            if has_alpha_support and "444" in profile:
                dnx_pix_fmt = "yuva444p10le"
            else:
                dnx_pix_fmt = "yuv422p10le"
            
            cmd.extend(["-vf", f"scale=in_range=pc:out_range=tv,format={dnx_pix_fmt}"])
            cmd.extend(["-pix_fmt", dnx_pix_fmt])
        else:
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

def detect_type(path: str):
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXTENSIONS: return "image"
    if ext in VIDEO_EXTENSIONS: return "video"
    return None

def separate_files_by_type(files):
    """Separate uploaded files into images and videos"""
    images = []
    videos = []

    if not files:
        return images, videos

    file_list = files if isinstance(files, list) else [files]

    for f in file_list:
        path = f.name if hasattr(f, 'name') else f
        ftype = detect_type(path)
        if ftype == "image":
            images.append(path)
        elif ftype == "video":
            videos.append(path)

    return images, videos

def preview_media(file):
    if not file:
        return gr.update(visible=False), gr.update(visible=False)
    path = file.name if hasattr(file, 'name') else file
    ftype = detect_type(path)
    if ftype == "image":
        return gr.update(value=path, visible=True), gr.update(visible=False)
    elif ftype == "video":
        return gr.update(visible=False), gr.update(value=path, visible=True)
    return gr.update(visible=False), gr.update(visible=False)

# Processing state
processing_state = {"running": False, "paused": False, "stop": False}

# Frame storage for navigation
frame_pairs = []

def rgba_to_rgb_for_display(img: Image.Image) -> np.ndarray:
    """Convert RGBA image to RGB with white background for display"""
    if img.mode == 'RGBA':
        # Create white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        return np.array(background)
    elif img.mode == 'RGB':
        return np.array(img)
    else:
        # Convert to RGB for other modes
        return np.array(img.convert('RGB'))

def stop_processing():
    """Stop and reset to initial state"""
    processing_state["stop"] = True
    processing_state["running"] = False
    processing_state["paused"] = False
    return "⏹️ Stopped"

def pause_processing():
    """Toggle pause/resume"""
    processing_state["paused"] = not processing_state["paused"]
    if processing_state["paused"]:
        return gr.update(value="▶️ Resume"), "⏸️ Paused"
    else:
        return gr.update(value="⏸️ Pause"), "▶️ Resumed"

def save_image_with_format(img: Image.Image, path: Path, output_format: str, jpeg_quality: int = 95):
    """Save image with specified format (without ICC profile to preserve exact colors)"""
    if output_format == "JPEG":
        # Convert RGBA to RGB for JPEG
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else img.split()[1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(path.with_suffix('.jpg'), 'JPEG', quality=jpeg_quality, optimize=True, icc_profile=None)
    elif output_format == "WebP":
        img.save(path.with_suffix('.webp'), 'WebP', quality=jpeg_quality, method=6, icc_profile=None)
    else:  # PNG (default)
        img.save(path.with_suffix('.png'), 'PNG', optimize=True, icc_profile=None)

def process_batch(files, model, image_scale_radio, video_resolution_dropdown, output_format, jpeg_quality, precision_mode, codec_name, profile_name, fps, preserve_alpha, export_video, keep_audio, frame_format,
                 auto_delete_input_frames, auto_delete_output_frames, auto_delete_frame_mapping, organize_videos_folder, skip_duplicate_frames,
                 use_auto_settings, tile_size, tile_overlap, sharpening, contrast, saturation,
                 video_naming_mode, video_suffix, video_custom_name, progress=gr.Progress()):
    """Process multiple files with video export support, auto-cleanup, and duplicate frame detection"""
    # Convert precision mode to boolean
    use_fp16 = (precision_mode == "FP16 (Half Precision)")

    # Conversion ×2/×3/×4 → float pour images
    scale_mapping = {"×2": 2.0, "×3": 3.0, "×4": 4.0}
    image_target_scale = scale_mapping.get(image_scale_radio, 2.0)

    # Résolution cible pour vidéos (déjà un int : 0, 720, 1080, etc.)
    video_target_resolution = video_resolution_dropdown

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

    global processing_state, frame_pairs
    processing_state = {"running": True, "paused": False, "stop": False}
    frame_pairs = []

    if not files:
        return None, None, "", "", gr.update(visible=False), gr.update(visible=False), ""

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

        for idx, img_path in enumerate(images):
            if processing_state["stop"]:
                break

            while processing_state["paused"]:
                import time
                time.sleep(0.1)
                if processing_state["stop"]:
                    break

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
            frame_pairs.append((rgba_to_rgb_for_display(orig_resized), rgba_to_rgb_for_display(result)))

            # Free memory from PIL images
            # NOTE: img and orig are the same reference, so only close once
            orig.close()
            result.close()
            del img, result, orig, orig_resized

            # Clear GPU cache every 5 images to prevent memory accumulation
            if DEVICE == "cuda" and idx % 5 == 0:
                clear_gpu_memory()

        status_messages.append(f"✅ {len(images)} image(s) completed")

    # Process videos
    if videos:
        status_messages.append(f"🎬 Processing {len(videos)} video(s)...")

        for vid_idx, video_path in enumerate(videos):
            if processing_state["stop"]:
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

            import time
            start_time = time.time()

            # PHASE 1: Analyze duplicates if option enabled
            frame_mapping = None
            if skip_duplicate_frames:
                progress(0.10, desc=f"{vid_name} - Analyzing duplicate frames...")
                frame_mapping = analyze_duplicate_frames(str(frames_in), progress_callback=progress)

                if frame_mapping and frame_mapping["stats"]["duplicates"] > 0:
                    stats = frame_mapping["stats"]
                    status_messages.append(
                        f"📊 {vid_name}: Found {stats['duplicates']} duplicates "
                        f"({stats['duplicate_percentage']:.1f}%) - Will upscale only {stats['unique_frames']} unique frames"
                    )

            # Initialize tracking
            frames_upscaled = 0
            frames_skipped = 0
            upscaled_cache = {}  # {input_frame_path: output_frame_path}

            # PHASE 2: Process frames with optimized memory management
            for i, fp in enumerate(frames):
                if processing_state["stop"]:
                    break

                while processing_state["paused"]:
                    import time
                    time.sleep(0.1)
                    if processing_state["stop"]:
                        break

                base_progress = (vid_idx + len(images)) / (len(images) + len(videos))
                frame_progress = (i + 1) / total_frames / len(videos)
                progress(base_progress + frame_progress, desc=f"{vid_name} - Frame {i+1}/{total_frames}")

                frame_path_out = frames_out / f"frame_{i:05d}"

                # Check if this frame is a duplicate using the pre-computed mapping
                is_duplicate = False
                
                # Normalize path to ensure consistency with analysis phase
                current_frame_path = os.path.normpath(os.path.abspath(str(fp)))
                unique_frame_path = current_frame_path

                if frame_mapping:
                    # Get the unique frame path for this frame
                    unique_frame_path = frame_mapping["frame_to_unique"].get(current_frame_path, current_frame_path)
                    is_duplicate = (unique_frame_path != current_frame_path)

                    # Debug logging
                    if i < 5 or is_duplicate:  # Log first 5 frames or any duplicate
                        print(f"DEBUG Frame {i}: {Path(fp).name} -> is_duplicate={is_duplicate}, unique={Path(unique_frame_path).name}")

                # Check if we already upscaled the unique version
                if is_duplicate and unique_frame_path in upscaled_cache:
                    # DUPLICATE FRAME: Copy existing upscaled frame
                    cached_output_path = upscaled_cache[unique_frame_path]

                    # Determine extension from frame format
                    if "JPEG" in frame_format:
                        frame_path_out = frame_path_out.with_suffix('.jpg')
                    else:
                        frame_path_out = frame_path_out.with_suffix('.png')

                    # Debug logging for duplicate copies
                    print(f"COPY: {Path(fp).name} → {frame_path_out.name} (from cached {Path(cached_output_path).name})")

                    shutil.copy2(cached_output_path, frame_path_out)
                    frames_skipped += 1

                    # Load cached result for UI preview only
                    # CRITICAL: Create in-memory copies to avoid "Operation on closed image" error
                    with Image.open(cached_output_path) as result_file:
                        result = result_file.copy()
                    with Image.open(fp) as orig_file:
                        orig = orig_file.copy()

                else:
                    # UNIQUE FRAME: Upscale normally
                    img = Image.open(fp)
                    result, orig = upscale_image(img, model, preserve_alpha,
                                                output_format, jpeg_quality, use_fp16,
                                                target_scale=2.0,
                                                target_resolution=video_target_resolution,
                                                is_video_frame=True, **params)
                    # Save frame with chosen intermediate format
                    save_frame_with_format(result, frame_path_out, frame_format)

                    # Determine actual saved path with extension
                    if "JPEG" in frame_format:
                        saved_path = frame_path_out.with_suffix('.jpg')
                    else:
                        saved_path = frame_path_out.with_suffix('.png')

                    # Debug logging for unique frames
                    print(f"UPSCALE: {Path(fp).name} → {Path(saved_path).name} (stored in cache for duplicates)")

                    # Store in cache for future duplicates
                    # CRITICAL: Use unique_frame_path as key so all duplicates can find it
                    upscaled_cache[unique_frame_path] = str(saved_path)
                    frames_upscaled += 1

                    # NOTE: Don't close img here because orig is the same reference
                    # It will be closed later via orig.close() at line ~1626

                # UI updates (same for both duplicate and unique frames)
                all_results.append(rgba_to_rgb_for_display(result))

                # Store pair for navigation with white background for display
                orig_resized = orig.resize(result.size, Image.Resampling.LANCZOS)
                frame_pairs.append((rgba_to_rgb_for_display(orig_resized), rgba_to_rgb_for_display(result)))

                # Free memory from PIL images
                result.close()
                orig.close()
                del result, orig, orig_resized

                # Clear GPU cache every 10 frames to prevent memory accumulation
                if DEVICE == "cuda" and i % 10 == 0:
                    clear_gpu_memory()

                # Auto-delete input frame after processing (if enabled)
                if auto_delete_input_frames:
                    try:
                        os.remove(fp)
                    except Exception as e:
                        print(f"⚠️ Failed to delete input frame {fp}: {e}")

            # Export video if requested
            if export_video and not processing_state["stop"]:
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

            # Auto-delete input frames folder if it's empty or if auto-delete was enabled
            if auto_delete_input_frames:
                try:
                    if frames_in.exists():
                        shutil.rmtree(frames_in)
                        status_messages.append(f"🗑️ {vid_name}: Cleaned up input frames folder")
                except Exception as e:
                    print(f"⚠️ Failed to delete input frames folder: {e}")

            # Auto-delete frame_mapping.json if enabled
            if auto_delete_frame_mapping:
                frame_mapping_file = vid_session / "frame_mapping.json"
                if frame_mapping_file.exists():
                    try:
                        os.remove(frame_mapping_file)
                        status_messages.append(f"🗑️ {vid_name}: Deleted frame_mapping.json")
                    except Exception as e:
                        print(f"⚠️ Failed to delete frame_mapping.json: {e}")

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

            # Add performance statistics if duplicate detection was enabled
            processing_time = time.time() - start_time
            if skip_duplicate_frames:
                if frames_skipped > 0:
                    percentage = (frames_skipped / total_frames) * 100
                    status_messages.append(f"⚡ {vid_name}: {frames_skipped}/{total_frames} duplicate frames skipped ({percentage:.1f}%)")
                    status_messages.append(f"⏱️ {vid_name}: Processing time: {processing_time:.1f}s ({frames_upscaled} frames upscaled)")
                else:
                    status_messages.append(f"⚠️ {vid_name}: No duplicate frames found (all {total_frames} frames unique)")
                    status_messages.append(f"⏱️ {vid_name}: Processing time: {processing_time:.1f}s")

            status_messages.append(f"✅ {vid_name}: {total_frames} frames processed")

    processing_state["running"] = False
    progress(1.0)

    # Prepare outputs
    first_pair = frame_pairs[0] if frame_pairs else (None, None)
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

    frame_updates = gr.update(maximum=max(1, len(frame_pairs)), value=1, visible=len(frame_pairs) > 1)
    frame_label_update = gr.update(value=f"Frame 1/{len(frame_pairs)}", visible=len(frame_pairs) > 1)

    return (first_pair, all_results, final_status, str(session),
            frame_updates, frame_label_update, download_text)

def navigate_frame(frame_idx):
    """Navigate to a specific frame and update the comparison slider"""
    global frame_pairs
    if not frame_pairs:
        return None, ""

    idx = int(frame_idx) - 1
    if 0 <= idx < len(frame_pairs):
        pair = frame_pairs[idx]
        return pair, f"Frame {idx + 1}/{len(frame_pairs)}"
    return None, ""

def update_codec_profiles(codec_name):
    """Update available profiles based on selected codec"""
    if codec_name in VIDEO_CODECS:
        profiles = list(VIDEO_CODECS[codec_name]["profiles"].keys())
        return gr.update(choices=profiles, value=profiles[0])
    return gr.update(choices=[], value=None)

def show_file_summary(files):
    """Display summary of uploaded files with dimensions"""
    if not files:
        return "No files uploaded"

    images, videos = separate_files_by_type(files)

    summary = []
    if images:
        # Display each image with its dimensions
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    img_name = Path(img_path).name
                    summary.append(f"📸 {img_name} ({width}×{height})")
            except Exception as e:
                img_name = Path(img_path).name
                summary.append(f"📸 {img_name} (error reading dimensions)")

    if videos:
        # Display each video with its resolution
        for vid_path in videos:
            try:
                vid_name = Path(vid_path).name
                # Get video resolution using ffprobe
                result = subprocess.run([
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=p=0",
                    vid_path
                ], capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    width, height = result.stdout.strip().split(',')
                    summary.append(f"🎬 {vid_name} ({width}×{height})")
                else:
                    summary.append(f"🎬 {vid_name}")
            except Exception as e:
                vid_name = Path(vid_path).name
                summary.append(f"🎬 {vid_name}")

    return "\n".join(summary) if summary else "No valid files"

def test_image_upscale(uploaded_files, model, image_scale_radio, video_resolution_dropdown, output_format, jpeg_quality,
                       precision_mode, preserve_alpha, use_auto_settings, tile_size, tile_overlap,
                       sharpening, contrast, saturation):
    """Quick test upscale on the first uploaded file (image or video first frame) for testing model"""
    # Convert precision mode to boolean
    use_fp16 = (precision_mode == "FP16 (Half Precision)")

    # Conversion ×2/×3/×4 → float pour images (test utilise toujours l'échelle image)
    scale_mapping = {"×2": 2.0, "×3": 3.0, "×4": 4.0}
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

    if not uploaded_files or len(uploaded_files) == 0:
        return None, gr.update(visible=True, value="❌ No files uploaded. Please upload images/videos first.")

    # Get first file
    first_file = uploaded_files[0] if isinstance(uploaded_files, list) else uploaded_files
    file_ext = Path(first_file).suffix.lower()

    try:
        # Handle video files - extract first frame
        if file_ext in VIDEO_EXTENSIONS:
            import tempfile
            # Create a persistent temp directory for the frame
            temp_dir = tempfile.mkdtemp()
            try:
                # Extract only the first frame
                temp_frame = os.path.join(temp_dir, "frame_00000.png")
                result = subprocess.run([
                    "ffmpeg", "-i", first_file,
                    "-vframes", "1",  # Extract only first frame
                    "-pix_fmt", "rgba",
                    temp_frame, "-y"
                ], capture_output=True, text=True)

                if result.returncode != 0 or not os.path.exists(temp_frame):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return None, gr.update(visible=True, value=f"❌ Failed to extract first frame from video: {result.stderr}")

                img = Image.open(temp_frame)
                source_type = "video (first frame)"
            finally:
                # Clean up temp directory after loading image
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass

        # Handle image files
        elif file_ext in IMAGE_EXTENSIONS:
            img = Image.open(first_file)
            source_type = "image"

        else:
            return None, gr.update(visible=True, value=f"❌ Unsupported file type ({file_ext}). Please upload an image or video file.")

        # Perform upscaling (test uses image scale for both images and videos)
        result, orig = upscale_image(img, model, preserve_alpha,
                                    output_format, jpeg_quality, use_fp16,
                                    target_scale=image_target_scale,
                                    target_resolution=0,
                                    is_video_frame=False, **params)

        # Prepare for display (convert RGBA to RGB with white background if needed)
        orig_resized = orig.resize(result.size, Image.Resampling.LANCZOS)
        display_pair = (rgba_to_rgb_for_display(orig_resized), rgba_to_rgb_for_display(result))

        return display_pair, gr.update(visible=True, value=f"✅ Test completed - {Path(first_file).name} ({source_type}) - {orig.width}x{orig.height} → {result.width}x{result.height}")

    except Exception as e:
        return None, gr.update(visible=True, value=f"❌ Error: {str(e)}")

def create_app():
    global current_language

    with gr.Blocks(title="Anime Upscaler - Batch & Video Export") as app:
        # Use translations based on system locale
        t = TRANSLATIONS[current_language]

        gr.Markdown(f"# {t['title']}")
        gr.Markdown(f"*{t['subtitle']}*")

        with gr.Row():
            # Left Panel
            with gr.Column(scale=1):
                upload_accordion = gr.Accordion(t['upload_title'], open=True)
                with upload_accordion:
                    file_input = gr.File(
                        label=t['upload_label'],
                        file_types=["image", "video"],
                        type="filepath",
                        file_count="multiple"
                    )
                    file_summary = gr.Textbox(label=t['files_summary'], interactive=False, value=t['no_files'])

                model_accordion = gr.Accordion(t['ai_model_title'], open=True)
                with model_accordion:
                    model_select = gr.Radio(
                        list(MODELS.keys()),
                        value="Ani4K v2 Compact (Recommended)",
                        label=t['select_model']
                    )

                format_accordion = gr.Accordion(t['output_format_title'], open=True)
                with format_accordion:
                    # NOUVEAU: Sélecteur d'échelle pour images
                    image_scale_radio = gr.Radio(
                        choices=["×2", "×3", "×4"],
                        value="×2",
                        label=t['image_scale_label'],
                        info=t['image_scale_info']
                    )

                    with gr.Row():
                        output_format_select = gr.Dropdown(
                            ["PNG", "JPEG", "WebP"],
                            value=DEFAULT_UPSCALING_SETTINGS["output_format"],
                            label=t['final_output_format'],
                            scale=1
                        )
                        jpeg_quality_slider = gr.Slider(80, 100, DEFAULT_UPSCALING_SETTINGS["jpeg_quality"],
                                                       step=5, label=t['jpeg_quality'], scale=2)

                    frame_format_select = gr.Dropdown(
                        list(FRAME_FORMAT_OPTIONS.keys()),
                        value="PNG - Normal (8-bit)",
                        label=t['video_frame_format'],
                        info=t['video_frame_format_info']
                    )

                upscaling_accordion = gr.Accordion(t['upscaling_params_title'], open=True)
                with upscaling_accordion:
                    use_auto_settings = gr.Checkbox(
                        label=t['use_auto'],
                        value=True,
                        info=t['use_auto_info']
                    )

                    with gr.Group(visible=False) as custom_settings_group:
                        tile_settings_md = gr.Markdown(f"**{t['tile_settings']}**")
                        tile_size_slider = gr.Slider(
                            256, 1024, DEFAULT_UPSCALING_SETTINGS["tile_size"],
                            step=64, label=t['tile_size'],
                            info=t['tile_size_info']
                        )
                        tile_overlap_slider = gr.Slider(
                            16, 64, DEFAULT_UPSCALING_SETTINGS["tile_overlap"],
                            step=8, label=t['tile_overlap'],
                            info=t['tile_overlap_info']
                        )

                        post_processing_md = gr.Markdown(f"**{t['post_processing']}**")
                        sharpening_slider = gr.Slider(
                            0.0, 2.0, DEFAULT_UPSCALING_SETTINGS["sharpening"],
                            step=0.1, label=t['sharpening'],
                            info=t['sharpening_info']
                        )
                        contrast_slider = gr.Slider(
                            0.8, 1.2, DEFAULT_UPSCALING_SETTINGS["contrast"],
                            step=0.05, label=t['contrast'],
                            info=t['contrast_info']
                        )
                        saturation_slider = gr.Slider(
                            0.8, 1.2, DEFAULT_UPSCALING_SETTINGS["saturation"],
                            step=0.05, label=t['saturation'],
                            info=t['saturation_info']
                        )


                advanced_accordion = gr.Accordion(t['advanced_title'], open=False)
                with advanced_accordion:
                    precision_radio = gr.Radio(
                        choices=["FP16 (Half Precision)", "FP32 (Full Precision)"],
                        value="FP16 (Half Precision)" if DEFAULT_UPSCALING_SETTINGS["use_fp16"] else "FP32 (Full Precision)",
                        label=t['precision_mode'],
                        info=t['precision_info']
                    )

                video_accordion = gr.Accordion(t['video_export_title'], open=True)
                with video_accordion:
                    export_video_check = gr.Checkbox(label=t['export_videos'], value=True)

                    # NOUVEAU: Sélecteur de résolution cible pour vidéos
                    video_resolution_dropdown = gr.Dropdown(
                        choices=[
                            ("Auto (2x upscale)", 0),
                            ("HD 720p", 720),
                            ("Full HD 1080p", 1080),
                            ("QHD 1440p", 1440),
                            ("4K UHD 2160p", 2160),
                            ("8K UHD 4320p", 4320)
                        ],
                        value=0,
                        label=t['video_resolution_label'],
                        info=t['video_resolution_info']
                    )

                    codec_select = gr.Dropdown(
                        list(VIDEO_CODECS.keys()),
                        value=DEFAULT_EXPORT_SETTINGS["codec"],
                        label=t['video_codec']
                    )

                    profile_select = gr.Dropdown(
                        list(VIDEO_CODECS[DEFAULT_EXPORT_SETTINGS["codec"]]["profiles"].keys()),
                        value=DEFAULT_EXPORT_SETTINGS["profile"],
                        label=t['codec_profile']
                    )

                    fps_slider = gr.Slider(
                        0, 60, DEFAULT_EXPORT_SETTINGS["fps"],
                        step=1,
                        label=t['fps_label']
                    )

                    preserve_alpha_check = gr.Checkbox(
                        label=t['preserve_transparency'],
                        value=DEFAULT_EXPORT_SETTINGS["preserve_alpha"]
                    )

                    keep_audio_check = gr.Checkbox(
                        label=t['keep_audio'],
                        value=True,
                        info=t['keep_audio_info']
                    )

                    skip_duplicate_frames_check = gr.Checkbox(
                        label=t['skip_duplicates'],
                        value=False,
                        info=t['skip_duplicates_info']
                    )

                    video_naming_md = gr.Markdown(f"**{t['video_naming_title']}**")
                    video_naming_mode = gr.Radio(
                        choices=[t['naming_same'], t['naming_suffix'], t['naming_custom']],
                        value=t['naming_same'],
                        label=t['video_naming_label'],
                        info=t['video_naming_info']
                    )

                    with gr.Group(visible=False) as suffix_group:
                        video_suffix = gr.Textbox(
                            label=t['suffix_label'],
                            value="",
                            placeholder=t['suffix_placeholder'],
                            info=t['suffix_info']
                        )

                    with gr.Group(visible=False) as custom_name_group:
                        video_custom_name = gr.Textbox(
                            label=t['custom_name_label'],
                            value="",
                            placeholder=t['custom_name_placeholder'],
                            info=t['custom_name_info']
                        )

                auto_cleanup_md = gr.Markdown(f"**{t['auto_cleanup_title']}**")
                cleanup_accordion = gr.Accordion(t['auto_cleanup_accordion'], open=False)
                with cleanup_accordion:
                    auto_delete_input_frames = gr.Checkbox(
                        label=t['delete_input_frames'],
                        value=False,
                        info=t['delete_input_frames_info']
                    )
                    auto_delete_output_frames = gr.Checkbox(
                        label=t['delete_output_frames'],
                        value=False,
                        info=t['delete_output_frames_info']
                    )
                    auto_delete_frame_mapping = gr.Checkbox(
                        label=t['delete_mapping'],
                        value=True,
                        info=t['delete_mapping_info']
                    )
                    organize_videos_folder = gr.Checkbox(
                        label=t['organize_videos'],
                        value=True,
                        info=t['organize_videos_info']
                    )

                with gr.Row():
                    test_btn = gr.Button(t['test_btn'], variant="secondary", size="lg")
                    process_btn = gr.Button(t['run_batch_btn'], variant="primary", size="lg")

                test_status = gr.Textbox(label=t['test_status'], interactive=False, visible=False)

                gr.Markdown("---")
                with gr.Row():
                    pause_btn = gr.Button(t['pause_btn'], size="lg")
                    stop_btn = gr.Button(t['stop_btn'], variant="stop", size="lg")

            # Right Panel
            with gr.Column(scale=2):
                with gr.Tabs() as tabs:
                    with gr.TabItem(t['compare_tab']) as compare_tab:
                        comparison_slider = gr.ImageSlider(label=t['before_after'], type="numpy")
                        with gr.Row():
                            frame_slider = gr.Slider(1, 1, 1, step=1, label=t['frame_navigation'], visible=False, scale=4)
                            frame_label = gr.Textbox(value="", label="", interactive=False, visible=False, scale=1, show_label=False)

                    with gr.TabItem(t['gallery_tab']) as gallery_tab:
                        gallery = gr.Gallery(label=t['all_results'], columns=4, height=500, object_fit="contain")

                status = gr.Textbox(label=t['status_label'], interactive=False, lines=5)
                download_info = gr.Textbox(label=t['download_info'], interactive=False, lines=3)
                output_folder = gr.Textbox(label=t['output_folder'], interactive=False)

        info_help_accordion = gr.Accordion(t['info_help_title'], open=False)
        with info_help_accordion:
            gr.Markdown(f"""
### 🤖 AI Models (Upscale-Hub)

**10 specialized models** from [Upscale-Hub](https://github.com/Sirosky/Upscale-Hub):

| Famille | Modèle | Vitesse | Qualité | Usage recommandé |
|---------|--------|---------|---------|------------------|
| **Ani4K v2** | Compact (Recommandé) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Anime moderne (Bluray/WEB) - Équilibré |
| **Ani4K v2** | UltraCompact | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Anime moderne - Très rapide |
| **AniToon** | Small/Regular/Large | ⭐⭐⭐-⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐-⭐⭐⭐⭐⭐ | Anime 90s/2000s basse qualité |
| **AniSD** | AC/Regular | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Anime ancien (vieux anime) |
| **OpenProteus** | Compact | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Alternative Topaz Proteus |
| **AniScale2** | Compact | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Usage général rapide |

**🏆 Ani4K v2 Compact** est recommandé pour la plupart des usages: excellent équilibre vitesse/qualité pour anime moderne.

**➕ Ajouter vos propres modèles:**
- Placez vos modèles (.pth, .safetensors) dans le dossier `models/`
- Formats supportés: PyTorch (.pth), SafeTensors (.safetensors)
- Source: [OpenModelDB](https://openmodeldb.info/) ou [Upscale-Hub](https://github.com/Sirosky/Upscale-Hub)
- Les modèles sont détectés automatiquement au démarrage

### ⚙️ Paramètres d'Upscaling

**Tile Settings:**
- **Tile Size**: Taille des tuiles pour le traitement
  - 256px: GPU avec 4GB VRAM
  - 512px: GPU avec 8GB+ VRAM (recommandé)
  - 1024px: GPU puissants (12GB+)
  - Plus petit = moins de VRAM, plus lent
- **Tile Overlap**: Chevauchement entre tuiles (16-64px)
  - Plus grand = meilleur blending, plus lent
  - Plus petit = plus rapide, possibles artifacts

**Output Format:**
- **PNG**: Sans perte, grande taille, supporte transparence
- **JPEG**: Compression avec perte, petits fichiers, pas de transparence
- **WebP**: Meilleure compression, moderne, supporte transparence

**Post-Processing:**
- **Sharpening**: Accentuation après upscaling (0-2.0)
  - 0 = Aucun
  - 0.5-1.0 = Léger à modéré
  - 1.5-2.0 = Fort (attention aux artifacts)
- **Contrast**: Ajustement du contraste (0.8-1.2)
- **Saturation**: Ajustement de la saturation couleur (0.8-1.2)

**Advanced:**
- **FP16 (Half Precision)**: Utilise moins de VRAM, légèrement plus rapide
  - ✅ Recommandé pour CUDA
  - Précision légèrement réduite (imperceptible)

### 🎬 Codecs Vidéo
| Codec | Transparence | Qualité | Taille | Utilisation recommandée |
|-------|--------------|---------|--------|------------------------|
| **H.264 (AVC)** | ❌ Non | Bonne | Petite | Web, streaming, compatibilité maximale |
| **H.265 (HEVC)** | ❌ Non | Excellente | Très petite | Vidéo 4K, appareils modernes |
| **ProRes** | ✅ Oui (4444/XQ) | Excellente | Grande | Montage professionnel, VFX |
| **DNxHD/DNxHR** | ✅ Oui (444) | Excellente | Grande | Montage professionnel, broadcast |

**FPS (Frames Per Second)**:
- 0 = Préserver le FPS original de la vidéo
- 24/30/60 = Forcer un FPS spécifique

**Preserve Transparency (Alpha Channel)**:
- ✅ Activé: Le canal alpha est copié de l'image originale vers la sortie
- Fonctionne avec tous les formats d'images
- Pour les vidéos: nécessite ProRes 4444/XQ ou DNxHR 444

### 🗑️ Auto-Cleanup (Économie d'Espace) **NOUVEAU v2.2**

**Delete input frames after processing:**
- Supprime les frames extraites au fur et à mesure du traitement
- Économise jusqu'à 50% d'espace pendant le traitement
- Recommandé si vous n'avez pas besoin des frames originales

**Delete upscaled frames after encoding:**
- Supprime les frames upscalées après encodage vidéo réussi
- Économise jusqu'à 90% d'espace final (garde uniquement la vidéo)
- Recommandé pour usage normal

**Delete frame mapping file after processing:**
- Supprime le fichier `frame_mapping.json` (données de détection des duplicatas)
- Activé par défaut car ce fichier est uniquement utile pour le débogage
- Économise un peu d'espace disque

**Export videos to dedicated videos/ folder:**
- ✅ Activé (défaut): Vidéos exportées dans `output/videos/nom_video.mp4`
- ❌ Désactivé: Vidéos dans `output/session/nom_video/...` (avec frames)

**💡 Recommandations:**
- Usage normal: Activez les 2 options de suppression → garde uniquement vidéos finales
- Archivage: Désactivez tout → conserve toutes les frames
- Économie progressive: Activez uniquement "Delete input frames"

### 📁 Organisation des Fichiers

**Images:**
- 1 seule: `output/session/nom_upscaled.ext`
- Plusieurs: `output/session/images/nom_upscaled.ext`

**Vidéos (avec "Export to dedicated videos/ folder" activé - défaut):**
- Vidéo finale: `output/videos/nom_video.mp4` (directement accessible)
- Frames temporaires: `output/session/temp_video_processing/nom_video/...` (supprimées automatiquement si auto-cleanup activé)

**Vidéos (avec "Export to dedicated videos/ folder" désactivé):**
- 1 seule: `output/session/nom_video/nom_video_upscaled.mp4`
- Plusieurs: `output/session/videos/nom_video/nom_video_upscaled.mp4`

**Nommage des vidéos exportées:**
- **Same as input**: `video.mp4` → `video.mp4` (remplace l'extension)
- **Add suffix**: `video.mp4` → `video_upscaled.mp4` (suffixe personnalisable)
- **Custom name**: Nom complètement personnalisé (utile pour batch de vidéos)

### 💡 Conseils
- **🧪 Test**: Testez avec le premier fichier uploadé (image ou vidéo) avant batch complet
- **Ani4K v2 Compact** recommandé pour anime moderne (meilleur équilibre vitesse/qualité)
- Les modèles d'upscaling traitent uniquement RGB, le canal alpha est copié séparément
- Utilisez le sharpening avec modération pour éviter les artifacts
- Activez Auto-Cleanup pour économiser de l'espace disque (surtout pour vidéos)
- Format JPEG/WebP recommandé pour réduire la taille (qualité 90-95)
- Pour les vidéos sans transparence, H.264 ou H.265 offrent la meilleure compression

### 🖥️ Système
**GPU:** {'✅ CUDA (' + torch.cuda.get_device_name(0) + ')' if DEVICE == 'cuda' else '❌ CPU (très lent)'}
**Modèles détectés:** {len(MODELS)}
""")

        # Event handlers
        file_input.change(show_file_summary, [file_input], [file_summary])

        codec_select.change(
            update_codec_profiles,
            [codec_select],
            [profile_select]
        )

        # Toggle custom settings visibility
        use_auto_settings.change(
            lambda auto: gr.update(visible=not auto),
            [use_auto_settings],
            [custom_settings_group]
        )

        # Toggle video naming controls visibility
        def update_naming_visibility(mode):
            t_curr = TRANSLATIONS[current_language]
            return (
                gr.update(visible=(mode == t_curr['naming_suffix'])),
                gr.update(visible=(mode == t_curr['naming_custom']))
            )

        video_naming_mode.change(
            update_naming_visibility,
            [video_naming_mode],
            [suffix_group, custom_name_group]
        )

        test_btn.click(
            test_image_upscale,
            [file_input, model_select, image_scale_radio, video_resolution_dropdown, output_format_select, jpeg_quality_slider,
             precision_radio, preserve_alpha_check, use_auto_settings, tile_size_slider, tile_overlap_slider,
             sharpening_slider, contrast_slider, saturation_slider],
            [comparison_slider, test_status]
        )

        process_btn.click(
            process_batch,
            [file_input, model_select, image_scale_radio, video_resolution_dropdown, output_format_select, jpeg_quality_slider,
             precision_radio,
             codec_select, profile_select, fps_slider, preserve_alpha_check, export_video_check, keep_audio_check, frame_format_select,
             auto_delete_input_frames, auto_delete_output_frames, auto_delete_frame_mapping, organize_videos_folder, skip_duplicate_frames_check,
             use_auto_settings, tile_size_slider, tile_overlap_slider, sharpening_slider, contrast_slider, saturation_slider,
             video_naming_mode, video_suffix, video_custom_name],
            [comparison_slider, gallery, status, output_folder, frame_slider, frame_label, download_info]
        )

        frame_slider.change(
            navigate_frame,
            [frame_slider],
            [comparison_slider, frame_label]
        )

        stop_btn.click(stop_processing, outputs=[status])
        pause_btn.click(pause_processing, outputs=[pause_btn, status])

    return app

if __name__ == "__main__":
    # Fix Windows console encoding for emojis
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass

    print(f"🚀 Starting on {DEVICE}")
    if DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   VRAM: {total_vram:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")

        # Check if torch.compile is available
        if hasattr(torch, 'compile'):
            print(f"   ℹ️ torch.compile available (requires Triton on Linux)")
            # Check if on Windows
            import platform
            if platform.system() == "Windows":
                print(f"   ℹ️ Note: torch.compile may not work on Windows (Triton limitation)")
        else:
            print(f"   ⚠️ torch.compile not available - upgrade PyTorch 2.0+ for potential speedup")
    else:
        print("   ⚠️ CUDA not available - running on CPU (very slow)")

    print(f"📂 Output: {OUTPUT_DIR}")

    # Show detected language
    lang_name = "Français" if current_language == "fr" else "English"
    print(f"🌐 Language: {lang_name} (detected from system locale)")

    # Pre-load default model with optimizations
    try:
        print("\n🔧 Pre-loading default model with optimizations...")
        load_model("Ani4K v2 Compact (Recommended)", use_fp16=True)
        if DEVICE == "cuda":
            print(f"   {get_gpu_memory_info()}")
    except Exception as e:
        print(f"⚠️ Loading error: {e}")

    app = create_app()

    # Try ports 7860-7869
    import socket
    for port in range(7860, 7870):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
            print(f"🌐 Using port {port}")
            app.launch(server_name="0.0.0.0", server_port=port, inbrowser=True)
            break
        except OSError:
            print(f"⚠️ Port {port} busy, trying next...")
            continue
