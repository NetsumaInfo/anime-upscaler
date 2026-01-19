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

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.flv'}

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Video export codec configurations
VIDEO_CODECS = {
    "H.264 (AVC)": {
        "codec": "libx264",
        "alpha_support": False,
        "profiles": {
            "Main": {"preset": "medium", "crf": 23, "profile": "main"},
            "High": {"preset": "medium", "crf": 23, "profile": "high"},
            "High 10-bit": {"preset": "medium", "crf": 23, "profile": "high10"},
        }
    },
    "H.265 (HEVC)": {
        "codec": "libx265",
        "alpha_support": False,
        "profiles": {
            "Main (8-bit)": {"preset": "medium", "crf": 28, "profile": "main", "pix_fmt": "yuv420p"},
            "Main10 (10-bit)": {"preset": "medium", "crf": 28, "profile": "main10", "pix_fmt": "yuv420p10le"},
            "Main12 (12-bit)": {"preset": "medium", "crf": 28, "profile": "main12", "pix_fmt": "yuv420p12le"},
            "Main 4:4:4 10-bit": {"preset": "medium", "crf": 28, "profile": "main444-10", "pix_fmt": "yuv444p10le"},
            "Main10 High Quality": {"preset": "slow", "crf": 22, "profile": "main10", "pix_fmt": "yuv420p10le"},
            "Main10 Fast": {"preset": "fast", "crf": 28, "profile": "main10", "pix_fmt": "yuv420p10le"},
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
    "profile": "High",
    "fps": 0,  # 0 = use original FPS
    "preserve_alpha": True
}

# Frame intermediate format settings
FRAME_FORMAT_OPTIONS = {
    "PNG - Uncompressed (16-bit)": {"format": "PNG", "compress_level": 0, "bits": 16},
    "PNG - Normal (8-bit)": {"format": "PNG", "compress_level": 6, "bits": 8},
    "PNG - High Compression (8-bit)": {"format": "PNG", "compress_level": 9, "bits": 8},
    "JPEG - Quality 100%": {"format": "JPEG", "quality": 100},
    "JPEG - Quality 95%": {"format": "JPEG", "quality": 95},
}

# Default upscaling settings
DEFAULT_UPSCALING_SETTINGS = {
    "tile_size": 512,
    "tile_overlap": 32,
    "output_format": "PNG",
    "jpeg_quality": 95,
    "sharpening": 0.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "use_fp16": True
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

def load_model(model_name: str):
    """Load model using Spandrel"""
    global loaded_models
    
    if model_name in loaded_models:
        return loaded_models[model_name]
    
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
    
    if DEVICE == "cuda":
        try:
            model = model.half()
        except:
            pass
    
    loaded_models[model_name] = model
    print(f"✅ {model_name} loaded on {DEVICE}")
    
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

def upscale_image(img: Image.Image, model_name: str, tile_size: int = 512, tile_overlap: int = 32,
                 preserve_alpha: bool = False, output_format: str = "PNG", jpeg_quality: int = 95,
                 sharpening: float = 0.0, contrast: float = 1.0, saturation: float = 1.0, use_fp16: bool = True):
    """Upscale image with tile-based processing and post-processing"""
    model = load_model(model_name)
    scale = MODELS[model_name]["scale"]

    # Store original alpha channel if present
    original_alpha = None
    if img.mode in ('RGBA', 'LA') and preserve_alpha:
        original_alpha = img.getchannel('A')

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    # Use FP16 if enabled and CUDA available
    if DEVICE == "cuda" and use_fp16:
        try:
            img_tensor = img_tensor.half()
        except:
            pass

    img_tensor = img_tensor.to(DEVICE)

    h, w = img_np.shape[:2]

    # For small images, process directly
    if h * w <= tile_size * tile_size:
        with torch.no_grad():
            output = model(img_tensor)
        output = output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        result_img = Image.fromarray(output)

        # Apply post-processing
        result_img = apply_post_processing(result_img, sharpening, contrast, saturation)

        # Apply preserved alpha channel
        if original_alpha is not None:
            upscaled_alpha = original_alpha.resize((result_img.width, result_img.height), Image.Resampling.LANCZOS)
            result_img = result_img.convert('RGBA')
            result_img.putalpha(upscaled_alpha)

        return result_img, img

    # Tile-based processing for large images
    result = np.zeros((h * scale, w * scale, 3), dtype=np.float32)
    weight = np.zeros((h * scale, w * scale, 3), dtype=np.float32)

    for y in range(0, h, tile_size - tile_overlap):
        for x in range(0, w, tile_size - tile_overlap):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = img_np[y:y_end, x:x_end]

            tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0)
            if DEVICE == "cuda" and use_fp16:
                try:
                    tile_tensor = tile_tensor.half()
                except:
                    pass
            tile_tensor = tile_tensor.to(DEVICE)

            with torch.no_grad():
                tile_output = model(tile_tensor)

            tile_output = tile_output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()

            y_out, x_out = y * scale, x * scale
            th, tw = tile_output.shape[:2]

            result[y_out:y_out+th, x_out:x_out+tw] += tile_output
            weight[y_out:y_out+th, x_out:x_out+tw] += 1

    result = result / np.maximum(weight, 1)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    result_img = Image.fromarray(result)

    # Apply post-processing
    result_img = apply_post_processing(result_img, sharpening, contrast, saturation)

    # Apply preserved alpha channel
    if original_alpha is not None:
        upscaled_alpha = original_alpha.resize((result_img.width, result_img.height), Image.Resampling.LANCZOS)
        result_img = result_img.convert('RGBA')
        result_img.putalpha(upscaled_alpha)

    return result_img, img

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

def extract_frames(video: str, out_dir: str):
    """Extract frames from video with alpha channel support"""
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", video,
        "-pix_fmt", "rgba",  # Support alpha channel
        "-start_number", "0",
        os.path.join(out_dir, "frame_%05d.png"), "-y"
    ], capture_output=True)
    return sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith('.png')])

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
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
    ]

    # Add original video as audio source if keep_audio is enabled
    if keep_audio and original_video_path:
        cmd.extend(["-i", original_video_path])

    # Codec-specific settings
    if codec_name == "H.264 (AVC)":
        cmd.extend([
            "-c:v", codec,
            "-preset", profile_config["preset"],
            "-crf", str(profile_config["crf"]),
            "-profile:v", profile_config["profile"],
            "-pix_fmt", "yuv420p"
        ])

    elif codec_name == "H.265 (HEVC)":
        pix_fmt = profile_config.get("pix_fmt", "yuv420p")
        cmd.extend([
            "-c:v", codec,
            "-preset", profile_config["preset"],
            "-crf", str(profile_config["crf"]),
            "-tag:v", "hvc1",  # For better compatibility
        ])
        if "profile" in profile_config:
            cmd.extend(["-profile:v", profile_config["profile"]])
        cmd.extend(["-pix_fmt", pix_fmt])

    elif codec_name == "ProRes":
        pix_fmt = profile_config["pix_fmt"] if has_alpha_support else "yuv422p10le"
        cmd.extend([
            "-c:v", codec,
            "-profile:v", profile_config["profile"],
            "-pix_fmt", pix_fmt,
            "-vendor", "apl0"
        ])

    elif codec_name == "DNxHD/DNxHR":
        profile = profile_config["profile"]
        if "dnxhr" in profile:
            cmd.extend(["-c:v", profile])
            if has_alpha_support and "444" in profile:
                cmd.extend(["-pix_fmt", "yuva444p10le"])
            else:
                cmd.extend(["-pix_fmt", "yuv422p10le"])
        else:
            cmd.extend([
                "-c:v", "dnxhd",
                "-b:v", profile_config["bitrate"],
                "-pix_fmt", "yuv422p"
            ])

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
    """Save image with specified format"""
    if output_format == "JPEG":
        # Convert RGBA to RGB for JPEG
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else img.split()[1])
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(path.with_suffix('.jpg'), 'JPEG', quality=jpeg_quality, optimize=True)
    elif output_format == "WebP":
        img.save(path.with_suffix('.webp'), 'WebP', quality=jpeg_quality, method=6)
    else:  # PNG (default)
        img.save(path.with_suffix('.png'), 'PNG', optimize=True)

def process_batch(files, model, tile_size, tile_overlap, output_format, jpeg_quality, sharpening, contrast,
                 saturation, use_fp16, codec_name, profile_name, fps, preserve_alpha, export_video, keep_audio, frame_format,
                 auto_delete_input_frames, auto_delete_output_frames, organize_videos_folder, progress=gr.Progress()):
    """Process multiple files with video export support and auto-cleanup"""
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
            result, orig = upscale_image(img, model, tile_size, tile_overlap, preserve_alpha,
                                        output_format, jpeg_quality, sharpening, contrast, saturation, use_fp16)

            img_name = Path(img_path).stem
            output_path = img_session / f"{img_name}_upscaled"
            save_image_with_format(result, output_path, output_format, jpeg_quality)
            all_results.append(rgba_to_rgb_for_display(result))

            # Store for comparison with white background for display
            orig_resized = orig.resize(result.size, Image.Resampling.LANCZOS)
            frame_pairs.append((rgba_to_rgb_for_display(orig_resized), rgba_to_rgb_for_display(result)))

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
                vid_session = session / "videos" / vid_name
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

            # Extract frames
            progress(0.05, desc=f"Extracting frames from {vid_name}...")
            frames = extract_frames(video_path, str(frames_in))
            total_frames = len(frames)

            if not total_frames:
                status_messages.append(f"❌ {vid_name}: No frames extracted")
                continue

            # Get original FPS
            original_fps = get_video_fps(video_path) if fps == 0 else fps

            # Process frames
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

                img = Image.open(fp)
                result, orig = upscale_image(img, model, tile_size, tile_overlap, preserve_alpha,
                                            output_format, jpeg_quality, sharpening, contrast, saturation, use_fp16)
                # Save frame with chosen intermediate format
                frame_path = frames_out / f"frame_{i:05d}"
                save_frame_with_format(result, frame_path, frame_format)
                all_results.append(rgba_to_rgb_for_display(result))

                # Store pair for navigation with white background for display
                orig_resized = orig.resize(result.size, Image.Resampling.LANCZOS)
                frame_pairs.append((rgba_to_rgb_for_display(orig_resized), rgba_to_rgb_for_display(result)))

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
                video_output = vid_session / f"{vid_name}_upscaled{ext}"

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
                    status_messages.append(f"✅ {vid_name}: Video exported ({codec_name} - {profile_name})")
                    download_files.append(str(video_output))

                    # Auto-delete upscaled frames after successful encoding (if enabled)
                    if auto_delete_output_frames:
                        try:
                            import shutil
                            shutil.rmtree(frames_out)
                            status_messages.append(f"🗑️ {vid_name}: Cleaned up upscaled frames")
                        except Exception as e:
                            status_messages.append(f"⚠️ {vid_name}: Failed to delete upscaled frames: {e}")
                else:
                    status_messages.append(f"⚠️ {vid_name}: {result_msg}")

            # Auto-delete input frames folder if it's empty or if auto-delete was enabled
            if auto_delete_input_frames:
                try:
                    import shutil
                    if frames_in.exists():
                        shutil.rmtree(frames_in)
                        status_messages.append(f"🗑️ {vid_name}: Cleaned up input frames")
                except Exception as e:
                    print(f"⚠️ Failed to delete input frames folder: {e}")

            status_messages.append(f"✅ {vid_name}: {total_frames} frames processed")

    processing_state["running"] = False
    progress(1.0)

    # Prepare outputs
    first_pair = frame_pairs[0] if frame_pairs else (None, None)
    final_status = "\n".join(status_messages)

    # Create download links text
    download_text = ""
    if download_files:
        download_text = "📥 Files ready:\n" + "\n".join([f"• {Path(f).name}" for f in download_files])

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
    """Display summary of uploaded files"""
    if not files:
        return "No files uploaded"

    images, videos = separate_files_by_type(files)

    summary = []
    if images:
        summary.append(f"📸 {len(images)} image(s)")
    if videos:
        summary.append(f"🎬 {len(videos)} video(s)")

    return " | ".join(summary) if summary else "No valid files"

def test_image_upscale(uploaded_files, model, tile_size, tile_overlap, output_format, jpeg_quality,
                       sharpening, contrast, saturation, use_fp16, preserve_alpha):
    """Quick test upscale on the first uploaded file (image or video first frame) for adjusting model parameters"""
    if not uploaded_files or len(uploaded_files) == 0:
        return None, gr.update(visible=True, value="❌ No files uploaded. Please upload images/videos first.")

    # Get first file
    first_file = uploaded_files[0] if isinstance(uploaded_files, list) else uploaded_files
    file_ext = Path(first_file).suffix.lower()

    try:
        # Handle video files - extract first frame
        if file_ext in VIDEO_EXTENSIONS:
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract only the first frame
                temp_frame = os.path.join(temp_dir, "frame_00000.png")
                result = subprocess.run([
                    "ffmpeg", "-i", first_file,
                    "-vframes", "1",  # Extract only first frame
                    "-pix_fmt", "rgba",
                    "-start_number", "0",
                    temp_frame, "-y"
                ], capture_output=True, text=True)

                if result.returncode != 0 or not os.path.exists(temp_frame):
                    return None, gr.update(visible=True, value=f"❌ Failed to extract first frame from video: {result.stderr}")

                img = Image.open(temp_frame)
                source_type = "video (first frame)"

        # Handle image files
        elif file_ext in IMAGE_EXTENSIONS:
            img = Image.open(first_file)
            source_type = "image"

        else:
            return None, gr.update(visible=True, value=f"❌ Unsupported file type ({file_ext}). Please upload an image or video file.")

        # Perform upscaling
        result, orig = upscale_image(img, model, tile_size, tile_overlap, preserve_alpha,
                                    output_format, jpeg_quality, sharpening, contrast, saturation, use_fp16)

        # Prepare for display (convert RGBA to RGB with white background if needed)
        orig_resized = orig.resize(result.size, Image.Resampling.LANCZOS)
        display_pair = (rgba_to_rgb_for_display(orig_resized), rgba_to_rgb_for_display(result))

        return display_pair, gr.update(visible=True, value=f"✅ Test completed - {Path(first_file).name} ({source_type}) - {orig.width}x{orig.height} → {result.width}x{result.height}")

    except Exception as e:
        return None, gr.update(visible=True, value=f"❌ Error: {str(e)}")

def create_app():
    with gr.Blocks(title="Anime Upscaler - Batch & Video Export") as app:

        gr.Markdown("# 🎨 Anime Upscaler - Batch & Video Export")
        gr.Markdown("*2x upscaling for anime and cartoons — Multi-file batch processing with professional video export*")

        with gr.Row():
            # Left Panel
            with gr.Column(scale=1):
                with gr.Accordion("📁 Upload Images/Videos (Multiple files supported)", open=True):
                    file_input = gr.File(
                        label="Drop files here or click to browse",
                        file_types=["image", "video"],
                        type="filepath",
                        file_count="multiple"
                    )
                    file_summary = gr.Textbox(label="Files Summary", interactive=False, value="No files uploaded")

                with gr.Accordion("🤖 AI Model", open=True):
                    model_select = gr.Radio(
                        list(MODELS.keys()),
                        value="Ani4K v2 Compact (Recommended)",
                        label="Select Model"
                    )

                with gr.Accordion("⚙️ Output Format, Style & Size", open=True):
                    tile_slider = gr.Slider(256, 1024, DEFAULT_UPSCALING_SETTINGS["tile_size"],
                                          step=128, label="Tile Size (VRAM)")
                    tile_overlap_slider = gr.Slider(16, 64, DEFAULT_UPSCALING_SETTINGS["tile_overlap"],
                                                   step=8, label="Tile Overlap")

                    with gr.Row():
                        output_format_select = gr.Dropdown(
                            ["PNG", "JPEG", "WebP"],
                            value=DEFAULT_UPSCALING_SETTINGS["output_format"],
                            label="Final Output Format",
                            scale=1
                        )
                        jpeg_quality_slider = gr.Slider(80, 100, DEFAULT_UPSCALING_SETTINGS["jpeg_quality"],
                                                       step=5, label="JPEG/WebP Quality", scale=2)

                    frame_format_select = gr.Dropdown(
                        list(FRAME_FORMAT_OPTIONS.keys()),
                        value="PNG - Normal (8-bit)",
                        label="🎬 Video Frame Intermediate Format",
                        info="Format used when saving upscaled frames before video encoding"
                    )

                with gr.Accordion("🎨 Post-Processing", open=False):
                    sharpening_slider = gr.Slider(0.0, 2.0, DEFAULT_UPSCALING_SETTINGS["sharpening"],
                                                 step=0.1, label="Sharpening (0=None)")
                    contrast_slider = gr.Slider(0.8, 1.2, DEFAULT_UPSCALING_SETTINGS["contrast"],
                                               step=0.05, label="Contrast")
                    saturation_slider = gr.Slider(0.8, 1.2, DEFAULT_UPSCALING_SETTINGS["saturation"],
                                                 step=0.05, label="Saturation")

                with gr.Accordion("⚡ Advanced", open=False):
                    use_fp16_check = gr.Checkbox(label="Use FP16 (Half Precision)",
                                                value=DEFAULT_UPSCALING_SETTINGS["use_fp16"],
                                                info="Faster on CUDA, slightly lower precision")

                with gr.Accordion("🎬 Video Export Settings", open=True):
                    export_video_check = gr.Checkbox(label="Export videos (not just frames)", value=True)

                    codec_select = gr.Dropdown(
                        list(VIDEO_CODECS.keys()),
                        value=DEFAULT_EXPORT_SETTINGS["codec"],
                        label="Video Codec"
                    )

                    profile_select = gr.Dropdown(
                        list(VIDEO_CODECS[DEFAULT_EXPORT_SETTINGS["codec"]]["profiles"].keys()),
                        value=DEFAULT_EXPORT_SETTINGS["profile"],
                        label="Codec Profile"
                    )

                    fps_slider = gr.Slider(
                        0, 60, DEFAULT_EXPORT_SETTINGS["fps"],
                        step=1,
                        label="FPS (0 = use original)"
                    )

                    preserve_alpha_check = gr.Checkbox(
                        label="Preserve transparency (if supported)",
                        value=DEFAULT_EXPORT_SETTINGS["preserve_alpha"]
                    )

                    keep_audio_check = gr.Checkbox(
                        label="🔊 Keep audio from original video",
                        value=True,
                        info="Copy audio track from source video to upscaled output"
                    )

                gr.Markdown("### 🗑️ Auto-Cleanup (Save Space)")
                with gr.Accordion("Auto-Deletion Settings", open=False):
                    auto_delete_input_frames = gr.Checkbox(
                        label="🗑️ Delete input frames after processing",
                        value=False,
                        info="Automatically delete extracted input frames after video is upscaled"
                    )
                    auto_delete_output_frames = gr.Checkbox(
                        label="🗑️ Delete upscaled frames after encoding",
                        value=False,
                        info="Automatically delete upscaled frames after video is successfully encoded"
                    )
                    organize_videos_folder = gr.Checkbox(
                        label="📁 Organize videos in videos/ folder",
                        value=True,
                        info="Always put videos in a 'videos' subfolder for consistent organization"
                    )

                with gr.Row():
                    test_btn = gr.Button("🧪 Test", variant="secondary", size="lg")
                    process_btn = gr.Button("▶️ Run Batch", variant="primary", size="lg")

                test_status = gr.Textbox(label="Test Status", interactive=False, visible=False)

                gr.Markdown("---")
                with gr.Row():
                    pause_btn = gr.Button("⏸️ Pause", size="lg")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")

            # Right Panel
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("⚖️ Compare"):
                        comparison_slider = gr.ImageSlider(label="Before / After", type="numpy")
                        with gr.Row():
                            frame_slider = gr.Slider(1, 1, 1, step=1, label="🎞️ Frame Navigation", visible=False, scale=4)
                            frame_label = gr.Textbox(value="", label="", interactive=False, visible=False, scale=1, show_label=False)

                    with gr.TabItem("🖼️ Gallery"):
                        gallery = gr.Gallery(label="All Results", columns=4, height=500, object_fit="contain")

                status = gr.Textbox(label="Status", interactive=False, lines=5)
                download_info = gr.Textbox(label="Download Info", interactive=False, lines=3)
                output_folder = gr.Textbox(label="📂 Output Folder", interactive=False)

        with gr.Accordion("ℹ️ Info & Help", open=False):
            gr.Markdown(f"""
### 🤖 AI Models (Upscale-Hub)

**10 modèles spécialisés** depuis [Upscale-Hub](https://github.com/Sirosky/Upscale-Hub):

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

**Organize videos in videos/ folder:**
- ✅ Activé (défaut): Toutes les vidéos vont dans `videos/` (cohérence)
- ❌ Désactivé: Organisation intelligente (1 vidéo = pas de sous-dossier)

**💡 Recommandations:**
- Usage normal: Activez les 2 options de suppression → garde uniquement vidéos finales
- Archivage: Désactivez tout → conserve toutes les frames
- Économie progressive: Activez uniquement "Delete input frames"

### 📁 Organisation des Fichiers

**Images:**
- 1 seule: `output/session/nom_upscaled.ext`
- Plusieurs: `output/session/images/nom_upscaled.ext`

**Vidéos (avec "Organize videos" activé - défaut):**
- Toujours: `output/session/videos/nom_video/...`

**Vidéos (avec "Organize videos" désactivé):**
- 1 seule: `output/session/nom_video/...`
- Plusieurs: `output/session/videos/nom_video/...`

Chaque vidéo contient (si auto-cleanup désactivé):
- `input/`: Frames originales extraites
- `output/`: Frames upscalées
- `nom_video_upscaled.mp4/.mov`: Vidéo encodée

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

        test_btn.click(
            test_image_upscale,
            [file_input, model_select, tile_slider, tile_overlap_slider, output_format_select, jpeg_quality_slider,
             sharpening_slider, contrast_slider, saturation_slider, use_fp16_check, preserve_alpha_check],
            [comparison_slider, test_status]
        )

        process_btn.click(
            process_batch,
            [file_input, model_select, tile_slider, tile_overlap_slider, output_format_select, jpeg_quality_slider,
             sharpening_slider, contrast_slider, saturation_slider, use_fp16_check,
             codec_select, profile_select, fps_slider, preserve_alpha_check, export_video_check, keep_audio_check, frame_format_select,
             auto_delete_input_frames, auto_delete_output_frames, organize_videos_folder],
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
    print(f"🚀 Starting on {DEVICE}")
    if DEVICE == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"📂 Output: {OUTPUT_DIR}")
    
    # Pre-load default model
    try:
        load_model("Ani4K v2 Compact (Recommended)")
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
