"""
Main Entry Point for Anime Upscaler
Launches the Gradio web interface.
"""

import sys
import socket
import platform
import torch

from app_upscale.config import OUTPUT_DIR, DEVICE
from app_upscale.models import scan_models, VRAMManager, get_gpu_vram_gb, load_model
from app_upscale.gpu import get_gpu_memory_info
from app_upscale.ui import create_app
import app_upscale.state as state_module


def main():
    """Main entry point for the application"""
    # Fix Windows console encoding for emojis
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass

    print("🚀 Anime Upscaler - Starting...")

    # Display GPU info (simplified)
    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU: {gpu_name} ({total_vram:.1f} GB)")
    else:
        print("   ⚠️ Running on CPU (very slow)")

    # Scan models
    print("🔍 Scanning models...")
    models_dict, display_mapping = scan_models()

    # Initialize VRAM manager
    total_vram_gb = get_gpu_vram_gb()
    vram_manager = VRAMManager()
    recommended_jobs = vram_manager.auto_calculate_slots(total_vram_gb)
    vram_manager.update_max_jobs(recommended_jobs)

    if state_module.current_language == "fr":
        vram_info_text = f"**Détecté automatiquement:** {recommended_jobs} jobs parallèles (VRAM: {total_vram_gb}GB)"
    else:
        vram_info_text = f"**Auto-detected:** {recommended_jobs} parallel jobs (VRAM: {total_vram_gb}GB)"

    # Pre-load default model
    try:
        print("🔧 Loading default model...")
        model, actual_fp16, scale = load_model("Ani4K v2 Compact (Recommended)", use_fp16=True)
    except Exception as e:
        print(f"⚠️ Loading error: {e}")

    # Create Gradio app
    print("🎨 Creating interface...")
    app = create_app(vram_manager=vram_manager, vram_info_text=vram_info_text)

    # Launch
    print("🌐 Starting server...")

    # Try ports 7860-7869 until we find one available
    def find_free_port(start_port=7860, end_port=7869):
        """Find the first available port in the range"""
        for port in range(start_port, end_port + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        return None

    # Find available port
    port = find_free_port()

    # Launch Gradio server with automatic browser opening
    if port is None:
        print("⚠️ No free port found in range 7860-7869, using automatic port selection...")
        app.launch(
            server_name="127.0.0.1",
            inbrowser=True,
            show_error=True
        )
    else:
        print(f"✅ Using port {port}")
        app.launch(
            server_name="127.0.0.1",
            server_port=port,
            inbrowser=True,
            show_error=True
        )


if __name__ == "__main__":
    main()
