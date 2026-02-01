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
    app = create_app(vram_manager=vram_manager, vram_info_text=vram_info_text, recommended_workers=recommended_jobs)

    # Launch
    print("🌐 Starting server...")

    # Try ports 7860-7880 with retry logic
    def launch_with_dynamic_port(app, start_port=7860, max_attempts=20):
        """Try to launch on available ports, retrying if a port is busy"""
        for i in range(max_attempts):
            port = start_port + i
            try:
                # Check if port is available
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(('127.0.0.1', port))
                    if result == 0:
                        # Port is in use, try next
                        print(f"⚠️ Port {port} is in use, trying next...")
                        continue
                
                print(f"✅ Using port {port}")
                app.launch(
                    server_name="127.0.0.1",
                    server_port=port,
                    inbrowser=True,
                    show_error=True,
                    css=getattr(app, 'custom_css', None)
                )
                return True
            except OSError as e:
                print(f"⚠️ Port {port} failed: {e}, trying next...")
                continue
        
        # If all ports failed, let Gradio choose automatically
        print("⚠️ No free port found, using automatic port selection...")
        app.launch(
            server_name="127.0.0.1",
            inbrowser=True,
            show_error=True,
            css=getattr(app, 'custom_css', None)
        )
        return True

    launch_with_dynamic_port(app)


if __name__ == "__main__":
    main()
