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

    print("🚀 Anime Upscaler v2.7.0 - Starting...")
    print(f"🚀 Starting on {DEVICE}")

    # Display GPU/CUDA diagnostics
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
            if platform.system() == "Windows":
                print(f"   ℹ️ Note: torch.compile may not work on Windows (Triton limitation)")
        else:
            print(f"   ⚠️ torch.compile not available - upgrade PyTorch 2.0+ for potential speedup")
    else:
        print("   ⚠️ CUDA not available - running on CPU (very slow)")

    print(f"📂 Output: {OUTPUT_DIR}")

    # Show detected language
    lang_name = "Français" if state_module.current_language == "fr" else "English"
    print(f"🌐 Language: {lang_name} (detected from system locale)")

    # Scan models
    print("\n🔍 Scanning models...")
    models_dict, display_mapping = scan_models()
    print(f"   Found {len(models_dict)} model(s)")

    # Initialize VRAM manager for parallel processing
    print("\n⚡ Initializing parallel processing...")
    total_vram_gb = get_gpu_vram_gb()
    vram_manager = VRAMManager()
    recommended_jobs = vram_manager.auto_calculate_slots(total_vram_gb)
    vram_manager.update_max_jobs(recommended_jobs)

    if state_module.current_language == "fr":
        vram_info_text = f"**Détecté automatiquement:** {recommended_jobs} jobs parallèles (VRAM: {total_vram_gb}GB)"
    else:
        vram_info_text = f"**Auto-detected:** {recommended_jobs} parallel jobs (VRAM: {total_vram_gb}GB)"

    print(f"   Workers: {recommended_jobs} parallel jobs")
    print(f"   VRAM: {total_vram_gb}GB")

    # Pre-load default model with optimizations
    try:
        print("\n🔧 Pre-loading default model with optimizations...")
        model, actual_fp16, scale = load_model("Ani4K v2 Compact (Recommended)", use_fp16=True)
        if DEVICE == "cuda":
            print(f"   {get_gpu_memory_info()}")
    except Exception as e:
        print(f"⚠️ Loading error: {e}")

    # Create Gradio app
    print("\n🎨 Creating UI...")
    app = create_app(vram_manager=vram_manager, vram_info_text=vram_info_text)

    # Launch with automatic browser opening and port selection
    print("\n🌐 Launching web interface...")

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
    if port is None:
        print("⚠️ Warning: No free port found in range 7860-7869, using random port")
        port = 0  # Let Gradio choose
    else:
        print(f"   Port: {port}")

    # Launch Gradio server with automatic browser opening
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        inbrowser=True,  # Gradio handles browser opening automatically
        show_error=True
    )


if __name__ == "__main__":
    main()
