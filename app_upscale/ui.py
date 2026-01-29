"""
Gradio UI Module
Creates and manages the complete user interface.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import gradio as gr
import torch

from .config import (
    VIDEO_CODECS, DEFAULT_UPSCALING_SETTINGS, DEFAULT_EXPORT_SETTINGS,
    FRAME_FORMAT_OPTIONS, TRANSLATIONS, VIDEO_EXTENSIONS, IMAGE_EXTENSIONS, DEVICE
)
from .models import MODELS, MODEL_DISPLAY_TO_FILE
from .state import (
    frame_pairs as state_frame_pairs, stop_processing, pause_processing,
    rgba_to_rgb_for_display, current_language
)
from .file_utils import separate_files_by_type
from .image_processing import upscale_image
from .batch_processor import process_batch


def navigate_frame(frame_idx):
    """Navigate to a specific frame and update the comparison slider"""
    import app_upscale.state as state_module

    if not state_module.frame_pairs:
        return None, ""

    idx = int(frame_idx) - 1
    if 0 <= idx < len(state_module.frame_pairs):
        pair = state_module.frame_pairs[idx]
        return pair, f"Frame {idx + 1}/{len(state_module.frame_pairs)}"
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


def auto_fill_video_name(files):
    """Automatically fill the custom video name field with the first video's name"""
    if not files:
        return ""

    images, videos = separate_files_by_type(files)

    # If there are videos, use the first video's name (without extension)
    if videos:
        first_video = Path(videos[0])
        return first_video.stem  # Returns filename without extension

    # If only images, use the first image's name
    if images:
        first_image = Path(images[0])
        return first_image.stem

    return ""


def create_file_rename_list(files):
    """Create a newline-separated list of file names for renaming"""
    if not files:
        return ""

    # Create a list of file names (one per line)
    file_names = []
    for file_path in files:
        new_name = Path(file_path).stem  # Name without extension
        file_names.append(new_name)

    # Join with newlines
    return "\n".join(file_names)


def test_image_upscale(uploaded_files, model, image_scale_radio, video_resolution_dropdown, output_format, jpeg_quality,
                      precision_mode, preserve_alpha, use_auto_settings, tile_size, tile_overlap,
                      sharpening, contrast, saturation):
    """Quick test upscale on the first uploaded file (image or video first frame) for testing model"""
    # Convert precision mode to boolean or None
    if precision_mode == "None":
        use_fp16 = None  # None = No conversion, PyTorch decides
    elif precision_mode == "FP16 (Half Precision)":
        use_fp16 = True
    else:  # FP32 (Full Precision)
        use_fp16 = False

    # Conversion ×1/×2/×4/×8/×16 → float pour images (test utilise toujours l'échelle image)
    scale_mapping = {"×1": 1.0, "×2": 2.0, "×4": 4.0, "×8": 8.0, "×16": 16.0}
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


def create_app(vram_manager=None, vram_info_text=""):
    """
    Create and return the Gradio interface.

    Args:
        vram_manager: VRAMManager instance for parallel processing
        vram_info_text: Text to display about VRAM configuration

    Returns:
        gr.Blocks: Complete Gradio application
    """
    import app_upscale.state as state_module

    # Simple CSS
    custom_css = """
    #file_upload {
        min-height: 120px !important;
    }
    """

    with gr.Blocks(title="Anime Upscaler - Batch & Video Export") as app:
        # Store CSS for launch() - Gradio 6.0 compatibility
        app.custom_css = custom_css
        # Use translations based on current language
        t = TRANSLATIONS[state_module.current_language]

        gr.Markdown(f"# {t['title']}")

        with gr.Row():
            # Left Panel
            with gr.Column(scale=1):
                upload_accordion = gr.Accordion(t['upload_title'], open=True)
                with upload_accordion:
                    file_input = gr.File(
                        label=t['upload_label'],
                        file_types=["image", "video"],
                        type="filepath",
                        file_count="multiple",
                        elem_id="file_upload"
                    )
                    file_summary = gr.Textbox(
                        label=t['files_summary'],
                        interactive=False,
                        value=t['no_files'],
                        lines=5,
                        max_lines=10
                    )

                model_accordion = gr.Accordion(t['ai_model_title'], open=True)
                with model_accordion:
                    model_select = gr.Radio(
                        list(MODELS.keys()),
                        value="Ani4K v2 Compact (Recommended)",
                        label=t['select_model']
                    )

                format_accordion = gr.Accordion(t['output_format_title'], open=True)
                with format_accordion:
                    # Image scale selector
                    image_scale_radio = gr.Radio(
                        choices=["×1", "×2", "×4", "×8", "×16"],
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
                        choices=["None", "FP16 (Half Precision)", "FP32 (Full Precision)"],
                        value="FP16 (Half Precision)",
                        label=t['precision_mode'],
                        info=t['precision_info']
                    )

                    # Parallel processing toggle
                    enable_parallel = gr.Checkbox(
                        label="⚡ Enable parallel image processing" if state_module.current_language == "en" else "⚡ Activer le traitement parallèle des images",
                        value=True,
                        info="Process multiple images simultaneously (auto-detects optimal worker count based on VRAM). Disable if experiencing stability issues." if state_module.current_language == "en" else "Traiter plusieurs images simultanément (détecte automatiquement le nombre optimal de workers selon la VRAM). Désactiver en cas de problèmes de stabilité."
                    )

                    # Batch size slider (visible only when parallel is enabled)
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=3,  # Default value (matches DEFAULT_PARALLEL_WORKERS)
                        step=1,
                        label=t.get('batch_size', '📦 Taille des batches'),
                        info=t.get('batch_size_info', 'Plus de frames par batch = plus rapide.'),
                        visible=True,
                        interactive=True
                    )

                    # VRAM info display
                    parallel_info = gr.Markdown(
                        value=vram_info_text,  # Filled from startup
                        visible=True
                    )

                video_accordion = gr.Accordion(t['video_export_title'], open=True)
                with video_accordion:
                    export_video_check = gr.Checkbox(label=t['export_videos'], value=True)

                    # Video resolution selector
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
                        value=True,
                        info=t['skip_duplicates_info']
                    )

                # Naming section (separate accordion)
                nommage_accordion = gr.Accordion("📄 Nommage", open=False)
                with nommage_accordion:
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
                        file_rename_textbox = gr.Textbox(
                            label="",
                            value="",
                            placeholder="video1\nvideo2\nimage1",
                            lines=5,
                            max_lines=15,
                            interactive=True,
                            info=""
                        )

                        # Hidden fields for backward compatibility
                        video_custom_name = gr.Textbox(value="", visible=False)
                        file_rename_table = gr.Dataframe(value=[], visible=False)

                auto_cleanup_md = gr.Markdown(f"**{t['auto_cleanup_title']}**")
                cleanup_accordion = gr.Accordion(t['auto_cleanup_accordion'], open=False)
                with cleanup_accordion:
                    auto_delete_input_frames = gr.Checkbox(
                        label=t['delete_input_frames'],
                        value=True,
                        info=t['delete_input_frames_info']
                    )
                    auto_delete_output_frames = gr.Checkbox(
                        label=t['delete_output_frames'],
                        value=True,
                        info=t['delete_output_frames_info']
                    )
                    auto_delete_extraction_folder = gr.Checkbox(
                        label=t['delete_extraction_folder'],
                        value=True,
                        info=t['delete_extraction_folder_info']
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
                    organize_images_folder = gr.Checkbox(
                        label=t['organize_images'],
                        value=True,
                        info=t['organize_images_info']
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

        # Info section removed for minimal UI - see documentation for details

        # Event handlers
        file_input.change(show_file_summary, [file_input], [file_summary])

        # Auto-fill video custom name when files are uploaded
        file_input.change(auto_fill_video_name, [file_input], [video_custom_name])

        # Fill rename textbox with all uploaded files (one per line)
        file_input.change(create_file_rename_list, [file_input], [file_rename_textbox])

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
            t_curr = TRANSLATIONS[state_module.current_language]
            return (
                gr.update(visible=(mode == t_curr['naming_suffix'])),
                gr.update(visible=(mode == t_curr['naming_custom']))
            )

        video_naming_mode.change(
            update_naming_visibility,
            [video_naming_mode],
            [suffix_group, custom_name_group]
        )

        # Toggle batch_size visibility based on enable_parallel
        enable_parallel.change(
            lambda enabled: gr.update(visible=enabled),
            [enable_parallel],
            [batch_size]
        )

        test_btn.click(
            test_image_upscale,
            [file_input, model_select, image_scale_radio, video_resolution_dropdown, output_format_select, jpeg_quality_slider,
             precision_radio, preserve_alpha_check, use_auto_settings, tile_size_slider, tile_overlap_slider,
             sharpening_slider, contrast_slider, saturation_slider],
            [comparison_slider, test_status]
        )

        # Define process_batch wrapper that passes vram_manager
        def process_with_vram_manager(*args):
            return process_batch(*args, vram_manager=vram_manager)

        process_btn.click(
            process_with_vram_manager,
            [file_input, model_select, image_scale_radio, video_resolution_dropdown, output_format_select, jpeg_quality_slider,
             precision_radio,
             codec_select, profile_select, fps_slider, preserve_alpha_check, export_video_check, keep_audio_check, frame_format_select,
             auto_delete_input_frames, auto_delete_output_frames, auto_delete_extraction_folder, auto_delete_frame_mapping, organize_videos_folder, organize_images_folder, skip_duplicate_frames_check,
             use_auto_settings, tile_size_slider, tile_overlap_slider, sharpening_slider, contrast_slider, saturation_slider,
             video_naming_mode, video_suffix, video_custom_name, enable_parallel, batch_size, file_rename_textbox],
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
