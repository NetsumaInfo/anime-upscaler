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

    with gr.Blocks(title="Anime Upscaler - Batch & Video Export", css=custom_css) as app:
        # Use translations based on current language
        t = TRANSLATIONS[state_module.current_language]

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

                    # Parallel workers slider (visible only when parallel is enabled)
                    parallel_workers = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=2,  # Default value
                        step=1,
                        label=t.get('parallel_workers', '👷 Nombre de workers parallèles'),
                        info=t.get('parallel_workers_info', 'Plus de workers = plus rapide, mais utilise plus de VRAM.'),
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

**➕ Ajouter vos propres modèles personnalisés:**

Vous pouvez facilement utiliser vos propres modèles d'upscaling !

1. **Téléchargez** des modèles depuis :
   - [Upscale-Hub](https://github.com/Sirosky/Upscale-Hub/releases) - Spécialisé anime/cartoon
   - [OpenModelDB](https://openmodeldb.info/) - Tous types d'images
2. **Placez** vos fichiers `.pth` ou `.safetensors` dans le dossier `models/`
3. **Redémarrez** l'application
4. **✨ Détection automatique** - Vos modèles apparaissent dans la liste !

💡 **Formats supportés:** PyTorch (.pth), SafeTensors (.safetensors)

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

        # Toggle parallel_workers visibility based on enable_parallel
        enable_parallel.change(
            lambda enabled: gr.update(visible=enabled),
            [enable_parallel],
            [parallel_workers]
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
             auto_delete_input_frames, auto_delete_output_frames, auto_delete_frame_mapping, organize_videos_folder, skip_duplicate_frames_check,
             use_auto_settings, tile_size_slider, tile_overlap_slider, sharpening_slider, contrast_slider, saturation_slider,
             video_naming_mode, video_suffix, video_custom_name, enable_parallel, parallel_workers],
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
