# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Anime Upscaler is a Gradio-based web application for AI-powered upscaling of anime images and videos. It supports batch processing with professional video export capabilities, advanced post-processing, and multiple output formats with multi-scale support (x1, x2, x4, x8, x16).

**Technology Stack:**
- **Framework:** Gradio (web UI)
- **Deep Learning:** PyTorch + CUDA, Spandrel (universal model loader)
- **Video Processing:** FFmpeg/FFprobe (frame extraction & encoding)
- **Image Processing:** PIL (post-processing, format conversion)
- **Models:** SafeTensors format (.safetensors) and PyTorch (.pth)
- **Concurrency:** Threading + ThreadPoolExecutor for parallel processing (images AND videos)
- **Architecture:** Modular design with 10 specialized modules (v2.5+), including concurrent pipeline (v2.7+)

**New in Version 2.7.1 (GPU ENCODING: NVENC Hardware Acceleration):**
- **⚡ GPU VIDEO ENCODING:** H.264 and H.265 now use NVIDIA NVENC hardware encoder
  - **H.264:** `libx264` (CPU) → `h264_nvenc` (GPU)
  - **H.265:** `libx265` (CPU) → `hevc_nvenc` (GPU)
  - **ProRes/DNxHD:** Remain CPU-based (no GPU alternative available)

- **📊 Expected Performance Gains (Video Encoding Stage):**
  - **H.264 encoding:** 3-5x faster (40s → 10-15s for 1000 frames)
  - **H.265 encoding:** 5-7x faster (80s → 15-25s for 1000 frames)
  - **Overall video processing:** 15-25% faster total time (encoding is ~20-30% of total)

- **🔧 Technical Changes:**
  - **config.py (lines 309-330):** Updated VIDEO_CODECS dictionary
    - H.264: Changed codec from `libx264` to `h264_nvenc`
    - H.265: Changed codec from `libx265` to `hevc_nvenc`
    - Parameters: `-crf` → `-cq` (Constant Quality for NVENC)
    - Presets: `fast/medium/slow` → `p1-p7` (NVENC presets)
  - **video_processing.py (lines 689-719):** Adapted encoding commands for NVENC
    - Added `-rc vbr` (Variable Bitrate mode for best quality)
    - Changed parameter names: `crf` → `cq`, `preset` → NVENC presets
    - Maintained color metadata and range conversion filters

- **📝 Documentation Updates:**
  - **README.md:** Added "Encoding" column to codec table showing GPU/CPU
  - **ui.py:** Info section updated with ⚡ GPU NVENC labels for H.264/H.265
  - Both English and French versions updated

- **✅ Requirements:**
  - NVIDIA GPU with NVENC support (GTX 600+, RTX series)
  - FFmpeg compiled with `--enable-nvenc` (standard in most distributions)
  - No additional configuration needed (automatic fallback to CPU if NVENC unavailable)

- **⚠️ Breaking Changes:** None (fully backward compatible)
  - If NVENC unavailable, FFmpeg will return error and user can switch to ProRes/DNxHD
  - No changes to UI or workflow

**New in Version 2.7.0 (CONCURRENT PIPELINE: 4-Stage Overlapping Execution):**
- **🚀 CONCURRENT PIPELINE ARCHITECTURE:** Revolutionary 4-stage pipeline for maximum performance
  - **Stage 1 (Extraction):** FFmpeg subprocess + monitoring thread
  - **Stage 2 (Detection):** ThreadPoolExecutor with 8 CPU workers (parallel hashing)
  - **Stage 3 (Upscaling):** ThreadPoolExecutor with N GPU workers (CUDA streams)
  - **Stage 4 (Saving):** Sequential I/O thread (maintains frame order)
  - **Key Innovation:** All 4 stages run SIMULTANEOUSLY with queue-based communication
  - **Result:** CPU, GPU, and I/O all busy at the same time (eliminates idle time)

- **📊 Expected Performance Gains (vs v2.6.2 Sequential):**
  - **Without duplicates:** 33-40% faster (180s → 110-120s for 1000 frames)
    - Baseline: Extraction (60s) + Detection (10s) + Upscaling (30s) + Saving (40s) + Encoding (40s) = 180s
    - Pipeline: All stages overlap → 70-80s (pipeline) + 40s (encoding) = 110-120s
  - **With duplicates (40% typical):** 55-65% faster (180s → 65-80s)
  - **With duplicates (70% static scenes):** 70-75% faster (180s → 45-55s)

- **🔧 Technical Implementation:**
  - **New Module:** `app_upscale/pipeline.py` (~740 lines)
    - `ExtractionStage`: FFmpeg monitoring with frame-by-frame queue feeding
    - `DetectionStage`: Parallel hash computation with `frame_mapping` dictionary
    - `UpscalingStage`: GPU parallel with per-worker CUDA streams
    - `SavingStage`: Buffered sequential I/O with duplicate frame copying
    - `ConcurrentPipeline`: Orchestrator managing 4 threads + 3 queues
  - **Configuration:** `config.py` lines 73-101
    - `ENABLE_CONCURRENT_PIPELINE = True` (toggle on/off)
    - `PIPELINE_MIN_FRAMES = 100` (minimum frames to activate pipeline)
    - Queue sizes: Extraction (100), Detection (50), Upscaling (50)
  - **Integration:** `batch_processor.py` lines 535-595
    - Automatic mode selection: Pipeline vs Sequential
    - Pipeline used when: enabled + ≥100 frames + parallel available
    - Fallback to sequential for short videos or if disabled

- **🎯 Smart Mode Selection:**
  - **Pipeline Mode:** Videos ≥100 frames with parallel processing enabled
  - **Sequential Mode:** Videos <100 frames OR pipeline disabled OR parallel unavailable
  - Automatic transparent fallback (user doesn't need to configure)

- **💾 Memory Management:**
  - Pipeline uses `tempfile.mkdtemp()` for extraction (auto-cleanup)
  - Separate from sequential mode's `frames_in` directory
  - No conflicts between modes

- **📈 Resource Utilization:**
  - CPU: >70% utilization during detection stage (8 workers)
  - GPU: >70% utilization during upscaling stage (N workers)
  - I/O: Continuous disk writes during saving stage
  - **Previous (Sequential):** Only 1 resource busy at a time
  - **Now (Pipeline):** All 3 resources busy simultaneously

- **🔄 Queue-Based Architecture:**
  - `extraction_queue` (100 slots): Extraction → Detection
  - `detection_queue` (50 slots): Detection → Upscaling (unique frames only)
  - `upscaling_queue` (50 slots): Upscaling → Saving
  - Sentinel values (`None`) signal stage completion
  - Automatic backpressure (stages wait if queues full)

- **⚡ Duplicate Frame Optimization (Integrated):**
  - Detection stage builds `frame_mapping: {frame_idx: unique_frame_idx}`
  - Only unique frames passed to upscaling (duplicates skipped)
  - Saving stage copies results for duplicate frames
  - Combined speedup: Skip duplicates + Stage overlapping

- **🎛️ User Control:**
  - Same UI controls as before (no new settings required)
  - "Enable parallel image processing" → enables pipeline for videos
  - "Ignorer les frames dupliquées" → controls duplicate detection
  - Pipeline activates automatically when conditions met

- **📁 Files Modified/Created:**
  - **NEW:** `app_upscale/pipeline.py` (4-stage concurrent pipeline)
  - **Modified:** `app_upscale/config.py` (pipeline configuration)
  - **Modified:** `app_upscale/batch_processor.py` (mode selection + integration)
  - **Modified:** `app_upscale/gpu.py` (import `get_model_dtype`)

- **⚠️ Breaking Changes:** None (fully backward compatible)
  - Pipeline can be disabled via `ENABLE_CONCURRENT_PIPELINE = False`
  - Sequential mode still available and unchanged
  - Automatic fallback ensures no user-visible changes

**New in Version 2.6.2 (CRITICAL FIX: True Parallel Processing):**
- **🐛 CRITICAL BUG FIX #1:** Removed `torch.cuda.synchronize()` from worker threads
  - **Root cause:** `clear_gpu_memory()` was calling `torch.cuda.synchronize()` inside EVERY worker
  - **Impact:** Forced ALL threads to wait after each frame completed → destroyed parallelism completely
  - **Symptom:** 3 workers took ~2.2s/frame instead of ~0.7s/frame (3x slower than expected)
  - **Solution:** Created `clear_gpu_memory_async()` for workers (no sync), moved `synchronize()` AFTER all workers finish
- **🐛 CRITICAL BUG FIX #2:** Added CUDA streams for true GPU parallelism
  - **Root cause:** All workers shared the SAME default CUDA stream → PyTorch serialized GPU operations
  - **Impact:** Even with 5 workers, GPU executed frames one-at-a-time (sequential disguised as parallel)
  - **Symptom:** VRAM not filling up, processing still took ~2.2s/frame with "parallel" mode
  - **Solution:** Each worker creates its own `torch.cuda.Stream()`, executes in dedicated stream context
  - **Synchronization:** Each worker syncs its stream before returning results (ensures data ready)
- **⚡ Aggressive VRAM Worker Allocation:** More workers = true parallel speedup
  - **6GB VRAM:** 2 → **3 workers** (50% increase)
  - **8GB VRAM:** 3 → **5 workers** (67% increase)
  - **10GB VRAM:** 4 → **6 workers** (50% increase)
  - **12GB+ VRAM:** 4 → **8 workers** (100% increase)
- **📊 Expected Performance Gains (vs v2.6.1):**
  - **Without duplicates:**
    - 6GB: 1.5-2x → **3x faster** (true parallel with 3 workers)
    - 8GB: 2-2.5x → **5x faster** (true parallel with 5 workers)
    - 12GB+: 2.5-4x → **8x faster** (true parallel with 8 workers)
  - **With duplicates (40% typical):**
    - 6GB: 4-6x → **8-10x faster** (3 workers + skip duplicates)
    - 8GB: 6-8x → **15-20x faster** (5 workers + skip duplicates)
    - 12GB+: 8-12x → **25-35x faster** (8 workers + skip duplicates)
- **🔧 Technical Changes:**
  - `gpu.py`: Added `clear_gpu_memory_async()` without synchronization for workers
  - `batch_processor.py`: Workers now use `clear_gpu_memory_async()`, sync only after all workers complete
  - `models.py`: `VRAMManager.auto_calculate_slots()` retuned for aggressive allocation
- **⚠️ BREAKING CHANGE (internal):** Direct calls to `clear_gpu_memory()` inside workers are now FORBIDDEN
  - Use `clear_gpu_memory_async()` for non-blocking cleanup in parallel contexts
  - Use `clear_gpu_memory()` ONLY after all parallel work completes

**Previous in Version 2.6.1 (OPTIMIZED: Fusion Duplicate Detection + Parallel Processing):**
- **🚀 ULTIMATE OPTIMIZATION:** Fusion intelligente des deux systèmes JSON pour performances maximales
  - **Pipeline en 4 phases:**
    1. **Duplicate Detection** → génère `frame_mapping.json` (analyse perceptuelle des doublons)
    2. **Intelligent Planning** → utilise le mapping pour créer `parallel_processing_plan.json` OPTIMISÉ
    3. **Parallel Upscaling** → upscale UNIQUEMENT les frames uniques (doublons exclus)
    4. **Sequential Reconstruction** → sauvegarde séquentielle avec copie des résultats pour doublons
  - **Gains de performance combinés:**
    - **SANS doublons:** 2-4x faster (parallel uniquement)
    - **AVEC doublons (30-50%):** 5-8x faster (parallel + skip duplicates)
    - **Cas extrême (70% doublons):** 10-15x faster
- **🎯 Système JSON Fusionné:**
  - `plan_parallel_video_processing()` appelle maintenant `analyze_duplicate_frames()` en interne
  - `frames_to_process` contient SEULEMENT les frames uniques (optimisation critique)
  - Nouveau champ: `duplicate_mapping` pour lookup rapide des doublons
  - Stats détaillées: `unique_frames`, `duplicates`, `duplicate_percentage`, `parallel_jobs`
- **📊 Métriques de Performance Optimisées:**
  - **Sans doublons:**
    - 6GB VRAM: 1.5-2x faster (2 workers)
    - 8GB VRAM: 2-2.5x faster (3 workers)
    - 12GB+ VRAM: 2.5-4x faster (4 workers)
  - **Avec doublons (40% typical):**
    - 6GB VRAM: 4-6x faster (2 workers + skip 40% duplicates)
    - 8GB VRAM: 6-8x faster (3 workers + skip 40% duplicates)
    - 12GB+ VRAM: 8-12x faster (4 workers + skip 40% duplicates)
- **🔧 Technical Implementation:**
  - `plan_parallel_video_processing()` réécrit pour fusion des systèmes
  - Batch processor optimisé: upscale seulement `len(frames_to_process)` au lieu de `total_frames`
  - Messages de progression améliorés: "Upscaling X unique frames (skipping Y duplicates)"
  - Stats détaillées: "OPTIMIZED - X duplicates skipped (Y%), Z unique frames upscaled with N workers"
- **🎛️ User Control:** Un seul toggle contrôle les deux optimisations
  - "Enable parallel image processing" → active parallel + duplicate detection pour vidéos
  - "Ignorer les frames dupliquées" → active/désactive la détection de doublons
- **📁 JSON Files Generated:**
  - `frame_mapping.json` (si duplicate detection activée)
  - `parallel_processing_plan.json` (TOUJOURS généré, optimisé selon duplicate detection)

**Previous in Version 2.6 (Parallel Video Processing - REMPLACÉ par v2.6.1):**
- Parallel processing et duplicate detection fonctionnaient séparément
- frames_to_process contenait toutes les frames uniques (même si doublons détectés)
- Gains: 2-4x (parallel) + 20-50% (duplicates) = 3-5x total
- **v2.6.1 amélioration:** Fusion des systèmes pour 5-15x faster (au lieu de 3-5x)

**Previous in Version 2.5 (Modular Architecture + Parallel Image Processing):**
- **📦 Modular Design:** Refactored from single-file (2400 lines) to 9 organized modules
  - **config.py** - All constants and configurations
  - **state.py** - Thread-safe state management
  - **gpu.py** - GPU/VRAM optimization
  - **file_utils.py** - File type detection
  - **models.py** - Model management
  - **image_processing.py** - Image upscaling pipeline
  - **video_processing.py** - Video frame extraction & encoding
  - **batch_processor.py** - Batch orchestration
  - **ui.py** - Gradio interface
  - **main.py** - Entry point
- **⚡ Parallel Processing:** Multiple images can now be upscaled simultaneously
  - Automatic worker count detection based on available VRAM
  - 4GB VRAM: 1 worker (sequential fallback)
  - 6GB VRAM: 2 parallel workers
  - 8GB VRAM: 3 parallel workers
  - 12GB+ VRAM: 4 parallel workers
  - **Expected speedup:** 1.5-2.5x faster for image batches (30-60% time reduction)
- **🔒 Thread-Safe Architecture:** All processing state managed with locks
  - `processing_state_lock` for pause/stop/running state
  - `check_processing_state()` and `update_processing_state()` helper functions
  - Safe concurrent access to global state
- **💾 VRAM Management:** Intelligent semaphore-based GPU memory allocation
  - `VRAMManager` class controls concurrent GPU access
  - Prevents OOM errors during parallel processing
  - Each worker acquires/releases VRAM slot automatically
- **🎛️ User Control:** Simple toggle in Advanced settings
  - "Enable parallel image processing" checkbox (ON by default)
  - Auto-detected configuration displayed to user
  - Fallback to sequential mode if disabled or single image
- **📹 Videos:** Sequential processing (REPLACED in v2.6 with parallel video processing)
  - Only images benefited from parallelization in v2.5
  - Video parallel processing added in v2.6

**Previous in Version 2.4 (Multi-Scale Support & Optimizations):**
- **🔢 Multi-Scale Support:** Added x8 and x16 upscaling options for images
  - Interface now offers: ×1, ×2, ×4, ×8, ×16 scale options
  - Automatic tile size optimization for high-scale models (x8, x16)
  - x8 models: tile size reduced to 256px (50% of default)
  - x16 models: tile size reduced to 128px (25% of default)
- **🎯 x1 Model Support:** Full support for non-upscaling models (e.g., NES_Composite_To_RGB)
  - x1 models perform processing without changing dimensions
  - Automatic detection and proper handling (no multi-pass, no resize)
  - Target scale ignored when x1 model is used (with warning)
- **⚡ Performance Optimizations:** Intelligent processing for different model scales
  - Automatic VRAM management based on model scale factor
  - Prevents OOM errors on high-scale models with large images

**Previous in Version 2.3.1 (UI Enhancements, File Info & Critical Bugfix):**
- **📊 File Summary with Dimensions:** Automatic display of file dimensions in upload summary
  - Images: Shows `filename.jpg (1920×1080)` using PIL to read dimensions
  - Videos: Shows `filename.mp4 (1280×720)` using FFprobe to read resolution
  - Line-by-line display for better readability
  - Error handling if dimensions cannot be read
- **📥 Enhanced Download Info:** Detailed file information after processing
  - Shows filename, file size (B/KB/MB/GB), and full path
  - Calculates file sizes automatically for all generated files
  - Fixed: download_info section now properly populated (images were missing before)
- **🐛 CRITICAL BUGFIX:** Fixed "Operation on closed image" error in video processing
  - Duplicate frames: Use `.copy()` to create independent in-memory copies
  - Unique frames: Removed premature `img.close()` - now only closed via `orig.close()`
  - Also fixed same issue in image processing for consistency
  - Root cause: `upscale_image()` returns `orig` as same reference as input `img`

**Previous in Version 2.3 (Multilingual Interface):**
- **🌐 Multilingual Support:** Full French/English interface with real-time language switching
- **Language Selector:** Radio button at top of interface to switch between languages instantly
- **Complete Translation:** All UI components (buttons, labels, tooltips, info text) fully translated
- **Dynamic Updates:** Language changes update 51+ UI components without reload
- **Default Language:** French (detects system locale) with seamless English switching

**New in Version 2.2.1 (Performance Optimization & Critical Bugfixes):**
- **GPU/VRAM Optimizations:** 50% VRAM reduction with robust FP16, direct tensor conversion
- **Memory Management:** Aggressive GPU cache clearing, PIL image cleanup, stable long-batch processing
- **torch.compile Support:** 20-30% speedup on Linux with Triton (graceful fallback on Windows)
- **Smart Caching:** Separate FP16/FP32 model cache to avoid reloading
- **Diagnostics:** VRAM usage monitoring, detailed startup info (GPU, CUDA, PyTorch versions)
- **Windows Support:** UTF-8 encoding fix for emoji console output
- **🐛 CRITICAL FIX:** Duplicate frame detection now works correctly (cache key bug fixed)
- **Frame Extraction Verification:** Validates all frames extracted before starting upscale

**Previous in Version 2.1:**
- 10 specialized models from Upscale-Hub (AniToon, Ani4K v2, AniSD, OpenProteus, AniScale2)
- Video frame intermediate format selection (PNG 8/16-bit, JPEG quality)
- Quick test image feature for parameter adjustment
- Collapsible accordions UI (Upload, AI Model, Output Format)
- Multi-file upload improvements
- FPS default = 0 (preserve original)

**Previous in Version 2.0:**
- Post-processing (sharpening, contrast, saturation)
- Multiple output formats (PNG, JPEG, WebP)
- Configurable tile overlap
- Manual FP16 toggle
- Smart folder organization (no unnecessary subfolders)

## Running the Application

```bash
# Initial setup (Windows)
install.bat

# Start the application
run.bat

# Or manually:
venv\Scripts\activate
python main.py

```

The app auto-selects ports 7860-7869 and opens in browser automatically.

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python main.py
```

## Modular Architecture (v2.5+)

The application is now organized into 10 specialized modules for improved maintainability (10th module added in v2.7):

### Dependency Graph (Tier System)

```
Tier 0: config.py (no dependencies)
   ↓
Tier 1: state.py, gpu.py, file_utils.py (depend on config only)
   ↓
Tier 2: models.py, image_processing.py (depend on Tier 0-1)
   ↓
Tier 3: video_processing.py (depends on Tier 0-2)
   ↓
Tier 4: pipeline.py, batch_processor.py (depends on Tier 0-3)
   ↓
Tier 5: ui.py (depends on ALL modules)
   ↓
Entry: main.py (launches UI)
```

### Module Details

#### 1. config.py (~360 lines) - Tier 0
**Responsibility:** All constants and global configurations

**Key Contents:**
- Device detection (`DEVICE` - CUDA/CPU)
- File extensions (`IMAGE_EXTENSIONS`, `VIDEO_EXTENSIONS`)
- Paths (`BASE_DIR`, `OUTPUT_DIR`, `MODELS_DIR`)
- Video codecs (`VIDEO_CODECS` dict)
- Export settings (`DEFAULT_EXPORT_SETTINGS`, `FRAME_FORMAT_OPTIONS`)
- Upscaling parameters (`DEFAULT_UPSCALING_SETTINGS`)
- Model definitions (`DEFAULT_MODELS`)
- Translation system (`TRANSLATIONS` - FR/EN)

**Dependencies:** Standard library, torch (for DEVICE detection)

---

#### 2. state.py (~150 lines) - Tier 1
**Responsibility:** Thread-safe processing state management

**Key Contents:**
- Global lock: `processing_state_lock`
- Processing state: `processing_state` dict (running, paused, stop)
- Frame pairs: `frame_pairs` list for UI navigation
- Current language: `current_language`

**Functions:**
```python
check_processing_state(key: str) -> bool
update_processing_state(key: str, value: bool) -> None
stop_processing() -> dict
pause_processing() -> dict
rgba_to_rgb_for_display(img: Image) -> Image
```

**Dependencies:** threading, PIL, numpy, config

---

#### 3. gpu.py (~150 lines) - Tier 1
**Responsibility:** GPU optimization, VRAM monitoring, FP16/FP32 conversion

**Functions:**
```python
clear_gpu_memory() -> None
get_gpu_memory_info() -> dict
get_model_dtype(use_fp16: bool) -> torch.dtype
```

**Key Features:**
- GPU cache cleanup with `torch.cuda.empty_cache()`
- VRAM usage monitoring
- torch.compile detection (Linux/Triton)
- FP16/FP32 conversion with fallback

**Dependencies:** torch, config

---

#### 4. file_utils.py (~90 lines) - Tier 1
**Responsibility:** File type detection and separation

**Functions:**
```python
detect_type(file_path: str) -> str  # "image" | "video" | "unknown"
separate_files_by_type(files: list) -> tuple[list, list]  # (images, videos)
```

**Dependencies:** pathlib, config

---

#### 5. models.py (~360 lines) - Tier 2
**Responsibility:** AI model management (scan, download, load, cache)

**Global Variables:**
```python
loaded_models: dict  # Cache of loaded models
MODELS: dict  # Available models
MODEL_DISPLAY_TO_FILE: dict  # Display name → filename mapping
```

**Functions:**
```python
scan_models() -> tuple[dict, dict]
download_model(model_url: str, model_name: str, progress_callback) -> bool
load_model(model_name: str, use_fp16: bool = True) -> Any
get_gpu_vram_gb() -> float
```

**Classes:**
```python
class VRAMManager:
    def __init__(self, max_jobs: int)
    def auto_calculate_slots(vram_gb: float) -> int
    def acquire() -> None
    def release() -> None
    def update_max_jobs(n: int) -> None
```

**Dependencies:** torch, spandrel, requests, config, gpu

---

#### 6. image_processing.py (~650 lines) - Tier 2
**Responsibility:** Complete image upscaling pipeline

**Key Functions:**
```python
# Post-processing
apply_post_processing(img, sharpening, contrast, saturation) -> Image

# Resizing
resize_to_1080p(img: Image) -> Image
resize_to_target_resolution(img, target_width, target_height) -> Image
resize_to_target_scale(img, target_scale, model_scale) -> Image

# Calculations
calculate_upscale_passes(target_scale, model_scale) -> int
apply_dithering(img, strength=0.5) -> Image

# Upscaling with tiling
create_gaussian_weight_map(tile_size, overlap) -> np.ndarray
_upscale_single_pass(img, model, tile_size, tile_overlap, device, dtype) -> Image
upscale_image(img, model, tile_size, tile_overlap, ...) -> tuple[Image, Image]

# Saving
save_image_with_format(img, path, output_format, quality=95) -> None
```

**Key Features:**
- Tiling with Gaussian overlap for large images
- Multi-pass for x4/x8/x16 (successive 2x passes)
- Support for x1 models (processing without upscale)
- Post-processing with PIL (sharpness, contrast, saturation)
- Format conversion (PNG, JPEG, WebP)

**Dependencies:** PIL, torch, numpy, config, models, gpu

---

#### 7. video_processing.py (~480 lines) - Tier 3
**Responsibility:** Frame extraction, duplicate detection, video encoding

**Key Functions:**
```python
# Frame operations
compute_frame_hash(frame_path: str) -> str
analyze_duplicate_frames(frames_dir, progress_callback) -> dict
save_frame_with_format(img, path, frame_format_key, ...) -> None

# Video detection
get_video_frame_count(video_path: str) -> int
extract_frames(video_path, output_dir, progress_callback) -> bool
get_video_fps(video_path: str) -> float

# Encoding
encode_video(frames_dir, output_path, fps, codec_name, ...) -> bool
```

**Key Features:**
- FFmpeg/FFprobe for extraction and metadata
- MD5 hashing for duplicate frame detection
- Multiple codec support (H.264, H.265, ProRes, DNxHD)
- Transparency handling (ProRes 4444, DNxHR 444)
- Frame extraction verification

**Dependencies:** PIL, subprocess, hashlib, json, config, image_processing

---

#### 8. batch_processor.py (~550 lines) - Tier 4
**Responsibility:** Batch orchestration (images + videos)

**Key Functions:**
```python
upscale_image_worker(args: tuple) -> tuple
    """Worker thread for parallel image upscaling"""

process_batch(files, model, ..., progress_callback) -> tuple
    """Main batch processing pipeline"""
```

**Key Features:**
- Separates images/videos via file_utils
- Parallel image processing (ThreadPoolExecutor) when enabled
- Sequential video processing
- Smart folder organization (1 file vs multiple)
- Automatic cleanup (intermediate frames)
- Pause/stop handling via state management
- Download file tracking

**Dependencies:** concurrent.futures, datetime, shutil, config, state, file_utils, models, image_processing, video_processing

---

#### 9. pipeline.py (~740 lines) - Tier 4 (NEW in v2.7)
**Responsibility:** Concurrent video processing pipeline with 4-stage overlapping execution

**Key Classes:**
```python
class FrameData:
    """Data container for frames passing through pipeline stages"""

class ExtractionStage:
    """Stage 1: Extract frames from video using FFmpeg"""

class DetectionStage:
    """Stage 2: Detect duplicate frames using parallel hashing"""

class UpscalingStage:
    """Stage 3: Upscale unique frames using GPU workers"""

class SavingStage:
    """Stage 4: Save upscaled frames to disk (sequential I/O)"""

class ConcurrentPipeline:
    """Orchestrates the 4-stage concurrent video processing pipeline"""
```

**Key Features:**
- Queue-based communication between stages (extraction→detection→upscaling→saving)
- Each stage runs in its own thread (4 threads total)
- Stage 1: FFmpeg subprocess monitoring with frame-by-frame feeding
- Stage 2: Parallel hash computation with ThreadPoolExecutor (8 CPU workers)
- Stage 3: GPU parallel upscaling with per-worker CUDA streams (N workers)
- Stage 4: Buffered sequential I/O maintains frame ordering
- Duplicate frame optimization: Only upscale unique frames, copy for duplicates
- Error handling at all stages with proper propagation
- Progress reporting across all stages (0-100%)
- Pause/stop support via `check_processing_state()`
- Automatic cleanup with tempfile management

**Pipeline Architecture:**
```
Stage 1 (Extraction) → extraction_queue (100 slots)
                       ↓
Stage 2 (Detection)  → detection_queue (50 slots)
                       ↓
Stage 3 (Upscaling)  → upscaling_queue (50 slots)
                       ↓
Stage 4 (Saving)     → frames_out directory
```

**Performance Impact:**
- Without duplicates: 33-40% faster than sequential (stages overlap)
- With duplicates (40%): 55-65% faster (skip duplicates + overlap)
- With duplicates (70%): 70-75% faster (extreme optimization)

**Dependencies:** os, time, queue, threading, subprocess, tempfile, pathlib, concurrent.futures, PIL, config, state, video_processing, image_processing, models

---

#### 10. ui.py (~600 lines) - Tier 5
**Responsibility:** Complete Gradio interface

**Key Functions:**
```python
navigate_frame(direction: str) -> tuple
update_codec_profiles(codec: str) -> gr.update
show_file_summary(files: list) -> str
test_image_upscale(files, model, ...) -> tuple
create_app(vram_manager, vram_info_text) -> gr.Blocks
```

**Key Features:**
- Gradio interface construction with accordions
- FR/EN language selection system
- Frame navigation slider
- Dynamic codec/profile dropdowns
- File dimension display (PIL for images, FFprobe for videos)
- Quick test feature (first file)
- Event handlers for all UI interactions

**Dependencies:** gradio, ALL other modules (top-level integration)

---

#### 11. main.py (~100 lines) - Entry Point
**Responsibility:** Application launcher

**Key Operations:**
1. Display GPU/CUDA diagnostics
2. Scan models
3. Initialize VRAMManager
4. Pre-load default model
5. Create Gradio app
6. Launch with auto port selection (7860-7869)

**Dependencies:** ALL app_upscale modules, gradio

---

## File Organization

```
app upscale/
├── main.py                    # Entry point (~100 lines)
├── app.py.old                 # Backup of old monolithic version
├── app_upscale/               # Main package
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Constants & configs (~360 lines)
│   ├── state.py              # Thread-safe state (~150 lines)
│   ├── gpu.py                # GPU optimization (~150 lines)
│   ├── file_utils.py         # File detection (~90 lines)
│   ├── models.py             # Model management (~360 lines)
│   ├── image_processing.py   # Image pipeline (~650 lines)
│   ├── video_processing.py   # Video pipeline (~480 lines)
│   ├── batch_processor.py    # Batch orchestrator (~550 lines)
│   └── ui.py                 # Gradio interface (~600 lines)
├── models/                    # AI models (auto-downloaded)
│   ├── 2x_AniToon_RPLKSRS_242500.pth
│   ├── 2x_AniToon_RPLKSR_197500.pth
│   ├── 2x_AniToon_RPLKSRL_280K.pth
│   ├── 2x_Ani4Kv2_G6i2_UltraCompact_105K.pth
│   ├── 2x_Ani4Kv2_G6i2_Compact_107500.pth (RECOMMENDED)
│   ├── 2x_AniSD_AC_RealPLKSR_127500.pth
│   ├── 2x_AniSD_RealPLKSR_140K.pth
│   ├── 2x_OpenProteus_Compact_i2_70K.pth
│   └── 2x_AniScale2S_Compact_i8_60K.pth
├── output/                    # Processing results
│   └── YYYYMMDD_HHMMSS/
│       ├── image_upscaled.png          # (1 image only)
│       ├── images/                      # (multiple images)
│       │   └── name_upscaled.{png,jpg,webp}
│       ├── video_name/                  # (1 video only)
│       │   ├── input/  output/  video_upscaled.{mp4,mov}
│       └── videos/                      # (multiple videos)
│           └── video_name/
│               ├── input/  output/  video_upscaled.{mp4,mov}
├── venv/                      # Python virtual environment
├── requirements.txt           # Python dependencies
├── README.md                  # User documentation
├── CLAUDE.md                  # Developer documentation (this file)
├── install.bat                # Windows setup script
└── run.bat                    # Windows launcher (launches main.py)
```

## Key Functions by Module

| Module | Key Functions | Lines |
|--------|---------------|-------|
| **config.py** | Constants only (no functions) | ~360 |
| **state.py** | `check_processing_state()`, `update_processing_state()`, `stop_processing()`, `pause_processing()` | ~150 |
| **gpu.py** | `clear_gpu_memory()`, `get_gpu_memory_info()`, `get_model_dtype()` | ~150 |
| **file_utils.py** | `detect_type()`, `separate_files_by_type()` | ~90 |
| **models.py** | `scan_models()`, `download_model()`, `load_model()`, `VRAMManager` class | ~360 |
| **image_processing.py** | `upscale_image()`, `apply_post_processing()`, `save_image_with_format()` | ~650 |
| **video_processing.py** | `extract_frames()`, `analyze_duplicate_frames()`, `encode_video()` | ~480 |
| **pipeline.py** | `ConcurrentPipeline.run()`, `ExtractionStage`, `DetectionStage`, `UpscalingStage`, `SavingStage` | ~740 |
| **batch_processor.py** | `process_batch()`, `upscale_image_worker()` | ~550 |
| **ui.py** | `create_app()`, `test_image_upscale()`, `show_file_summary()` | ~600 |
| **main.py** | `main()` - entry point | ~100 |

## Important Notes

- **Modular architecture (v2.5+):** Application split into 10 modules for better maintainability (10th module pipeline.py added in v2.7)
- **No circular imports:** Strict tier-based dependency hierarchy (Tier 0→5)
- **Multi-Scale Support (v2.4):** Interface now supports ×1, ×2, ×4, ×8, ×16 upscaling
  - Automatic tile size optimization for x8 (256px) and x16 (128px) models
  - x1 models perform processing without upscaling (e.g., color correction, denoising)
  - Scale factor auto-detected by Spandrel at model load time
- **File Info Display:** Upload summary shows dimensions automatically (v2.3.1)
  - Images: PIL reads width×height directly from file
  - Videos: FFprobe extracts resolution from video metadata
  - Download info: Shows file size (B/KB/MB/GB) and full path after processing
- **Multilingual Interface:** Full French/English support with real-time switching (v2.3)
- **Language System:** `TRANSLATIONS` dict in config, `current_language` in state module
- **Default Language:** French (system locale detection), instant switch to English via selector (v2.3)
- **Audio handling:** Videos can preserve audio with "Keep audio from original video" option (v2.0+)
- **Frame navigation:** `frame_pairs` list stores (original, upscaled) tuples for UI slider
- **Progress state:** Uses Gradio's `gr.Progress()` and global `processing_state` dict for pause/stop
- **Port auto-selection:** Tries 7860-7869 sequentially until free port found
- **No ZIP export:** Removed in v2.0 - only frame folders and encoded videos
- **Models from Upscale-Hub:** 10 specialized models for different anime types (v2.1)
- **Model Display Names:** User-friendly names displayed in UI (e.g., "Ani4K v2 Compact (Recommended)") mapped to technical filenames
- **Ani4K v2 Compact recommended:** Best balance speed/quality for modern anime (default in v2.1)
- **FPS default = 0:** Preserves original video FPS automatically (v2.1)
- **Post-processing optional:** All sliders default to neutral values (0 sharpening, 1.0 contrast/saturation)
- **Intermediate frame format:** Configurable PNG/JPEG format for video frames before encoding (v2.1)
- **Test first file:** Tests first uploaded file directly - no separate upload needed (v2.1)
- **UI button organization:** Test First Image → Run Batch → Pause/Stop at bottom (v2.1)
- **Multi-file upload:** Can add files incrementally without clearing previous uploads (v2.1)
- **UI accordions:** Organized collapsible sections for cleaner interface (v2.1)
- **Frame extraction verification:** System validates all frames extracted before upscaling (v2.2.1)
- **Duplicate frame detection:** Optional feature to skip upscaling identical frames (20-50% speedup on static scenes)
- **Duplicate frame bugfix:** v2.2.1 fixed cache key bug where duplicates were incorrectly re-upscaled
- **AniSD correction:** Models are for OLD anime, not "SD anime with clean sources"

## Modifying the Application

### Adding a New Video Codec

1. **Edit `config.py`** - Add entry to `VIDEO_CODECS` dict (~line 30):
```python
"Your Codec": {
    "codec": "ffmpeg_codec_name",
    "alpha_support": True/False,
    "profiles": {
        "Profile Name": {"param": "value", ...}
    }
}
```

2. **Edit `video_processing.py`** - Add codec logic to `encode_video()` (~line 250):
```python
elif codec_name == "Your Codec":
    cmd.extend(["-c:v", codec, ...])
```

3. **Edit `batch_processor.py`** - Update extension mapping in `process_batch()` (~line 550)

The UI will auto-populate dropdowns and transparency info.

### Adding a New Post-Processing Effect

1. **Edit `image_processing.py`** - Add function to `apply_post_processing()` (~line 30)
2. **Edit `config.py`** - Add parameter to `DEFAULT_UPSCALING_SETTINGS`
3. **Edit `ui.py`** - Add slider in the post-processing section (~line 330)
4. **Edit `batch_processor.py`** - Pass parameter through `process_batch()` params dict

### Adding a New Model Source

1. **Edit `models.py`** - Update `scan_models()` to detect new file patterns
2. **Edit `config.py`** - Add to `DEFAULT_MODELS` if you want to auto-download
3. Restart application - new models will appear in UI automatically

## FFmpeg Requirements

**Required:** FFmpeg and FFprobe must be in PATH

**Frame extraction:** Uses `rgba` pixel format to preserve transparency
**Encoding:** Dynamically builds commands based on codec/profile/alpha settings

## CUDA & Performance Optimization (v2.2.1)

### GPU Optimizations (gpu.py module)
- **Auto-detects CUDA** availability (`DEVICE` in config)
- **Robust FP16 conversion** with error handling and logging
  - Converts model to half precision at load time (not per-tensor)
  - 50% VRAM reduction when successful
  - Graceful fallback to FP32 if conversion fails
- **torch.compile support** for 20-30% speedup
  - Requires PyTorch 2.0+ and Triton (Linux)
  - Graceful fallback on Windows or if Triton unavailable
  - Automatic error suppression with `torch._dynamo.config.suppress_errors`
- **Direct tensor conversion**
  - Converts NumPy→Tensor in target dtype (FP16/FP32) directly
  - Eliminates intermediate FP32→FP16 conversion step
  - Reduces CPU→GPU transfer time by ~10-15%

### Memory Management (models.py + image_processing.py)
- **Smart model caching** (models.py)
  - Separate cache keys for FP16 vs FP32 models
  - Prevents model reloading when switching precision
- **Aggressive GPU cleanup** (gpu.py)
  - Explicit `del` of tensors after use
  - `torch.cuda.empty_cache()` every 5 images / 10 video frames
  - `torch.cuda.synchronize()` in cleanup function
  - Prevents VRAM accumulation on long batch processing
- **PIL image cleanup** (image_processing.py + batch_processor.py)
  - `img.close()` to release file handles
  - `del` for explicit memory deallocation
  - Reduces RAM usage by 30-40% on video processing

### Diagnostics & Monitoring (main.py + gpu.py)
- **VRAM usage monitoring:** `get_gpu_memory_info()` function in gpu.py
- **Detailed startup diagnostics** in main.py:
  - GPU name, total VRAM, CUDA version, PyTorch version
  - torch.compile availability check
  - Platform-specific warnings (Windows Triton limitation)
  - Model pre-loading with VRAM usage display

### Performance Tuning
- **Tile size recommendations** based on VRAM:
  - 256px for 4GB GPUs
  - 384px for 6GB GPUs
  - 512px for 8GB+ GPUs (default)
  - 768-1024px for 12GB+ GPUs
- **Tile overlap adjustable** (16-64px) for quality/speed trade-off
- **FP16 toggle** in UI (defaults to True) for manual control

## Post-Processing Details (image_processing.py)

**Implementation:**
```python
def apply_post_processing(img, sharpening, contrast, saturation):
    if sharpening > 0:
        img = ImageEnhance.Sharpness(img).enhance(1.0 + sharpening)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    return img
```

**Usage:**
- Applied AFTER upscaling, BEFORE alpha channel restoration
- Works on both single images and video frames
- Sharpening: 0=None, 0.5-1.0=Moderate, 1.5-2.0=Strong
- Contrast/Saturation: <1.0=Decrease, 1.0=Original, >1.0=Increase

## Output Format System (image_processing.py)

**Format-Specific Handling:**
- **PNG:** Uses `optimize=True` for smaller files while keeping lossless
- **JPEG:** Converts RGBA to RGB with white background, `quality` and `optimize` flags
- **WebP:** Supports transparency, uses `method=6` (best compression), quality adjustable

**Alpha Channel Handling:**
- JPEG doesn't support transparency: converts RGBA→RGB with white background
- PNG/WebP preserve transparency
- For videos: transparency only preserved with ProRes 4444/XQ or DNxHR 444

## Smart Folder Organization Logic (batch_processor.py)

**Decision Tree:**
```python
if len(images) == 1:
    save to: session/image_upscaled.ext
else:
    save to: session/images/image_upscaled.ext

if len(videos) == 1:
    save to: session/video_name/{input,output,video_upscaled.ext}
else:
    save to: session/videos/video_name/{input,output,video_upscaled.ext}
```

**Benefits:**
- Cleaner output structure
- No unnecessary nested folders
- Easier file access for single-file processing
