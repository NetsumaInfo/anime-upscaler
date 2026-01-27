# Parallel Video Processing - Version 2.6

## 🚀 Overview

Version 2.6 introduces **true parallel processing for video frames**, dramatically improving video upscaling performance. Unlike previous versions where frames were processed sequentially one-by-one, the new system upscales multiple frames simultaneously using available GPU workers.

## 📊 Performance Improvements

### Expected Speedup
- **4GB VRAM (1 worker):** 1.0x (sequential fallback)
- **6GB VRAM (2 workers):** 1.5-1.8x faster
- **8GB VRAM (3 workers):** 2.0-2.3x faster
- **12GB+ VRAM (4 workers):** 2.5-3.0x faster

### With Duplicate Detection
When combined with duplicate frame detection, additional speedup:
- Videos with static scenes (30-50% duplicates): **3-5x faster overall**
- Anime with talking heads: **2-3x faster overall**
- Action-heavy content (few duplicates): **1.5-2.5x faster**

## 🔧 How It Works

### Phase 1: Planning (NEW)
```python
processing_plan = plan_parallel_video_processing(
    frames_dir,
    detect_duplicates=True/False  # User option
)
```

**Always generates a JSON processing plan:**
- `parallel_processing_plan.json` - Contains frame mapping and optimization strategy
- Works with OR without duplicate detection
- Identifies unique frames to upscale in parallel

**Output:**
```json
{
  "frames_to_process": ["frame_00001.png", "frame_00002.png", ...],
  "frame_output_mapping": {
    "0": {"input_path": "...", "unique_frame": "...", "output_path": "...", "is_duplicate": false},
    "1": {"input_path": "...", "unique_frame": "...", "output_path": "...", "is_duplicate": false},
    "5": {"input_path": "...", "unique_frame": "...", "output_path": "...", "is_duplicate": true}
  },
  "stats": {
    "total_frames": 1000,
    "unique_frames": 700,
    "duplicates": 300,
    "duplicate_percentage": 30.0,
    "parallel_jobs": 700
  }
}
```

### Phase 2: Parallel Upscaling (NEW)
```python
with ThreadPoolExecutor(max_workers=vram_manager.max_concurrent_jobs) as executor:
    for frame_path in frames_to_process:
        future = executor.submit(upscale_video_frame_worker, frame_path, ...)
```

**Key changes:**
- Upscales **only unique frames** in parallel
- Uses `VRAMManager` to prevent OOM errors
- Returns PIL images in memory (not saved yet)
- Completes all upscaling before saving

### Phase 3: Sequential Save (NEW)
```python
for frame_idx in range(total_frames):
    frame_info = frame_output_mapping[frame_idx]
    result_img = upscaled_results[frame_info["unique_frame"]]
    save_frame_with_format(result_img, output_path, frame_format)
```

**Reconstructs full sequence:**
- Saves frames in correct order (important for video encoding)
- Reuses upscaled results for duplicate frames
- No redundant file copies

## 📁 New Files & Functions

### `video_processing.py`
```python
def plan_parallel_video_processing(
    frames_dir: str,
    detect_duplicates: bool = True,
    progress_callback=None
) -> Optional[Dict]:
    """
    Create parallel processing plan for video frames.

    ALWAYS generates JSON mapping for optimization, regardless of
    duplicate detection setting.

    Returns:
        - frames_to_process: List of unique frames to upscale
        - frame_output_mapping: How to reconstruct full sequence
        - stats: Performance metrics
    """
```

### `batch_processor.py`
```python
def upscale_video_frame_worker(
    frame_path, model, settings, vram_manager, frame_format
) -> Tuple[str, Image, Image, Optional[str]]:
    """
    Worker for parallel video frame upscaling.

    Returns PIL images WITHOUT saving (deferred to main thread).
    """
```

## 🔄 Comparison: Old vs New System

### Old System (v2.5 and earlier)
```
Extract → [Frame 1 → Upscale → Save] → [Frame 2 → Upscale → Save] → ... → Encode
          ^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^
          Sequential (one at a time)
```

**Problems:**
- GPU idle time between frames
- No parallelization for videos
- Duplicate detection helped but still sequential

### New System (v2.6)
```
Extract → Plan → [Upscale 1, 2, 3, 4 in PARALLEL] → Save in order → Encode
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                  4x faster with 12GB VRAM
```

**Benefits:**
- Full GPU utilization
- True parallel processing
- Smart duplicate handling
- Automatic VRAM management

## 🎯 Use Cases

### Case 1: Video without duplicates, parallel enabled
```
1000 frames, all unique
→ Plan: 1000 parallel jobs
→ Upscale: 1000 frames in parallel (4 at a time with 12GB VRAM)
→ Result: ~2.5x faster than v2.5
```

### Case 2: Video with 40% duplicates, parallel enabled
```
1000 frames, 600 unique, 400 duplicates
→ Plan: 600 parallel jobs
→ Upscale: 600 frames in parallel (4 at a time)
→ Save: Reuse upscaled frames for 400 duplicates
→ Result: ~4x faster than v2.5
```

### Case 3: Video with parallel disabled (fallback)
```
1000 frames
→ Plan: Still generated (for UI statistics)
→ Upscale: Sequential (one at a time)
→ Result: Same speed as v2.5
```

## 🔧 Configuration

### Automatic Worker Count (No user config needed)
```python
# In models.py - VRAMManager class
def auto_calculate_slots(vram_gb: float) -> int:
    if vram_gb < 5:
        return 1   # Sequential fallback
    elif vram_gb < 7:
        return 2   # 2 parallel frames
    elif vram_gb < 10:
        return 3   # 3 parallel frames
    else:
        return 4   # 4 parallel frames
```

### User Controls
- **"Enable parallel image processing"** toggle (already exists) - Now applies to videos too!
- **"Skip duplicate frames"** toggle - Optimizes parallel job count

## 📝 Technical Details

### Thread Safety
- All workers use `VRAMManager.acquire()` before GPU access
- Prevents OOM by limiting concurrent GPU operations
- Main thread handles sequential saving to avoid race conditions

### Memory Management
```python
# Workers return PIL images (not saved)
result_img, orig_img = upscale_video_frame_worker(...)

# Main thread saves in order
save_frame_with_format(result_img, output_path, frame_format)

# Cleanup after all frames saved
for result_img, orig_img in upscaled_results.values():
    result_img.close()
    orig_img.close()
```

### Backward Compatibility
- Old sequential code preserved as fallback
- Activates when:
  - Parallel processing disabled in UI
  - Only 1 frame to process
  - VRAM < 5GB (automatic)

## 🐛 Known Limitations

1. **UI Frame Preview:** Limited to first 100 frames to prevent memory issues
2. **Progress Updates:** May appear "jumpy" as frames complete out-of-order
3. **VRAM Requirements:** Parallel mode requires minimum 6GB VRAM for benefits

## 🔮 Future Improvements

- [ ] Batch saving (group of frames) instead of one-by-one
- [ ] Adaptive worker count based on frame complexity
- [ ] Resume capability (skip already processed frames)
- [ ] Real-time progress bar showing completed frames

## 📚 Related Files

- `app_upscale/video_processing.py` - Planning & frame operations
- `app_upscale/batch_processor.py` - Parallel execution
- `app_upscale/models.py` - VRAMManager
- `test_parallel_video.py` - Unit tests
