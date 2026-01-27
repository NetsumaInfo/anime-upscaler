# Instructions de Rollback - GPU Pipeline v2.8

## Si le Nouveau Système Ne Fonctionne Pas

Voici comment revenir temporairement au système v2.6.2 (qui marchait bien chez toi).

## Option 1 : Désactiver via Config (RECOMMANDÉ)

**Avantage :** Garde le code v2.8 intact, fallback automatique vers v2.6.2

**Étapes :**

1. Ouvre `app_upscale/config.py`

2. Ligne 78, change `True` en `False` :
```python
# Avant
ENABLE_GPU_PIPELINE = True

# Après
ENABLE_GPU_PIPELINE = False
```

3. Sauvegarde et relance l'app
```bash
run.bat
```

**Résultat :** Le système utilisera automatiquement v2.6.2 (sequential parallel avec CUDA streams).

---

## Option 2 : Rollback Complet du Code

**Avantage :** Supprime complètement v2.8, revient au code exact de v2.6.2

### A. Via Git (SI tu as commit avant)

```bash
cd "s:\projet_app\app upscale"

# Voir l'historique
git log --oneline

# Revenir au commit avant v2.8
git reset --hard <commit-id>

# OU annuler seulement les derniers commits
git reset --hard HEAD~1  # Annule 1 commit
git reset --hard HEAD~3  # Annule 3 commits
```

### B. Manuellement (SI pas de Git)

**Fichiers à restaurer depuis backup :**

1. **Supprimer les nouveaux fichiers :**
```bash
del app_upscale\gpu_pipeline.py
del GPU_PIPELINE_V28.md
del NOUVEAU_SYSTEME_V28.md
del test_gpu_pipeline.py
del ROLLBACK_INSTRUCTIONS.md
```

2. **Restaurer `config.py` :**

Cherche la section "Phase 3" (lignes 73-101) et remplace par :
```python
# ============================================================================
# Phase 3 & 4: Concurrent Pipeline (4-Stage Overlapping)
# ============================================================================

# Enable concurrent pipeline for video processing
ENABLE_CONCURRENT_PIPELINE = True  # Set to False to use sequential video processing
# Minimum frames required to use pipeline (overhead not worth it for short videos)
PIPELINE_MIN_FRAMES = 100

# Pipeline queue sizes (balances memory vs throughput)
PIPELINE_EXTRACTION_QUEUE_SIZE = 100   # Extracted frames waiting for detection
PIPELINE_DETECTION_QUEUE_SIZE = 50     # Unique frames waiting for upscaling
PIPELINE_UPSCALING_QUEUE_SIZE = 50     # Upscaled frames waiting for saving

# Expected Performance Gains (Phases 3 & 4):
# - vs Sequential baseline: 46-61% faster (205s → 110-80s for 1000 frames)
# - Stage overlapping eliminates CPU/GPU idle time
# - Maximizes resource utilization (CPU, GPU, I/O all busy simultaneously)
#
# Performance Breakdown:
#   Stage 1 (Extraction): FFmpeg subprocess + monitoring thread
#   Stage 2 (Detection): ThreadPool with 8 CPU workers (parallel hashing)
#   Stage 3 (Upscaling): ThreadPool with N GPU workers (CUDA streams)
#   Stage 4 (Saving): Sequential I/O thread (maintains frame order)
#
# Total Expected Speedup (All Phases Combined):
#   Without duplicates: 46-61% faster (stages overlapping)
#   With duplicates (40%): 5-8x faster (skip duplicates + overlapping)
#   Best case (70% duplicates): 10-15x faster
```

3. **Restaurer `batch_processor.py` :**

Cherche la section "CHOOSE PROCESSING MODE" (lignes 549-610) et remplace par :
```python
            # ============================================================
            # CHOOSE PROCESSING MODE: Concurrent Pipeline vs Sequential
            # ============================================================
            from .config import ENABLE_CONCURRENT_PIPELINE, PIPELINE_MIN_FRAMES

            # Determine if concurrent pipeline should be used
            use_concurrent_pipeline = (
                ENABLE_CONCURRENT_PIPELINE and
                total_frames >= PIPELINE_MIN_FRAMES and
                enable_parallel and
                vram_manager is not None
            )

            if use_concurrent_pipeline:
                # ============================================================
                # CONCURRENT PIPELINE MODE (Phases 3 & 4)
                # ============================================================
                from .pipeline import ConcurrentPipeline
                import torch

                # Prepare upscale parameters (matching upscale_image() signature)
                upscale_params = {
                    "preserve_alpha": preserve_alpha,
                    "use_fp16": use_fp16,
                    "tile_size": params.get("tile_size", 512),
                    "tile_overlap": params.get("tile_overlap", 32),
                    "target_scale": 2.0,  # Video always uses 2x
                    "target_resolution": video_target_resolution,
                    "sharpening": params.get("sharpening", 0),
                    "contrast": params.get("contrast", 1.0),
                    "saturation": params.get("saturation", 1.0),
                    "is_video_frame": True
                }

                status_messages.append(f"🚀 {vid_name}: Using CONCURRENT PIPELINE (4 stages overlapping)")

                # Run concurrent pipeline
                pipeline = ConcurrentPipeline(
                    video_path=video_path,
                    output_dir=str(frames_out),
                    model=model,
                    vram_manager=vram_manager,
                    upscale_params=upscale_params,
                    detect_duplicates=skip_duplicate_frames,
                    frame_format=frame_format,
                    progress_callback=progress
                )

                success, result_path, pipeline_stats = pipeline.run()

                if not success:
```

ET remplace aussi la partie "Reason for not using pipeline" (ligne 617) :
```python
                # Reason for not using pipeline
                if not ENABLE_CONCURRENT_PIPELINE:
                    status_messages.append(f"ℹ️ {vid_name}: Pipeline disabled in config")
                elif total_frames < PIPELINE_MIN_FRAMES:
                    status_messages.append(f"ℹ️ {vid_name}: Video too short for pipeline ({total_frames} < {PIPELINE_MIN_FRAMES} frames)")
                elif not enable_parallel or vram_manager is None:
                    status_messages.append(f"ℹ️ {vid_name}: Parallel processing not available")
```

4. **Teste que tout marche :**
```bash
cd "s:\projet_app\app upscale"
python -c "from app_upscale.batch_processor import process_batch; print('OK')"
run.bat
```

---

## Option 3 : Rollback vers v2.6.2 Simple (ULTIMATE FALLBACK)

Si tu veux revenir au système v2.6.2 (avant le pipeline v2.7 qui marchait pas) :

**Dans `config.py` ligne 78 :**
```python
ENABLE_GPU_PIPELINE = False  # OU ENABLE_CONCURRENT_PIPELINE = False
```

**Résultat :** Le système utilisera le code "SEQUENTIAL PARALLEL PROCESSING" qui se trouve dans `batch_processor.py` à partir de la ligne 612.

Ce système :
- ✅ A fait ses preuves chez toi
- ✅ Utilise CUDA streams correctement
- ✅ Parallélisation GPU fonctionnelle
- ✅ Détection de doublons fonctionnelle

---

## Vérification Post-Rollback

Après avoir fait un rollback, vérifie que tout marche :

```bash
# Test 1 : Imports OK
cd "s:\projet_app\app upscale"
python -c "from app_upscale import config, batch_processor; print('Imports OK')"

# Test 2 : Lance l'app
run.bat

# Test 3 : Traite une vidéo de test
# Upload vidéo dans l'interface et vérifie qu'il n'y a pas d'erreur
```

---

## Support

Si tu as des problèmes après rollback :

1. **Vérifie les imports :**
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

2. **Vérifie FFmpeg :**
```bash
ffmpeg -version
ffmpeg -hwaccels  # Liste les accélérateurs hardware
```

3. **Logs détaillés :**
Ouvre l'app et regarde les messages dans la console pour identifier le problème.

---

## Recommandations

**Avant de rollback complètement, essaye d'abord Option 1 (désactiver via config).**

Pourquoi ?
- Garde le code v2.8 intact (au cas où tu veux réessayer plus tard)
- Fallback automatique vers v2.6.2 qui marche
- Pas de risque de casser quelque chose
- Réversible en 1 seconde (change True → False)

**Si vraiment v2.8 cause des problèmes, contacte-moi avec :**
- Les messages d'erreur exacts
- Ton setup (GPU, VRAM, OS, versions PyTorch/FFmpeg)
- Ce qui ne marche pas (extraction ? detection ? upscaling ?)

Je pourrai corriger les bugs spécifiques à ton environnement ! 🔧
