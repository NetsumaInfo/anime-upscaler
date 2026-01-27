# 📋 Historique des Versions

Historique complet de toutes les versions d'Anime Upscaler.

---

## Version 2.4.2 - Optimisations de Performance
**Date:** 2026-01-22

### 🚀 Optimisations Majeures

#### Gains de Performance
- ⚡ **+8-12% de vitesse** sur le traitement images/vidéos
- 💾 **Cache poids gaussiens** - Évite recalculs redondants sur grandes images (+5-8%)
- 🎯 **Conversion tensors optimisée** - Dtype vérifié 1 fois au lieu de N fois (+10-15%)
- 🔥 **torch.inference_mode()** - Remplace torch.no_grad() pour inférence plus rapide (+2-5%)

#### Corrections de Bugs
- 🔄 **FIX CRITIQUE: Changement FP16/FP32 fonctionne maintenant**
  - Le cache utilisait une clé incorrecte pour FP32
  - Changer de précision recharge désormais correctement le modèle
  - Message "♻️ Using cached model" pour confirmation

#### Améliorations Techniques
- Clé de cache FP32 explicite: `f"{model_name}_fp32"`
- Suppression vérifications dtype redondantes
- Optimisation boucle de traitement tiles
- Code plus propre et performant

### 🗑️ Nettoyage
- Suppression fichiers de test obsolètes (test_*.py, nul)
- Suppression __pycache__/
- Ajout patterns .gitignore pour éviter accumulation fichiers inutiles

### 📚 Documentation
- Nouveau: CHANGELOG_OPTIMIZATIONS.md (détails techniques)
- Nouveau: OPTIMIZATIONS_SUMMARY.md (résumé utilisateurs)
- Mis à jour: README.md (section optimisations + doc précision)

### 📊 Benchmarks
| Type | Avant | Après | Gain |
|------|-------|-------|------|
| Image 1080p | 2.5s | 2.2s | ~12% |
| Image 4K | 8.0s | 7.2s | ~10% |
| Vidéo 1080p (100f) | 250s | 230s | ~8% |

---

## Version 2.4.1 - Hotfix Multi-Scale
**Date:** 2026-01-20

### 🐛 Corrections
- Fix: Tile size auto-ajusté pour modèles x8/x16
- Fix: Avertissement si échelle cible ignorée avec modèle x1
- Amélioration: Messages d'info plus clairs

---

## Version 2.4 - Support Universel des Modèles
**Date:** 2026-01-19

### 🔥 Nouveautés Majeures

#### Support Multi-Scale Universel
- ✅ **Détection automatique** du facteur d'upscaling (x1, x2, x4, x8, x16+)
- ✅ **Modèles x1 supportés** - Processing sans upscaling (ex: color correction)
- ✅ **Multi-passes intelligents** - Modèle 4x peut faire 2 passes pour 16x
- ✅ **Ajout modèles simple** - Glisser-déposer dans models/ suffit

#### Optimisations Tile Size
- **x8 modèles:** Tile size réduit à 256px (50% du défaut)
- **x16 modèles:** Tile size réduit à 128px (25% du défaut)
- **Prévention OOM** sur modèles haute résolution

#### Interface
- 🔢 **Sélecteur d'échelle** - ×1, ×2, ×4, ×8, ×16 dans l'interface
- ⚠️ **Avertissements** - Si échelle incompatible avec modèle

### 📚 Documentation
- Nouveau: ADDING_MODELS.md (guide ajout modèles personnalisés)
- Nouveau: QUICK_START_4X.md (guide rapide modèles 4x)

---

## Version 2.3.1 - UI Enhancements & Critical Bugfix
**Date:** 2026-01-18

### 🐛 Correction Critique
- **FIX MAJEUR:** Erreur "Operation on closed image" en traitement vidéo
  - Frames dupliquées: Utilisation de `.copy()` pour copies indépendantes
  - Frames uniques: Suppression `img.close()` prématuré
  - Cause: `upscale_image()` retournait référence partagée
  - Aussi corrigé dans traitement images pour cohérence

### 📊 Améliorations Interface

#### Résumé Fichiers avec Dimensions
- **Images:** Affiche `filename.jpg (1920×1080)`
  - Lecture dimensions via PIL
- **Vidéos:** Affiche `filename.mp4 (1280×720)`
  - Extraction résolution via FFprobe
- Affichage ligne par ligne pour meilleure lisibilité
- Gestion erreurs si dimensions illisibles

#### Infos Téléchargement Enrichies
- **Nom fichier** complet
- **Taille fichier** (B/KB/MB/GB auto-formaté)
- **Chemin complet** pour accès facile
- Fix: Section download_info maintenant peuplée pour images

---

## Version 2.3 - Interface Multilingue
**Date:** 2026-01-15

### 🌐 Support Multilingue Complet

#### Système de Traduction
- **Français/Anglais** avec changement instantané
- **Sélecteur langue** - Radio button en haut de l'interface
- **51+ composants UI** traduits dynamiquement
- **Détection locale** - Français par défaut (système)

#### Traductions Complètes
- Tous les accordéons (Upload, AI Model, Output, etc.)
- Tous les labels, info text, placeholders
- Boutons d'action (Test, Run Batch, Pause, Stop)
- Noms d'onglets (Compare, Gallery)
- Messages statut et download

#### Technique
- Dict `TRANSLATIONS` avec clés "fr"/"en"
- Variable globale `current_language`
- Fonction `update_ui_language()` retourne 51+ `gr.update()`
- Pas de rechargement page nécessaire

---

## Version 2.2.1 - Performance & Critical Bugfixes
**Date:** 2026-01-12

### ⚡ Optimisations GPU/VRAM

#### Réduction VRAM 50%
- **FP16 robuste** avec gestion erreurs
- **Conversion directe tensors** (numpy→FP16→GPU en 1 étape)
- **Cache modèles séparé** FP16 vs FP32

#### Gestion Mémoire Agressive
- Nettoyage GPU cache tous les 5 images / 10 frames vidéo
- `torch.cuda.empty_cache()` + `synchronize()`
- Nettoyage images PIL avec `.close()` + `del`
- Prévention accumulation VRAM sur longs batches

#### torch.compile Support
- **20-30% speedup** sur Linux avec Triton
- Fallback gracieux sur Windows
- Suppression erreurs automatique avec `suppress_errors`

### 🐛 Corrections Critiques

#### Duplicate Frame Detection Fix
- **BUG MAJEUR:** Clé cache utilisait frame actuelle au lieu de unique_frame
- Résultat: Duplicates étaient re-upscalés inutilement
- Fix ligne 1025: `upscaled_cache[unique_frame_path]`

#### Frame Extraction Verification
- Nouvelle fonction `get_video_frame_count()` via FFprobe
- `extract_frames()` vérifie nb frames extrait = attendu
- Lève `RuntimeError` si extraction incomplète
- Méthode `-count_packets` avec fallback durée×FPS

### 🔧 Diagnostics
- **Monitoring VRAM** avec `get_gpu_memory_info()`
- **Startup diagnostics** détaillés:
  - GPU name, VRAM total
  - CUDA version, PyTorch version
  - Disponibilité torch.compile
  - Pré-chargement modèle avec affichage VRAM
- **UTF-8 console** (Windows) pour support emoji

---

## Version 2.2 - Auto-Cleanup System
**Date:** 2026-01-08

### 🗑️ Système de Nettoyage Automatique

#### Options de Cleanup
- **Delete input frames** - Suppression progressive pendant traitement
- **Delete upscaled frames** - Suppression après encodage vidéo réussi
- **Économie d'espace:** Jusqu'à 90% sur traitement vidéo

#### Organisation Flexible
- **Organize videos in videos/ folder** - Checkbox pour contrôle
- Mode "intelligent" si désactivé:
  - 1 vidéo → `session/video_name/`
  - Plusieurs → `session/videos/video_name/`

### 🎯 Améliorations
- Nettoyage sécurisé avec vérifications
- Messages confirmation dans UI
- Préservation fichiers importants

---

## Version 2.1 - Professional Features
**Date:** 2026-01-05

### 🎨 10 Modèles Spécialisés (Upscale-Hub)

#### Anime Moderne (HD)
- **Ani4K v2 Compact** ⭐ - Recommandé par défaut
- **Ani4K v2 Ultra Compact** - Version ultra-rapide

#### Anime Ancien / Basse Qualité
- **AniToon Medium** - Équilibre parfait
- **AniToon Small** - Version rapide
- **AniToon Large** - Qualité maximale

#### Anime Ancien (Old Style)
- **AniSD AC RealPLKSR** - Variante optimisée
- **AniSD RealPLKSR** - Version générale

#### Usage Général
- **OpenProteus Compact** - Alternative Topaz
- **AniScale2 Compact** - Très rapide

### 🎞️ Format Frames Vidéo Intermédiaire

#### Options PNG
- **Uncompressed 16-bit** - Qualité maximale, volumineux
- **Normal 8-bit** - Compression niveau 6 (défaut)
- **High Compression 8-bit** - Compression niveau 9

#### Options JPEG
- **Quality 100%** - Quasi-lossless
- **Quality 95%** - Bon compromis

### 🧪 Test First Image
- Fonction `test_image_upscale()` pour tests rapides
- Teste **premier fichier uploadé** automatiquement
- Before/After sans sauvegarde disque
- Intégré avec bouton "🧪 Test First Image"

### 🎯 Interface Améliorée
- **Accordéons collapsibles** - Organisation claire
- **Noms modèles user-friendly** - "Ani4K v2 Compact (Recommended)"
- **Upload multi-fichiers** - Ajout incrémental possible
- **Boutons réorganisés** - Test → Run Batch → Pause/Stop

### ⚙️ Paramètres
- **FPS default = 0** - Préserve FPS original automatiquement
- **Tile overlap** configurable (16-64px)

---

## Version 2.0 - Major Overhaul
**Date:** 2026-01-01

### ✨ Post-Processing System
- **Sharpening** - 0-2.0 multiplier (ImageEnhance.Sharpness)
- **Contrast** - 0.8-1.2 multiplier (ImageEnhance.Contrast)
- **Saturation** - 0.8-1.2 multiplier (ImageEnhance.Color)
- Application après upscaling, avant restauration alpha

### 🎨 Multi-Format Output
- **PNG** - Lossless, transparence, optimize flag
- **JPEG** - Quality 80-100, conversion RGBA→RGB
- **WebP** - Quality 80-100, transparence, method=6

### 📁 Organisation Intelligente
- **1 image:** `session/image_upscaled.ext`
- **Multiple images:** `session/images/image_upscaled.ext`
- **1 vidéo:** `session/video_name/...`
- **Multiple vidéos:** `session/videos/video_name/...`
- Réduit profondeur dossiers inutile

### 🎬 Video Export Improvements
- Suppression export ZIP (seulement frames folders + vidéo)
- Audio preservation avec "Keep audio" option
- Alpha channel support amélioré

### ⚙️ Tile Settings
- **Tile overlap** configurable (meilleur blending)
- **Manual FP16 toggle** dans UI
- Optimisations mémoire

---

## Version 1.5 - Video Support
**Date:** 2025-12-20

### 🎬 Support Vidéo Initial
- Extraction frames avec FFmpeg
- Upscaling frame par frame
- Encodage vidéo H.264/H.265
- Préservation FPS original

### 📊 Interface Gradio
- Compare tab avec image slider
- Gallery tab pour batch results
- Progress bars
- Status messages

---

## Version 1.0 - Initial Release
**Date:** 2025-12-15

### 🎨 Fonctionnalités de Base
- Upscaling images 2x avec Spandrel
- Tile-based processing pour grandes images
- Support PNG/JPEG
- Basic batch processing
- CUDA acceleration
- FP16 support

### 🏗️ Architecture
- Single-file application (app.py)
- Model auto-download
- Gaussian weight blending
- Alpha channel preservation

---

## 🔮 Roadmap Futur

### Planifié pour v2.5
- [ ] Support modèles vidéo natifs (VapourSynth)
- [ ] Batch processing GPU (tiles en parallèle)
- [ ] Interface dark mode
- [ ] Presets sauvegardables
- [ ] Historique traitement

### Considéré pour v3.0
- [ ] torch.compile activation (compatible tous modèles)
- [ ] API REST pour intégration externe
- [ ] Support cloud processing (AWS/GCP)
- [ ] Comparaison A/B automatique
- [ ] Metrics qualité automatiques (PSNR, SSIM)

---

## 📊 Statistiques du Projet

**Lignes de Code:**
- v1.0: ~1200 lignes
- v2.0: ~1800 lignes
- v2.3: ~2300 lignes (multilingual)
- v2.4.2: ~2400 lignes (optimisations)

**Performance Evolution:**
- v1.0 → v2.0: +15% (post-processing optimizations)
- v2.0 → v2.2.1: +25% (GPU optimizations)
- v2.2.1 → v2.4.2: +10% (inference optimizations)
- **Total:** ~50% plus rapide que v1.0

**Features Count:**
- v1.0: 5 fonctionnalités principales
- v2.4.2: 20+ fonctionnalités principales

---

**Dernière mise à jour:** 2026-01-22
