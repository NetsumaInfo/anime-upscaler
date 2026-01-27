# 🔬 Guide Avancé - Anime Upscaler

Guide complet des fonctionnalités avancées et paramètres techniques.

---

## 📑 Table des Matières

1. [Modèles IA en Détail](#-modèles-ia-en-détail)
2. [Multi-Scale Support](#-multi-scale-support)
3. [Mode de Précision (FP16/FP32)](#-mode-de-précision-fp16fp32)
4. [Tile Processing System](#-tile-processing-system)
5. [Post-Processing Avancé](#-post-processing-avancé)
6. [Formats de Sortie](#-formats-de-sortie)
7. [Export Vidéo Professionnel](#-export-vidéo-professionnel)
8. [Duplicate Frame Detection](#-duplicate-frame-detection)
9. [Auto-Cleanup System](#-auto-cleanup-system)
10. [Optimisation Performance](#-optimisation-performance)

---

## 🤖 Modèles IA en Détail

### Ani4K v2 Series (Moderne HD)

#### **Ani4K v2 Compact** ⭐ (Recommandé)
- **Fichier:** `2x_Ani4Kv2_G6i2_Compact_107500.pth`
- **Upscale:** 2x
- **VRAM:** ~2.5GB @ FP16
- **Vitesse:** Rapide (8-12s pour 1080p)
- **Qualité:** Excellente pour anime moderne
- **Usage optimal:**
  - Anime 2010+ (résolution HD native)
  - Sources propres avec peu de bruit
  - Batch processing de séries anime
- **Points forts:**
  - Préserve les détails fins (cheveux, yeux)
  - Excellent sur ligne art
  - Minimal artifacting

#### **Ani4K v2 Ultra Compact**
- **Fichier:** `2x_Ani4Kv2_G6i2_UltraCompact_105K.pth`
- **VRAM:** ~1.8GB @ FP16
- **Vitesse:** Très rapide (5-8s pour 1080p)
- **Qualité:** Bonne
- **Trade-off:** Légère perte détails vs Compact
- **Quand utiliser:**
  - GPU limité (<6GB VRAM)
  - Gros batches nécessitant vitesse max
  - Tests rapides

### AniToon Series (Ancien/Basse Qualité)

#### **AniToon Medium** (Équilibré)
- **Fichier:** `2x_AniToon_RPLKSR_197500.pth`
- **VRAM:** ~3GB @ FP16
- **Vitesse:** Moyenne (12-18s pour 1080p)
- **Spécialité:** Anime 1990-2010, sources compressées
- **Points forts:**
  - Excellent sur sources bruitées
  - Récupère détails perdus par compression
  - Bon sur upscale DVD/VHS
- **Usage optimal:**
  - Anime ancien numérisé
  - Sources web basse qualité
  - Récupération archives

#### **AniToon Small** (Rapide)
- **VRAM:** ~2GB @ FP16
- **Vitesse:** Rapide (8-12s pour 1080p)
- **Trade-off:** Moins de récupération détails

#### **AniToon Large** (Qualité Max)
- **Fichier:** `2x_AniToon_RPLKSRS_242500.pth`
- **VRAM:** ~4.5GB @ FP16
- **Vitesse:** Lente (20-30s pour 1080p)
- **Points forts:**
  - Maximum récupération détails
  - Excellent anti-aliasing
  - Meilleur sur sources très dégradées

### AniSD Series (Old Anime Style)

#### **AniSD AC RealPLKSR** (Optimisé)
- **Fichier:** `2x_AniSD_AC_RealPLKSR_127500.pth`
- **VRAM:** ~3GB @ FP16
- **Spécialité:** Anime ancien (1980-2000)
- **Points forts:**
  - Préserve le grain film classique
  - Respecte aesthetic rétro
  - Bon sur cel animation traditionnelle
- **Usage optimal:**
  - Anime cel classique
  - Sources film numérisées
  - Préservation aesthetic vintage

#### **AniSD RealPLKSR**
- **Fichier:** `2x_AniSD_RealPLKSR_140K.pth`
- **Différence:** Moins aggressive, plus naturel

### OpenProteus & AniScale2 (Général)

#### **OpenProteus Compact**
- **Fichier:** `2x_OpenProteus_Compact_i2_70K.pth`
- **VRAM:** ~2.5GB @ FP16
- **Spécialité:** Vidéos, cartoons occidentaux
- **Points forts:**
  - Polyvalent (pas seulement anime)
  - Bon sur cartoons 3D
  - Alternative à Topaz Video AI

#### **AniScale2 Compact**
- **Fichier:** `2x_AniScale2S_Compact_i8_60K.pth`
- **VRAM:** ~2GB @ FP16
- **Vitesse:** Très rapide
- **Usage:** Tests rapides, gros volumes

---

## 🔢 Multi-Scale Support

### Comment Fonctionne le Multi-Scale

L'application détecte automatiquement le facteur d'upscaling via **Spandrel** et effectue des passes multiples si nécessaire.

#### Échelles Disponibles

**×1 - Quality Enhancement (Pas d'upscaling)**
- Modèle upscale 2x puis downscale à taille originale
- **Utilité:** Améliore qualité sans changer dimensions
- **Technique:** Upscaling → Downscaling intelligent = meilleure qualité
- **Cas d'usage:**
  - Nettoyage artefacts compression
  - Amélioration détails sans redimensionner
  - Post-traitement images finales

**×2 - Standard Upscaling (1 passe)**
- Application directe du modèle 2x
- **Temps:** ~10s pour 1080p → 4K
- **Qualité:** Optimale (pas de perte multi-passes)
- **Recommandé:** Pour la plupart des usages

**×4 - High Resolution (2 passes)**
- Passe 1: 2x upscale
- Passe 2: 2x upscale du résultat
- **Temps:** ~2× temps ×2 (20s pour 1080p → 8K)
- **Qualité:** Légère perte vs modèle 4x natif
- **Note:** Si modèle 4x disponible, utiliser 1 passe directe

**×8 - Ultra Resolution (Optimisé)**
- 3 passes 2x avec tile size réduit (256px)
- **Temps:** ~3× temps ×2
- **VRAM:** Optimisé automatiquement
- **Prévention OOM:** Tile size auto-réduit

**×16 - Extreme Resolution (Optimisé)**
- 4 passes 2x avec tile size réduit (128px)
- **Temps:** ~4× temps ×2
- **VRAM:** Tile size 25% du défaut
- **Avertissement:** Fichiers énormes (4K → 64K)

### Optimisations Automatiques

```python
if scale >= 16:
    tile_size = base_size * 0.25  # 128px
elif scale >= 8:
    tile_size = base_size * 0.5   # 256px
else:
    tile_size = base_size         # 512px
```

---

## ⚡ Mode de Précision (FP16/FP32)

### Comprendre FP16 vs FP32

**FP32 (Float32) - Précision Complète**
- 32 bits par nombre
- Précision: ~7 chiffres décimaux
- Range: ±3.4 × 10³⁸
- Utilisation: 100% VRAM baseline

**FP16 (Float16) - Demi-Précision**
- 16 bits par nombre
- Précision: ~3 chiffres décimaux
- Range: ±6.5 × 10⁴
- Utilisation: 50% VRAM baseline

### Impact sur la Qualité

**Tests comparatifs:**
- PSNR difference: <0.1 dB (imperceptible)
- SSIM difference: <0.001 (imperceptible)
- Visual inspection: Aucune différence visible

**Conclusion:** FP16 est recommandé pour 99% des cas.

### Quand Utiliser FP32?

1. **Recherche/Analyse scientifique**
   - Besoin précision absolue
   - Mesures quantitatives critiques

2. **Debug modèles instables**
   - Artefacts étranges avec FP16
   - NaN/Inf dans outputs

3. **VRAM abondante (16GB+)**
   - Pas de contrainte mémoire
   - Préférence pour "maximum quality"

### Mode "None" (Automatic)

- PyTorch décide automatiquement
- Généralement équivaut à FP32
- **Utiliser si:**
  - Problèmes compatibilité FP16
  - CPU processing (pas de GPU)
  - Tests/debug

### Problèmes Connus

**Modèles DAT:**
- Incompatibilité FP16 (dtype mismatches internes)
- Application force FP32 automatiquement
- Message: "DAT architecture detected - FP16 disabled"

---

## 🧩 Tile Processing System

### Pourquoi les Tiles?

Les images/vidéos sont souvent trop grandes pour tenir en VRAM. Le système découpe en "tiles" (tuiles) qui sont traitées individuellement puis recombinées.

### Fonctionnement

```
Image 4K (3840×2160)
↓
Découpage en tiles 512×512 avec overlap 32px
↓
Traitement chaque tile individuellement sur GPU
↓
Recombination avec Gaussian blending
↓
Image upscalée 8K (7680×4320)
```

### Tile Size Recommendations

**Basé sur VRAM disponible:**

| VRAM | Tile Size | Overlap | Usage |
|------|-----------|---------|-------|
| 4GB | 256px | 16px | Minimal, rapide |
| 6GB | 384px | 24px | Équilibré |
| 8GB | 512px | 32px | **Recommandé** |
| 12GB | 768px | 48px | Haute qualité |
| 16GB+ | 1024px | 64px | Maximum qualité |

**Formule approximative:**
```
Max Tile Size ≈ sqrt(VRAM_GB × 65536)
```

### Tile Overlap

**Définition:** Nombre de pixels de chevauchement entre tiles adjacentes.

**Impact:**
- **16px:** Rapide, possibles lignes visibles
- **32px:** Équilibré (recommandé)
- **48px:** Excellent blending
- **64px:** Maximum qualité, plus lent

**Gaussian Blending:**
L'overlap utilise pondération gaussienne pour transitions lisses:
```
Poids = 1.0 au centre, 0.0 aux bords
Résultat = Σ(tile × poids) / Σ(poids)
```

### Optimisation v2.4.2

**Cache des Poids Gaussiens:**
- Poids de blending calculés une fois, réutilisés
- Clé cache: `(tile_height, tile_width)`
- Gain: ~5-8% sur images avec beaucoup de tiles

---

## 🎨 Post-Processing Avancé

### Sharpening (Accentuation)

**Technique:** ImageEnhance.Sharpness de Pillow

**Formule:**
```python
sharpened = original + (edges × sharpening_factor)
```

**Valeurs recommandées:**
- **0.0:** Aucun (par défaut)
- **0.3-0.5:** Subtil, naturel
- **0.8-1.0:** Modéré, améliore détails
- **1.5-2.0:** Fort, attention artifacts

**Quand utiliser:**
- Après upscaling si image semble "douce"
- Pour récupérer détails fins
- Jamais sur sources déjà sharp (crée halos)

**Artifacts possibles:**
- Halos autour contours (sharpening > 1.5)
- Bruit amplifié (sur sources bruitées)
- "Crunchy" appearance (> 2.0)

### Contrast (Contraste)

**Technique:** ImageEnhance.Contrast

**Valeurs:**
- **0.8:** Réduction 20% (image plus douce)
- **1.0:** Original (par défaut)
- **1.1-1.2:** Augmentation subtile (recommandé)
- **>1.3:** Risque écrasement highlights/shadows

**Utilité:**
- Compenser perte contraste post-upscaling
- Améliorer "punch" visuel
- Correction sources fades

**Attention:**
- >1.2: Perte détails shadows/highlights
- Vérifier histogramme (pas de clipping)

### Saturation (Saturation Couleur)

**Technique:** ImageEnhance.Color

**Valeurs:**
- **0.8:** Désaturation 20% (look "washed")
- **1.0:** Original (par défaut)
- **1.1:** Légère augmentation (subtil)
- **1.2:** Augmentation modérée
- **>1.3:** Risque couleurs "cartoon"

**Cas d'usage:**
- Compenser désaturation JPEG
- Style "vibrant" pour anime
- Correction sources ternes

**Ordre d'Application:**
```
Upscaling → Sharpening → Contrast → Saturation → Alpha Restore
```

---

## 📦 Formats de Sortie

### PNG - Portable Network Graphics

**Caractéristiques:**
- **Compression:** Lossless (aucune perte)
- **Transparence:** Supportée (alpha channel)
- **Profondeur:** 8-bit ou 16-bit par canal
- **Optimize flag:** Active (réduit taille sans perte qualité)

**Taille fichiers:**
- 1080p: ~5-15MB
- 4K: ~25-80MB
- 8K: ~150-400MB

**Quand utiliser:**
- Sources avec transparence
- Archivage qualité maximale
- Pipeline édition (Photoshop, etc.)
- Pas de contrainte espace disque

**Optimisations:**
```python
img.save(path, format="PNG", optimize=True, compress_level=6)
```

### JPEG - Joint Photographic Experts Group

**Caractéristiques:**
- **Compression:** Lossy (avec perte)
- **Transparence:** Non supportée (converti RGBA→RGB)
- **Qualité:** 0-100 (95 recommandé)

**Taille fichiers (quality 95):**
- 1080p: ~1-3MB
- 4K: ~5-12MB
- 8K: ~25-60MB

**Conversion RGBA→RGB:**
```python
if img.mode == 'RGBA':
    bg = Image.new('RGB', img.size, (255, 255, 255))  # Fond blanc
    bg.paste(img, mask=img.split()[3])  # Composite
    img = bg
```

**Quand utiliser:**
- Pas de transparence nécessaire
- Contrainte espace disque
- Distribution web/social media
- Compatibilité maximale

**Artifacts JPEG:**
- **Q < 90:** Blocs 8×8 visibles
- **Q 90-95:** Quasi-imperceptible
- **Q > 95:** Minimal gain vs taille

### WebP - Modern Web Format

**Caractéristiques:**
- **Compression:** Lossy ou Lossless
- **Transparence:** Supportée
- **Qualité:** 0-100
- **Method:** 0-6 (6 = meilleure compression)

**Taille fichiers (quality 95, method 6):**
- 1080p: ~800KB-2MB (30% < JPEG)
- 4K: ~4-8MB
- 8K: ~20-40MB

**Avantages:**
- Meilleure compression que JPEG/PNG
- Support transparence (vs JPEG)
- Format moderne (2010+)

**Inconvénients:**
- Support limité vieux logiciels
- Encodage plus lent

**Quand utiliser:**
- Web moderne (Chrome, Firefox, Edge)
- Meilleur compromis qualité/taille
- Besoin transparence + compression

---

## 🎬 Export Vidéo Professionnel

### H.264 (AVC) - Universal Compatibility

**Profiles:**
- **Baseline:** Vieux devices, décodage simple
- **Main:** Équilibre (recommandé web)
- **High:** Meilleure qualité, devices modernes

**Paramètres FFmpeg:**
```bash
-c:v libx264 -preset slow -crf 18 -profile:v high
```

**CRF (Constant Rate Factor):**
- **0:** Lossless (énorme)
- **18:** Visually lossless (recommandé)
- **23:** Défaut (bon)
- **28:** Qualité moyenne

**Usage optimal:**
- Streaming web (YouTube, Twitch)
- Compatibilité maximale
- Partage social media

### H.265 (HEVC) - Modern Efficiency

**Avantages:**
- 40-50% meilleure compression vs H.264
- Meilleure qualité à même bitrate
- Support 10-bit (HDR)

**Profiles:**
- **Main:** 8-bit standard
- **Main10:** 10-bit (HDR support)

**Paramètres:**
```bash
-c:v libx265 -preset slow -crf 20 -profile:v main10
```

**Inconvénients:**
- Encodage 3-5× plus lent que H.264
- Support limité vieux devices
- Licensing complexe

**Quand utiliser:**
- 4K/8K content
- Archivage (meilleure compression)
- Devices modernes seulement

### ProRes - Professional Post-Production

**Profiles:**
- **ProRes 422 Proxy:** ~45 Mbps, preview/offline
- **ProRes 422 LT:** ~100 Mbps, édition légère
- **ProRes 422:** ~147 Mbps, standard broadcast
- **ProRes 422 HQ:** ~220 Mbps, haute qualité
- **ProRes 4444:** ~330 Mbps, alpha support
- **ProRes 4444 XQ:** ~500 Mbps, maximum qualité + alpha

**Transparence:**
- **4444:** Support alpha channel complet
- **4444 XQ:** Qualité alpha maximale

**Tailles (1080p 30fps):**
- 422 HQ: ~1.5GB/min
- 4444: ~2.3GB/min
- 4444 XQ: ~3.5GB/min

**Quand utiliser:**
- Pipeline post-production pro
- VFX avec alpha channel
- Color grading (10-bit, 12-bit)
- Archivage masters

### DNxHD/DNxHR - Avid Broadcast

**DNxHD (1080p):**
- **DNxHD 36:** ~36 Mbps, offline
- **DNxHD 115:** ~115 Mbps, broadcast
- **DNxHD 175:** ~175 Mbps, haute qualité

**DNxHR (>1080p):**
- **DNxHR LB:** Low bandwidth (~45 Mbps @ 4K)
- **DNxHR SQ:** Standard quality (~145 Mbps @ 4K)
- **DNxHR HQ:** High quality (~220 Mbps @ 4K)
- **DNxHR HQX:** Very high + 10-bit
- **DNxHR 444:** Maximum + alpha support

**Quand utiliser:**
- Workflow Avid Media Composer
- Broadcast television
- Alternative ProRes (open source)

### FPS Management

**FPS = 0 (Recommandé):**
- Auto-détecte FPS original via FFprobe
- Préserve timing parfait
- Évite judder/stutter

**FPS fixe (24/30/60):**
- Override FPS original
- **Attention:** Peut causer audio desync
- Utiliser seulement si FPS source incorrect

### Audio Preservation

**Keep Audio = True:**
- Copie stream audio original sans ré-encodage
- Codec audio préservé (AAC, MP3, FLAC, etc.)
- Sync parfait si FPS préservé

**Keep Audio = False:**
- Vidéo muette
- Utile si audio séparé ou problématique

---

## ⚡ Duplicate Frame Detection

### Fonctionnement

**Phase 1 - Analyse:**
```python
for frame in video_frames:
    hash = MD5(frame_pixels)
    if hash in seen_hashes:
        mark_as_duplicate(frame, first_occurrence)
    else:
        mark_as_unique(frame)
```

**Phase 2 - Upscaling Intelligent:**
```python
for frame in video_frames:
    if is_unique(frame):
        upscaled = upscale_image(frame)
        cache[frame_unique_path] = upscaled
    else:
        upscaled = cache[first_occurrence]  # Réutilise
    save(upscaled)
```

### Gains de Performance

**Vidéo typique:**
- Static scenes: 10-30% duplicates
- Fade to black: 50%+ duplicates
- Credits: 80%+ duplicates

**Exemple concret:**
```
Vidéo: 1000 frames
Duplicates: 300 frames (30%)
Sans detection: 1000 upscales × 2s = 2000s (33min)
Avec detection: 700 upscales × 2s = 1400s (23min)
Gain: 10 minutes (30%)
```

### Fichier frame_mapping.json

**Format:**
```json
{
  "total_frames": 1000,
  "unique_frames": 700,
  "duplicate_percentage": 30.0,
  "frame_mapping": {
    "frame_0001.png": "frame_0001.png",  // Unique
    "frame_0002.png": "frame_0001.png",  // Duplicate de 0001
    "frame_0003.png": "frame_0003.png",  // Unique
    ...
  }
}
```

**Utilité:**
- Inspection manuelle duplicates
- Debug si résultat inattendu
- Statistiques détaillées

### Limitations

**Ne détecte PAS:**
- Frames très similaires (hash différent)
- Compression temporelle (motion compensated)
- Fades progressifs

**Détecte SEULEMENT:**
- Frames pixel-identiques
- Duplicates exacts (freeze frames)

---

## 🗑️ Auto-Cleanup System

### Delete Input Frames (Progressive)

**Fonctionnement:**
```python
for frame in input_frames:
    upscale(frame)
    save_upscaled(frame)
    os.remove(frame)  # Suppression immédiate
```

**Avantages:**
- Libère espace pendant traitement
- Réduit pic d'utilisation disque
- Pas de grosse suppression finale

**Économie:**
- 1080p frame PNG: ~2MB
- 100 frames: ~200MB libérés progressivement

### Delete Upscaled Frames (Post-Encode)

**Fonctionnement:**
```python
encode_video(upscaled_frames, output_video)
if encoding_success:
    shutil.rmtree(upscaled_frames_folder)
```

**Sécurité:**
- Suppression SEULEMENT si encodage réussi
- Vérification exitcode FFmpeg == 0
- Garde frames si échec encodage

**Économie:**
- 4K frame PNG: ~8MB
- 100 frames: ~800MB final

### Scénarios d'Utilisation

**Maximum Economy (Garde uniquement vidéo):**
```
☑ Delete input frames
☑ Delete upscaled frames
Résultat: Seulement video_upscaled.mp4 (~100MB)
Total economy: ~1GB (90%)
```

**Keep Upscaled (Ré-encodage futur):**
```
☑ Delete input frames
☐ Delete upscaled frames
Résultat: output/ + video_upscaled.mp4 (~900MB)
Utilité: Ré-encoder avec codec différent sans re-upscale
```

**Full Archive (Debug/Archivage):**
```
☐ Delete input frames
☐ Delete upscaled frames
Résultat: input/ + output/ + video (~1.1GB)
Utilité: Inspection manuelle frames, debug
```

---

## 🚀 Optimisation Performance

### GPU Memory Management (v2.4.2)

**Cache Clearing Strategy:**
```python
# Every 5 images
if idx % 5 == 0:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

**Bénéfice:**
- Prévient accumulation mémoire fragmentée
- Stable sur longs batches (100+ images)
- Minimal impact performance (~0.1s/5 images)

### Model Caching (v2.4.2)

**Cache Key System:**
```python
cache_keys = {
    "model_fp16": f"{model_name}_fp16",
    "model_fp32": f"{model_name}_fp32",
    "model_none": f"{model_name}_none"
}
```

**Avantage:**
- Changement FP16↔FP32 instantané (si déjà loadé)
- Pas de rechargement disk inutile
- ~3-5s économisés par changement

### Tensor Conversion Optimization (v2.4.2)

**Avant:**
```python
# Multiple conversions
img_tensor = torch.from_numpy(img).to(device=DEVICE)  # FP32
img_tensor = img_tensor.half()  # FP32→FP16
# Check dtype
if img_tensor.dtype != model_dtype:
    img_tensor = img_tensor.to(dtype=model_dtype)  # Possible 3ème conversion
```

**Après:**
```python
# Single conversion directe
model_dtype = get_model_dtype(model)  # Une fois
img_tensor = torch.from_numpy(img).to(dtype=model_dtype, device=DEVICE)
```

**Gain:** 10-15% transfert CPU→GPU

### Gaussian Weights Caching (v2.4.2)

**Avant:**
```python
for tile in tiles:  # 100 tiles
    weight = create_gaussian_weight_map(512, 512, 32)  # Calculé 100×
```

**Après:**
```python
weight_cache = {}
for tile in tiles:
    key = (512, 512)
    if key not in weight_cache:
        weight_cache[key] = create_gaussian_weight_map(512, 512, 32)
    weight = weight_cache[key]  # Calculé 1×, réutilisé 99×
```

**Gain:** 5-8% sur images 4K+ (beaucoup de tiles)

### torch.inference_mode() (v2.4.2)

**Différence vs no_grad():**
```python
# Ancien
with torch.no_grad():  # Désactive gradients uniquement
    output = model(input)

# Nouveau
with torch.inference_mode():  # Désactive gradients + optimisations
    output = model(input)
```

**Optimisations activées:**
- View operations au lieu de copies
- Pas de version tracking
- Autograd hooks désactivés
- View chain shortcuts

**Gain:** 2-5% vitesse inférence

### Benchmarking Tips

**Mesure précise:**
```python
import time
torch.cuda.synchronize()  # Attendre fin GPU
start = time.time()
result = upscale(image)
torch.cuda.synchronize()
elapsed = time.time() - start
```

**Facteurs influençant performance:**
- Taille image (linéaire avec pixels)
- Tile size (optimal = 512px @ 8GB VRAM)
- Tile overlap (minimal impact < 64px)
- Model complexity (Compact vs Large)
- FP16 vs FP32 (~5-10% différence)
- GPU utilization (vérifier avec nvidia-smi)

---

## 🔬 Diagnostic Tools

### VRAM Monitoring

```python
def get_gpu_memory_info():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
```

**Interpréter résultats:**
- **Allocated:** Mémoire actuellement utilisée
- **Reserved:** Mémoire réservée par CUDA (cache)
- **Diff:** Reserved - Allocated = Cache disponible

**Trigger OOM si:**
- Reserved > 90% VRAM physique
- Allocated growth linéaire (memory leak)

### Startup Diagnostics

**Informations affichées:**
```
🎮 GPU: NVIDIA GeForce RTX 3080
💾 VRAM: 10.0GB
🔧 CUDA: 12.1
🐍 PyTorch: 2.1.0
⚡ torch.compile: Available (Linux only)
```

**Utilité:**
- Vérifier versions compatibles
- Identifier limitations platform
- Confirmer GPU détecté

---

**Dernière mise à jour:** 2026-01-22
**Version couverte:** 2.4.2
