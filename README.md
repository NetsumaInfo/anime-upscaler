# 🎨 Anime Upscaler

Application d'upscaling 2x optimisée pour les anime et dessins animés, avec traitement batch et export vidéo professionnel.

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![CUDA](https://img.shields.io/badge/CUDA-supported-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## ✨ Fonctionnalités

- **🖼️ Upscaling 2x AI** - Modèles spécialisés pour anime et dessins animés
- **📦 Traitement Batch** - Images et vidéos multiples simultanément
- **🎬 Export Vidéo Pro** - H.264, H.265, ProRes, DNxHD/HR
- **✨ Post-Processing** - Sharpening, contraste, saturation
- **🎨 Formats Multiples** - PNG, JPEG, WebP
- **💎 Gestion Transparence** - Support alpha channel complet
- **⚡ CUDA Optimisé** - Accélération GPU NVIDIA
- **📁 Organisation Intelligente** - Arborescence simplifiée automatique

## 🚀 Installation

### Windows

```bash
# Installation automatique
install.bat

# Lancement
run.bat
```

### Linux / macOS

```bash
# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Installer PyTorch avec CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Installer dépendances
pip install -r requirements.txt

# Lancer l'application
python app.py
```

### Prérequis

- **Python** 3.8 ou supérieur
- **FFmpeg** et **FFprobe** dans le PATH (pour traitement vidéo)
- **GPU NVIDIA** recommandé (CUDA) pour performance optimale
- **8GB+ VRAM** recommandé pour vidéos haute résolution

## 📖 Guide d'Utilisation

### Démarrage Rapide

1. Lancez l'application avec `run.bat` (Windows) ou `python app.py`
2. L'interface web s'ouvre automatiquement sur `http://localhost:7860`
3. Glissez-déposez vos fichiers (images/vidéos)
4. Sélectionnez un modèle AI (AnimeSharpV4-Fast recommandé)
5. Cliquez sur "▶️ Run Batch"

### Interface

```
┌─────────────────────────────────────────────────────────────┐
│  📁 Input Files                    │  ⚖️ Compare           │
│  - Upload images/videos            │  - Before/After       │
│                                     │  - Frame navigation   │
│  ⚙️ Upscaling Settings             │                       │
│  - Model selection                 │  🖼️ Gallery           │
│  - Tile size / overlap             │  - All results        │
│  - Output format                   │                       │
│  - Post-processing                 │                       │
│                                     │                       │
│  🎬 Video Export Settings          │  📊 Status            │
│  - Codec / Profile                 │  - Progress           │
│  - FPS                             │  - Downloads          │
│                                     │                       │
│  ▶️ Run Batch  ⏸️ Pause  ⏹️ Stop   │  📂 Output folder     │
└─────────────────────────────────────────────────────────────┘
```

## 🤖 Modèles AI

### Modèles Inclus

| Modèle | Vitesse | Qualité | Recommandé pour |
|--------|---------|---------|-----------------|
| [**AnimeSharpV4-Fast**](https://openmodeldb.info/models/2x-AnimeSharpV4-Fast-RCAN-PU) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Vidéos, usage quotidien |
| [**AnimeSharpV4**](https://openmodeldb.info/models/2x-AnimeSharpV4) | ⭐⭐ | ⭐⭐⭐⭐⭐ | Images haute qualité |
| [**Ani4VK-v2-Compact**](https://openmodeldb.info/models/2x-Ani4VK-v2-Compact) ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | **Recommandé** - Tests, GPU limité |

### AnimeSharpV4-Fast

- **6x plus rapide** que AnimeSharpV4
- **95% de la qualité** du modèle complet
- Optimisé pour **artifacts de compression** (MPEG2, H264)
- Parfait pour **traitement vidéo**
- Reproduction **extrêmement fidèle**

### Ajouter Vos Modèles

1. Téléchargez des modèles depuis [OpenModelDB](https://openmodeldb.info/)
2. Placez-les dans le dossier `models/`
3. Formats supportés: `.pth`, `.safetensors`
4. Redémarrez l'application

Les modèles sont détectés automatiquement!

## ⚙️ Paramètres Détaillés

### Tile Settings

**Tile Size** - Taille des tuiles de traitement
- `256px` : GPU 4GB VRAM
- `512px` : GPU 8GB+ VRAM (recommandé)
- `1024px` : GPU 12GB+ VRAM

**Tile Overlap** - Chevauchement entre tuiles (16-64px)
- Plus grand = meilleur blending, plus lent
- Plus petit = plus rapide, possibles artifacts

### Output Format

- **PNG** : Sans perte, transparence supportée, fichiers volumineux
- **JPEG** : Compression avec perte, petits fichiers, pas de transparence
- **WebP** : Meilleure compression, moderne, transparence supportée

**Quality** : 80-100 (JPEG/WebP uniquement)
- 95-100 : Quasi-lossless, recommandé
- 85-95 : Bon compromis qualité/taille
- 80-85 : Maximum compression

### Post-Processing

**Sharpening** (0.0 - 2.0)
- `0` : Aucun
- `0.5-1.0` : Léger à modéré (recommandé)
- `1.5-2.0` : Fort (attention artifacts)

**Contrast** (0.8 - 1.2)
- `< 1.0` : Réduire contraste
- `1.0` : Original
- `> 1.0` : Augmenter contraste

**Saturation** (0.8 - 1.2)
- `< 1.0` : Désaturation
- `1.0` : Original
- `> 1.0` : Couleurs vives

### Advanced

**Use FP16 (Half Precision)**
- ✅ Activé : Moins de VRAM, plus rapide (recommandé CUDA)
- ❌ Désactivé : Précision maximale (FP32), plus lent

## 🎬 Export Vidéo

### Codecs Disponibles

| Codec | Alpha | Qualité | Taille | Usage |
|-------|-------|---------|--------|-------|
| **H.264 (AVC)** | ❌ | Bonne | Petite | Web, streaming |
| **H.265 (HEVC)** | ❌ | Excellente | Très petite | 4K, moderne |
| **ProRes** | ✅ 4444/XQ | Excellente | Grande | VFX, montage |
| **DNxHD/DNxHR** | ✅ 444 | Excellente | Grande | Broadcast |

### Profils Recommandés

**Pour le web / streaming:**
- H.264 High (compatibilité max)
- H.265 Main10 (meilleure qualité, fichiers plus petits)

**Pour montage professionnel:**
- ProRes 422 HQ (sans transparence)
- ProRes 4444 (avec transparence)
- DNxHR HQ / HQX

**FPS (Frames Per Second):**
- `0` : Préserver FPS original (recommandé)
- `24/30/60` : Forcer FPS spécifique

**Preserve Transparency:**
- Copie le canal alpha original vers la sortie
- Nécessite ProRes 4444/XQ ou DNxHR 444 pour vidéos

## 📁 Organisation des Fichiers

### Structure de Sortie Intelligente

L'application organise automatiquement les fichiers pour éviter les dossiers inutiles:

#### 1 seule image
```
output/
└── 20260115_143022/
    └── image_upscaled.png
```

#### Plusieurs images
```
output/
└── 20260115_143022/
    └── images/
        ├── image1_upscaled.png
        ├── image2_upscaled.png
        └── image3_upscaled.png
```

#### 1 seule vidéo
```
output/
└── 20260115_143022/
    └── video_name/
        ├── input/           # Frames originales
        ├── output/          # Frames upscalées
        └── video_name_upscaled.mp4
```

#### Plusieurs vidéos
```
output/
└── 20260115_143022/
    └── videos/
        ├── video1/
        │   ├── input/
        │   ├── output/
        │   └── video1_upscaled.mp4
        └── video2/
            ├── input/
            ├── output/
            └── video2_upscaled.mp4
```

## 💻 Architecture Technique

### Single-File Architecture

Toute l'application est contenue dans `app.py` (~900 lignes):

- **Chargement Modèles** : Spandrel (universal loader)
- **Traitement Images** : PyTorch + CUDA, tile-based processing
- **Traitement Vidéos** : FFmpeg extraction/encoding
- **Interface** : Gradio web UI
- **Cache Modèles** : Évite rechargements inutiles

### Pipeline de Traitement

```
Input → Separate (images/videos) → Process → Post-Processing → Save
                                       ↓
                                  Tile System
                                  (overlap blend)
                                       ↓
                                  AI Upscale 2x
```

### Optimisations

- **Tile-based processing** : Gère images/frames haute résolution
- **FP16 half-precision** : Réduit VRAM 50%
- **Model caching** : Charge une seule fois par session
- **Smart batching** : Traite en continu sans downtime

## 🔧 Dépannage

### Problèmes Courants

**❌ CUDA not available**
```bash
# Réinstaller PyTorch avec CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**❌ FFmpeg not found**
```bash
# Windows : Télécharger depuis ffmpeg.org
# Linux : sudo apt install ffmpeg
# Mac : brew install ffmpeg
```

**❌ Out of Memory (OOM)**
- Réduire Tile Size (256 pour 4GB VRAM)
- Désactiver FP16
- Fermer autres applications GPU

**❌ Port déjà utilisé**
- L'app essaie automatiquement ports 7860-7869
- Ou spécifier manuellement: `python app.py --server-port 8080`

### Performance Tips

**GPU NVIDIA (CUDA) :**
- Activer FP16
- Tile Size 512-1024
- AnimeSharpV4-Fast recommandé

**CPU uniquement :**
- Tile Size 256
- Ani4VK-v2-Compact (plus rapide)
- Patience... (10-50x plus lent)

## 📝 Changelog

### Version 2.0 (2026-01-15)

**Nouvelles fonctionnalités:**
- ✨ Post-processing (sharpening, contrast, saturation)
- 📦 Formats multiples (PNG, JPEG, WebP)
- ⚙️ Tile overlap configurable
- 🎛️ Toggle FP16 manuel
- 📁 Organisation dossiers intelligente
- 🗑️ Suppression système ZIP frames

**Améliorations:**
- 📖 Documentation complète (README + Info Help)
- 🎯 AnimeSharpV4-Fast recommandé par défaut
- 🔧 Interface réorganisée avec accordéons
- 💡 Tooltips et descriptions améliorées

### Version 1.0

- Version initiale
- Traitement batch images/vidéos
- Export multi-codec
- Support transparence

## 🙏 Crédits

### Modèles AI

- **AnimeSharpV4** / **AnimeSharpV4-Fast** : [Kim2091](https://github.com/Kim2091/Kim2091-Models)
- **Ani4VK-v2-Compact** : [Sirosky](https://github.com/Sirosky/Upscale-Hub)

### Technologies

- [PyTorch](https://pytorch.org/) - Deep Learning framework
- [Gradio](https://gradio.app/) - Web UI framework
- [Spandrel](https://github.com/chaiNNer-org/spandrel) - Universal model loader
- [FFmpeg](https://ffmpeg.org/) - Video processing
- [OpenModelDB](https://openmodeldb.info/) - Model database

## 📄 License

MIT License - Libre d'utilisation pour projets personnels et commerciaux.

**Note:** Les modèles AI peuvent avoir leurs propres licences (généralement CC-BY-NC-SA-4.0).

## 🔗 Liens Utiles

- [OpenModelDB](https://openmodeldb.info/) - Base de données modèles
- [Gradio Documentation](https://gradio.app/docs/) - Framework UI
- [PyTorch CUDA Setup](https://pytorch.org/get-started/locally/) - Installation GPU
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html) - Traitement vidéo

---

**Développé avec ❤️ pour la communauté anime**

*Pour toute question ou problème, ouvrez une issue sur GitHub.*
