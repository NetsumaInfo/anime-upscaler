# 🎨 Anime Upscaler

Application d'upscaling AI pour anime et dessins animés avec traitement batch et export vidéo professionnel.

![Version](https://img.shields.io/badge/version-2.7.1-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## ⚡ Démarrage Rapide

### Installation (Windows)

```bash
# Installation automatique
install.bat

# Lancement
run.bat
```

L'interface web s'ouvre automatiquement sur `http://localhost:7860`

### Installation (Linux/macOS)

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python app.py
```

**Prérequis:**
- Python 3.10+
- FFmpeg et FFprobe dans le PATH (pour vidéos)
- GPU NVIDIA recommandé

---

## 🎯 Fonctionnalités Principales

- **⚡ Pipeline Concurrent (v2.7+)** - Traitement vidéo 2-8x plus rapide avec 4 étages parallèles
- **🔢 Multi-Scale Support** - Upscaling ×1, ×2, ×4, ×8, ×16
- **🌐 Interface Bilingue** - Français/Anglais avec changement instantané
- **📦 Traitement Batch** - Images et vidéos multiples avec parallélisation
- **🎬 Export Vidéo Pro** - H.264, H.265, ProRes, DNxHD/HR
- **✨ Post-Processing** - Sharpening, contraste, saturation
- **🧪 Test Rapide** - Testez le premier fichier avant le batch complet
- **⚡ CUDA Optimisé** - Accélération GPU avec FP16 (50% moins de VRAM)

---

## 🆕 Nouveautés Version 2.7.1

### ⚡ Pipeline Concurrent 4-Étages (v2.7)

Le traitement vidéo utilise maintenant un **pipeline concurrent révolutionnaire** avec 4 étages s'exécutant simultanément !

**Architecture:**
1. **Extraction** - FFmpeg extrait les frames en continu
2. **Détection** - 8 workers CPU détectent les doublons en parallèle
3. **Upscaling** - N workers GPU upscalent simultanément (selon VRAM)
4. **Sauvegarde** - Thread I/O écrit les résultats de manière séquentielle

**Performance (vs version séquentielle):**
- Sans doublons: **33-40% plus rapide** (1000 frames: 180s → 110s)
- Avec doublons (40% typique): **55-65% plus rapide** (180s → 65-80s)
- Scènes statiques (70% doublons): **70-75% plus rapide** (180s → 45-55s)

**Utilisation des ressources:**
- CPU, GPU et I/O occupés **simultanément** (élimine les temps d'attente)
- Activation automatique pour vidéos ≥100 frames
- Fallback transparent vers mode séquentiel si <100 frames

> 💡 **Activation:** Cochez "Enable parallel image processing" dans Advanced Settings (activé par défaut)

---

## 📖 Guide d'Utilisation

### 1. Télécharger vos fichiers

Glissez-déposez vos images/vidéos dans la zone de téléchargement. Vous pouvez ajouter autant de fichiers que vous voulez.

**Formats supportés:**
- Images: JPG, PNG, WebP, BMP, GIF
- Vidéos: MP4, MOV, AVI, WebM, MKV

### 2. Tester (Recommandé)

Cliquez sur **"🧪 Test First Image"** pour tester rapidement le premier fichier uploadé. Cela vous permet d'ajuster les paramètres avant de traiter tout le batch.

### 3. Choisir un modèle

**Modèles recommandés par usage:**

| Modèle | Usage | Vitesse | Qualité |
|--------|-------|---------|---------|
| **Ani4K v2 Compact** ⭐ | Anime moderne HD | Rapide | Excellente |
| AniToon Medium | Anime ancien/basse qualité | Moyenne | Très bonne |
| OpenProteus Compact | Vidéos/usage général | Rapide | Bonne |

> 💡 **Astuce:** Ani4K v2 Compact est le meilleur compromis pour la plupart des utilisations.

**➕ Ajouter vos propres modèles:**
1. Téléchargez depuis [OpenModelDB](https://openmodeldb.info/)
2. Placez les fichiers `.pth` ou `.safetensors` dans `models/`
3. Redémarrez l'application → détection automatique ✨

### 4. Configurer les paramètres

#### Échelle Finale (Image Scale)

- **×2** - Double la résolution (recommandé par défaut)
- **×4** - Quadruple la résolution (2 passes)
- **×1** - Améliore la qualité sans changer la taille

#### Format de Sortie

- **PNG** - Sans perte, fichiers volumineux
- **JPEG** - Compression, petits fichiers (qualité 95 recommandée)
- **WebP** - Meilleur compromis qualité/taille

#### Post-Processing (Optionnel)

- **Sharpening:** 0-2.0 (0.5-1.0 recommandé)
- **Contrast:** 0.8-1.2 (1.0 = original)
- **Saturation:** 0.8-1.2 (1.0 = original)

### 5. Lancer le traitement

Cliquez sur **"▶️ Run Batch"** pour démarrer.

**Contrôles pendant le traitement:**
- ⏸️ **Pause** - Met en pause
- ⏹️ **Stop** - Arrête complètement

---

## 🎬 Export Vidéo

### Codecs Disponibles

| Codec | Qualité | Taille | Usage |
|-------|---------|--------|-------|
| **H.264** | Bonne | Petite | Web, streaming |
| **H.265** | Excellente | Très petite | 4K, moderne |
| **ProRes** | Excellente | Grande | Montage professionnel |
| **DNxHD/HR** | Excellente | Grande | Broadcast |

### Paramètres Vidéo

- **FPS:** `0` = Préserver FPS original (recommandé)
- **Preserve Alpha:** Active pour conserver la transparence
- **Keep Audio:** Active pour garder l'audio original

---

## 📁 Organisation des Fichiers

**Option 1: Dossiers dédiés** (activé par défaut - Recommandé)
- ✅ "Dossier images/ dédié" + "Dossier videos/ dédié" cochés

```
output/
├── images/                     (toutes les images)
│   ├── photo1_upscaled.png
│   └── photo2_upscaled.png
└── videos/                     (toutes les vidéos)
    └── video_upscaled.mp4
```

**Option 2: Organisation par session** (décochées)
- ❌ Options "Dossier dédié" décochées

```
output/20260122_143000/
├── image_upscaled.png          (1 image)
├── images/                     (plusieurs images)
│   ├── photo1_upscaled.png
│   └── photo2_upscaled.png
└── video_name/                 (vidéos avec frames)
    ├── input/
    ├── output/
    └── video_upscaled.mp4
```

> 💡 **Recommandation:** Utilisez les dossiers dédiés pour un accès direct et rapide aux résultats.

---

## ⚙️ Paramètres Avancés

### Mode de Précision (Avancé)

- **FP16** - Recommandé (50% moins VRAM, plus rapide)
- **FP32** - Précision maximale (plus lent, plus de VRAM)
- **None** - Automatique (PyTorch décide)

### Tile Settings

Utilisez des tiles plus petits si vous manquez de VRAM:

- **256px** - GPU 4GB
- **512px** - GPU 8GB+ (recommandé)
- **1024px** - GPU 12GB+

### Organisation & Nettoyage

- **Dossier images/ dédié** - Toutes les images dans `output/images/` (activé par défaut)
- **Dossier videos/ dédié** - Toutes les vidéos dans `output/videos/` (activé par défaut)
- **Delete input frames** - Supprime frames originales après traitement
- **Delete upscaled frames** - Supprime frames upscalées après encodage
- 💡 Dossiers dédiés recommandés pour accès rapide aux résultats

---

## 📚 Historique des Versions

- **v2.7.1** - Correctifs pause/stop, ordre des frames, optimisations pipeline
- **v2.7.0** - Pipeline concurrent 4-étages pour traitement vidéo
- **v2.6.2** - CUDA streams, fix synchronisation, workers VRAM agressifs
- **v2.6.1** - Fusion détection doublons + traitement parallèle
- **v2.5.0** - Architecture modulaire, traitement parallèle images
- **v2.4.0** - Multi-scale support (×1, ×8, ×16)

> 📚 Voir [docs/CHANGELOG.md](docs/CHANGELOG.md) pour l'historique complet

---

## 📚 Documentation

- **[docs/CHANGELOG.md](docs/CHANGELOG.md)** - Historique complet des versions
- **[docs/ADVANCED.md](docs/ADVANCED.md)** - Fonctionnalités avancées
- **[docs/ADDING_MODELS.md](docs/ADDING_MODELS.md)** - Ajouter vos propres modèles

---

## 🐛 Résolution de Problèmes

### L'application ne démarre pas
- Vérifiez que Python 3.10+ est installé
- Exécutez `install.bat` à nouveau
- Vérifiez que FFmpeg est dans le PATH

### Erreur "Out of Memory" (OOM)
- Réduisez le **Tile Size** (256px ou 384px)
- Activez **FP16** dans les paramètres avancés
- Traitez moins de fichiers à la fois

### La vidéo n'a pas de son
- Activez **"Keep audio from original video"** dans les paramètres vidéo

### Traitement vidéo lent
- Vérifiez que "Enable parallel image processing" est activé
- Le pipeline concurrent s'active automatiquement pour vidéos ≥100 frames
- Activez "Ignorer les frames dupliquées" pour gains supplémentaires

---

## 🤝 Contribution

Contributions bienvenues! Ouvrez une issue ou pull request sur GitHub.

---

## 📝 Licence

**Code source:** MIT License - Utilisation libre pour projets personnels et commerciaux.

**Modèles AI:** Les modèles téléchargés restent sous les droits de leurs propriétaires respectifs. Consultez les licences individuelles sur [OpenModelDB](https://openmodeldb.info/) avant utilisation commerciale.

---

## ⭐ Crédits

- **Modèles AI** - [Upscale-Hub](https://github.com/Sirosky/Upscale-Hub) et [OpenModelDB](https://openmodeldb.info/)
- **Architecture** - Spandrel (universal model loader)
- **Interface** - Gradio

---
