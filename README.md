# 🎨 Anime Upscaler

Application d'upscaling AI pour anime et dessins animés avec traitement batch et export vidéo professionnel.

![Version](https://img.shields.io/badge/version-2.6.0-blue)
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
- GPU NVIDIA recommandé (8GB+ VRAM)

---

## 🎯 Fonctionnalités Principales

- **⚡ NOUVEAU v2.6: Traitement Parallèle Vidéo** - 2-3x plus rapide avec traitement simultané des frames
- **🔢 Multi-Scale Support** - Upscaling ×1, ×2, ×4, ×8, ×16
- **🌐 Interface Bilingue** - Français/Anglais avec changement instantané
- **📦 Traitement Batch** - Images et vidéos multiples avec parallélisation
- **🎬 Export Vidéo Pro** - H.264, H.265, ProRes, DNxHD/HR
- **✨ Post-Processing** - Sharpening, contraste, saturation
- **🧪 Test Rapide** - Testez le premier fichier avant le batch complet
- **⚡ CUDA Optimisé** - Accélération GPU avec FP16 (50% moins de VRAM)

---

## 🆕 Nouveautés Version 2.6

### ⚡ Traitement Parallèle Vidéo

Le traitement vidéo est maintenant **2-3x plus rapide** grâce au traitement simultané des frames !

**Comment ça fonctionne:**
1. **Planification intelligente** - Analyse les frames et génère un plan JSON de traitement
2. **Upscaling parallèle** - Traite 2-4 frames simultanément selon votre VRAM
3. **Reconstruction** - Sauvegarde les frames dans le bon ordre

**Performance attendue:**
- 6GB VRAM: **1.5-1.8x plus rapide** (2 frames parallèles)
- 8GB VRAM: **2.0-2.3x plus rapide** (3 frames parallèles)
- 12GB+ VRAM: **2.5-3.0x plus rapide** (4 frames parallèles)

**Avec détection de duplications:**
- Vidéos statiques (30-50% duplicatas): **3-5x plus rapide**
- Anime avec dialogues: **2-3x plus rapide**
- Vidéos d'action: **1.5-2.5x plus rapide**

> 💡 **Activation:** Cochez "Enable parallel image processing" dans Advanced Settings (activé par défaut)

📚 [Documentation complète](docs/PARALLEL_VIDEO_PROCESSING.md)

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

Vos fichiers traités se trouvent dans `output/YYYYMMDD_HHMMSS/`

**Structure:**
```
output/20260122_143000/
├── image_upscaled.png          (1 image seule)
├── images/                     (plusieurs images)
│   ├── photo1_upscaled.png
│   └── photo2_upscaled.png
├── video_name/                 (1 vidéo seule)
│   ├── input/                  (frames originales)
│   ├── output/                 (frames upscalées)
│   └── video_upscaled.mp4
└── videos/                     (plusieurs vidéos)
    └── video_name/
        └── ...
```

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

### Auto-Cleanup Vidéo

- **Delete input frames** - Supprime frames originales après traitement
- **Delete upscaled frames** - Supprime frames upscalées après encodage
- 💡 Active les deux pour économiser l'espace disque

---

## 🆕 Nouveautés v2.4.2

### Optimisations de Performance
- ⚡ **+8-12% de vitesse** sur images/vidéos
- 🔄 **Fix FP16/FP32** - Le changement de précision fonctionne maintenant
- 💾 **Cache optimisé** - Réutilisation intelligente des calculs
- 🔥 **Inférence accélérée** - Utilisation de torch.inference_mode()

> 📚 Voir [docs/VERSIONS.md](docs/VERSIONS.md) pour l'historique complet

---

## 📚 Documentation Complète

- **[docs/INDEX](docs/DOCUMENTATION_INDEX.md)** - Index complet de la documentation
- **[docs/VERSIONS.md](docs/VERSIONS.md)** - Historique des versions et changements
- **[docs/ADVANCED.md](docs/ADVANCED.md)** - Guide des fonctionnalités avancées
- **[docs/ADDING_MODELS.md](docs/ADDING_MODELS.md)** - Comment ajouter vos propres modèles
- **[docs/OPTIMIZATIONS.md](docs/CHANGELOG_OPTIMIZATIONS.md)** - Détails techniques des optimisations

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

### Le changement FP16/FP32 ne fonctionne pas
- Version 2.4.2+ : Le problème est corrigé ✅
- Version antérieure : Redémarrez l'application après changement

---

## 🤝 Contribution

Contributions bienvenues! Ouvrez une issue ou pull request sur GitHub.

---

## 📝 Licence

MIT License - Utilisation libre pour projets personnels et commerciaux.

---

## ⭐ Crédits

- **Modèles AI** - [Upscale-Hub](https://github.com/Sirosky/Upscale-Hub) par Sirosky
- **Architecture** - Spandrel (universal model loader)
- **Interface** - Gradio

---
