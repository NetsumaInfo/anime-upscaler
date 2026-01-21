# 🎨 Anime Upscaler

Application d'upscaling 2x optimisée pour les anime et dessins animés, avec traitement batch et export vidéo professionnel.

![Version](https://img.shields.io/badge/version-2.3.1-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![CUDA](https://img.shields.io/badge/CUDA-supported-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)


## ✨ Fonctionnalités

- **🌐 Interface Multilingue** - Français/Anglais avec changement instantané (v2.3)
- **📊 Résumé Fichiers Enrichi** - Affichage des dimensions (largeur×hauteur) pour chaque fichier (NOUVEAU v2.3.1)
- **📥 Infos Téléchargement Détaillées** - Nom, taille, chemin complet des fichiers générés (NOUVEAU v2.3.1)
- **🖼️ Upscaling Flexible** - Échelles ×1 (qualité++), ×2, ×3, ×4 avec 10 modèles AI spécialisés depuis [Upscale-Hub](https://github.com/Sirosky/Upscale-Hub)
- **📦 Traitement Batch** - Images et vidéos multiples simultanément
- **🎬 Export Vidéo Pro** - H.264, H.265, ProRes, DNxHD/HR
- **✨ Post-Processing** - Sharpening, contraste, saturation
- **🎨 Formats Multiples** - PNG, JPEG, WebP (sortie finale)
- **🎞️ Format Frames Vidéo** - PNG 8/16-bit, JPEG configurable (décompression intermédiaire)
- **🧪 Test Rapide** - Testez sur premier fichier (image ou vidéo) avant batch complet
- **🗑️ Auto-Cleanup** - Suppression automatique frames intermédiaires pour économiser espace
- **💎 Gestion Transparence** - Support alpha channel complet
- **⚡ CUDA Optimisé** - Accélération GPU NVIDIA avec FP16
- **📁 Organisation Flexible** - Arborescence configurable (vidéos dans sous-dossier ou non)
- **🎯 Interface Accordéons** - UI organisée et épurée

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
3. Glissez-déposez vos fichiers (images/vidéos) - vous pouvez ajouter autant de fichiers que vous voulez
4. **Recommandé** : Testez d'abord avec le bouton "🧪 Test" (teste automatiquement le premier fichier uploadé)
5. Sélectionnez un modèle AI (Ani4K v2 Compact recommandé par défaut)
6. Ajustez les paramètres selon vos besoins
7. Cliquez sur "▶️ Run Batch"

### Interface

```
┌─────────────────────────────────────────────────────────────┐
│  🎨 Anime Upscaler         🌐 Français / English (NOUVEAU)  │
├─────────────────────────────────────────────────────────────┤
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

### Modèles Inclus depuis [Upscale-Hub](https://github.com/Sirosky/Upscale-Hub)

L'application télécharge automatiquement **10 modèles spécialisés** lors de l'installation :

| Famille | Modèle | Vitesse | Qualité | Recommandé pour |
|---------|--------|---------|---------|-----------------|
| **AniToon** | RPLKSRS Small | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Anime 90s/2000s basse qualité - RAPIDE |
| **AniToon** | RPLKSR | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Anime 90s/2000s basse qualité - Équilibré |
| **AniToon** | RPLKSRL Large | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Anime 90s/2000s basse qualité - QUALITÉ MAX |
| **Ani4K v2** | UltraCompact | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Anime moderne (Bluray/WEB) - TRÈS RAPIDE |
| **Ani4K v2** | Compact  | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **RECOMMANDÉ** - Anime moderne - Équilibré |
| **AniSD** | AC RealPLKSR | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Anime ancien (vieux anime) - Variante AC |
| **AniSD** | RealPLKSR | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Anime ancien (vieux anime) - Général |
| **OpenProteus** | Compact | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Alternative gratuite à Topaz Proteus |
| **AniScale2** | Compact | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Usage général rapide |

### Recommandations par Usage

**Anime moderne (2010+, Bluray/WEB):**
- 🏆 **Ani4K v2 Compact** (recommandé) - Meilleur équilibre vitesse/qualité
- ⚡ Ani4K v2 UltraCompact - Si GPU limité
- 💎 AniToon Large - Pour qualité maximale sur sources compressées

**Anime ancien (90s-2000s, VHS/DVD/sources basse qualité):**
- 🏆 **AniToon RPLKSR** - Excellent pour restauration
- ⚡ AniToon Small - Version rapide
- 💎 AniToon Large - Qualité maximale

**Anime ancien (Vieux anime):**
- 🏆 **AniSD AC RealPLKSR** - Variante AC optimisée
- 🎯 AniSD RealPLKSR - Version générale

**Usage général / Vidéos:**
- 🏆 **OpenProteus Compact** - Alternative Topaz
- ⚡ AniScale2 Compact - Très rapide

### Ajouter Vos Modèles Personnalisés

Vous pouvez facilement ajouter vos propres modèles d'upscaling :

1. **Téléchargez** des modèles depuis :
   - [Upscale-Hub](https://github.com/Sirosky/Upscale-Hub/releases) (spécialisé anime/cartoon)
   - [OpenModelDB](https://openmodeldb.info/) (tous types d'images)

2. **Placez-les** dans le dossier `models/` de l'application

3. **Formats supportés** : `.pth`, `.safetensors`

4. **Redémarrez** l'application

**✨ Détection automatique :** Les modèles sont scannés au démarrage et apparaissent automatiquement dans la liste de sélection !

**💡 Astuce :** Les modèles 2x sont optimaux car l'application peut faire plusieurs passes pour atteindre ×3 ou ×4.

## ⚙️ Paramètres Détaillés

### Tile Settings

**Tile Size** - Taille des tuiles de traitement
- `256px` : GPU 4GB VRAM
- `512px` : GPU 8GB+ VRAM (recommandé)
- `1024px` : GPU 12GB+ VRAM

**Tile Overlap** - Chevauchement entre tuiles (16-64px)
- Plus grand = meilleur blending, plus lent
- Plus petit = plus rapide, possibles artifacts

### Image Scale (Échelle finale)

Contrôle l'échelle finale de vos images après upscaling :

- **×1** : Upscale 2x puis redimensionne à la taille originale
  - 💡 **Améliore la qualité** sans changer les dimensions
  - Idéal pour nettoyer/améliorer des images sans modifier leur taille
  - Technique : upscale → downscale intelligent = meilleure qualité
- **×2** : Upscaling standard 2x (1 passe)
  - Recommandé par défaut
  - Double la résolution (ex: 1920×1080 → 3840×2160)
- **×3** : Upscaling 3x via multi-passes
  - 2 passes : 2x → 4x, puis downscale à ×3
  - Plus lent mais qualité supérieure
- **×4** : Upscaling 4x via multi-passes
  - 2 passes : 2x → 2x
  - Quadruple la résolution

### Output Format (Final)

Format de sortie final pour images et vidéos encodées :

- **PNG** : Sans perte, transparence supportée, fichiers volumineux
- **JPEG** : Compression avec perte, petits fichiers, pas de transparence
- **WebP** : Meilleure compression, moderne, transparence supportée

**Quality** : 80-100 (JPEG/WebP uniquement)
- 95-100 : Quasi-lossless, recommandé
- 85-95 : Bon compromis qualité/taille
- 80-85 : Maximum compression

### Video Frame Intermediate Format 🎞️ **NOUVEAU**

Format utilisé pour sauvegarder les frames upscalées **avant** l'encodage vidéo :

**PNG Options:**
- **PNG - Uncompressed (16-bit)** : Aucune compression, qualité maximale, fichiers très volumineux, 16-bit depth
- **PNG - Normal (8-bit)** : Compression niveau 6 (défaut), bon équilibre, 8-bit
- **PNG - High Compression (8-bit)** : Compression niveau 9, fichiers plus petits, plus lent, 8-bit

**JPEG Options:**
- **JPEG - Quality 100%** : Qualité maximale, légère compression
- **JPEG - Quality 95%** : Très bonne qualité, fichiers plus petits

**💡 Recommandation:**
- Pour qualité maximale : PNG Uncompressed (16-bit)
- Pour équilibre : PNG Normal (8-bit) - *par défaut*
- Pour économiser espace : JPEG Quality 95%

**Note:** Ce paramètre affecte uniquement les vidéos, pas les images finales.

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

### Test Rapide 🧪 **VERSION 2.2**

Fonction de test rapide qui teste automatiquement le **premier fichier uploadé** (image ou vidéo) :

**Comment utiliser:**
1. Uploadez vos fichiers (images/vidéos) dans la section principale
2. Ajustez vos paramètres (modèle, post-processing, etc.)
3. Cliquez sur le bouton "🧪 Test"
4. Le premier fichier uploadé est automatiquement testé :
   - **Si image** : Upscalée directement
   - **Si vidéo** : Première frame extraite et upscalée
5. Visualisez le résultat dans l'onglet "⚖️ Compare"
6. Ajustez les paramètres et testez à nouveau si nécessaire
7. Une fois satisfait, lancez le traitement batch complet avec "▶️ Run Batch"

**💡 Avantages:**
- Pas besoin d'upload séparé - utilise vos fichiers déjà uploadés
- Supporte vidéos (teste la première frame automatiquement)
- Prévisualisation rapide sans traiter tous les fichiers
- Ajustement des paramètres en temps réel
- Économie de temps pour gros batches
- Comparaison Before/After instantanée

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
- `0` : Préserver FPS original (*par défaut et recommandé*)
- `24/30/60` : Forcer FPS spécifique si nécessaire

**Preserve Transparency:**
- Copie le canal alpha original vers la sortie
- Nécessite ProRes 4444/XQ ou DNxHR 444 pour vidéos

## 🗑️ Auto-Cleanup (Économie d'Espace) **VERSION 2.2**

Système de nettoyage automatique pour économiser de l'espace disque pendant le traitement vidéo :

### Options Disponibles

**🗑️ Delete input frames after processing**
- Supprime automatiquement les frames extraites **au fur et à mesure** du traitement
- Chaque frame originale est supprimée juste après son upscaling
- Le dossier `input/` est supprimé complètement à la fin
- **Recommandé si** : Vous n'avez pas besoin de conserver les frames originales extraites
- **Économie** : Jusqu'à 50% d'espace pendant le traitement

**🗑️ Delete upscaled frames after encoding**
- Supprime automatiquement les frames upscalées après l'encodage vidéo réussi
- Le dossier `output/` entier est supprimé si la vidéo est encodée avec succès
- **Recommandé si** : Vous ne voulez garder que la vidéo finale encodée
- **Économie** : Jusqu'à 90% d'espace final (garde uniquement la vidéo)

**📁 Organize videos in videos/ folder**
- Activé par défaut - toutes les vidéos vont dans `output/session/videos/nom_video/`
- Désactivé - organisation "intelligente" :
  - 1 vidéo seule → `output/session/nom_video/`
  - Plusieurs vidéos → `output/session/videos/nom_video/`
- **Recommandé** : Garder activé pour une organisation cohérente et prévisible

### Exemples d'Utilisation

**Scénario 1 - Maximum d'espace économisé (garde uniquement vidéo finale):**
- ✅ Delete input frames after processing
- ✅ Delete upscaled frames after encoding
- Résultat : Seulement `video_upscaled.mp4` conservé

**Scénario 2 - Garde frames upscalées (pour réencodage ultérieur):**
- ✅ Delete input frames after processing
- ❌ Delete upscaled frames after encoding
- Résultat : `output/` (frames upscalées) + `video_upscaled.mp4`

**Scénario 3 - Conservation complète (debug/archivage):**
- ❌ Delete input frames after processing
- ❌ Delete upscaled frames after encoding
- Résultat : `input/` + `output/` + `video_upscaled.mp4`

### 💡 Recommandations

**Pour usage normal :**
- ✅ Delete input frames
- ✅ Delete upscaled frames
- Économise énormément d'espace, garde uniquement les vidéos finales

**Pour archivage / réencodage futur :**
- ❌ Delete input frames
- ❌ Delete upscaled frames
- Conserve tout pour flexibilité maximale

**Pour économie d'espace pendant traitement :**
- ✅ Delete input frames (suppression au fur et à mesure)
- ❌ Delete upscaled frames
- Libère de l'espace progressivement pendant le traitement

## 📁 Organisation des Fichiers

### Structure de Sortie

L'application organise automatiquement les fichiers. La structure dépend de l'option "Organize videos in videos/ folder" :

#### Images

**1 seule image:**
```
output/
└── 20260115_143022/
    └── image_upscaled.png
```

**Plusieurs images:**
```
output/
└── 20260115_143022/
    └── images/
        ├── image1_upscaled.png
        ├── image2_upscaled.png
        └── image3_upscaled.png
```

#### Vidéos (avec "Organize videos" activé - par défaut)

**1 ou plusieurs vidéos:**
```
output/
└── 20260115_143022/
    └── videos/                      # Toujours créé
        ├── video1/
        │   ├── input/               # Supprimé si auto-delete activé
        │   ├── output/              # Supprimé si auto-delete activé
        │   └── video1_upscaled.mp4
        └── video2/
            ├── input/
            ├── output/
            └── video2_upscaled.mp4
```

#### Vidéos (avec "Organize videos" désactivé - mode intelligent)

**1 seule vidéo:**
```
output/
└── 20260115_143022/
    └── video_name/                  # Pas de sous-dossier "videos"
        ├── input/
        ├── output/
        └── video_name_upscaled.mp4
```

**Plusieurs vidéos:**
```
output/
└── 20260115_143022/
    └── videos/                      # Créé seulement si plusieurs vidéos
        ├── video1/
        └── video2/
```

#### Avec Auto-Cleanup activé (recommandé)

**Maximum nettoyage (les 2 options activées):**
```
output/
└── 20260115_143022/
    └── videos/
        └── video_name/
            └── video_name_upscaled.mp4   # SEULEMENT la vidéo finale
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

### Version 2.3.1 (2026-01-21)

**Nouvelles fonctionnalités:**
- 📊 **Résumé Fichiers Enrichi** - Affichage automatique des dimensions pour chaque fichier
  - **Images** : Nom + dimensions (ex: `photo.jpg (1920×1080)`) obtenues via PIL
  - **Vidéos** : Nom + résolution (ex: `video.mp4 (1280×720)`) obtenues via FFprobe
  - Affichage ligne par ligne pour meilleure lisibilité
  - Gestion d'erreurs si dimensions illisibles
- 📥 **Informations de Téléchargement Détaillées** - Section complète après traitement
  - Nom du fichier avec extension
  - Taille du fichier (B/KB/MB/GB) calculée automatiquement
  - Chemin complet vers le fichier pour accès rapide
  - Nombre total de fichiers générés
  - Format :
    ```
    📥 2 file(s) ready:

    • image_upscaled.png (5.2 MB)
      📁 s:\projet_app\app upscale\output\20260121_123456\image_upscaled.png

    • video_upscaled.mp4 (125.3 MB)
      📁 s:\projet_app\app upscale\output\20260121_123456\video_upscaled.mp4
    ```

**Corrections:**
- ✅ Section "Informations de Téléchargement" maintenant remplie automatiquement après traitement
- ✅ Images ajoutées à la liste download_files (était seulement vidéos avant)
- 🐛 **BUGFIX CRITIQUE** : Correction "Operation on closed image" lors du traitement vidéo
  - Images dupliquées : Utilisation de `.copy()` pour créer copies en mémoire indépendantes
  - Images uniques : Suppression double fermeture de `img` (déjà fermé via `orig.close()`)
  - Affecte traitement vidéo avec détection de frames dupliquées activée

### Version 2.3 (2026-01-21)

**Nouvelles fonctionnalités:**
- 🌐 **Interface Multilingue** - Support complet Français/Anglais
  - Sélecteur de langue en haut à droite (Français / English)
  - Changement instantané sans rechargement de page
  - Tous les textes UI traduits (boutons, labels, tooltips, accordéons)
  - Plus de 51 composants mis à jour dynamiquement
  - Langue par défaut: Français (détection locale système)
- 📚 **Documentation complète** - README et CLAUDE.md mis à jour avec v2.3

**Nettoyage:**
- Suppression des fichiers de test inutilisés
- Nettoyage du dossier output

### Version 2.2 (2026-01-19)

**Nouvelles fonctionnalités:**
- 🧪 **Test vidéo supporté** - La fonction Test supporte maintenant les vidéos (extrait et teste la première frame automatiquement)
- 🗑️ **Auto-Cleanup système** - Suppression automatique des frames intermédiaires pour économiser l'espace disque
  - Delete input frames after processing (suppression au fur et à mesure)
  - Delete upscaled frames after encoding (garde uniquement vidéo finale)
- 📁 **Organisation vidéos configurable** - Checkbox pour choisir entre organisation cohérente (toujours videos/) ou intelligente
- 🎨 **UI compacte améliorée** - Sliders Tile Size/Overlap réorganisés verticalement pour gagner de la place

**Améliorations:**
- Test fonctionne sur premier fichier uploadé (pas besoin d'upload séparé)
- Organisation par défaut : toutes les vidéos dans `videos/` (cohérence maximale)
- Messages de statut pour nettoyage (🗑️) pour feedback utilisateur
- Documentation complète sur Auto-Cleanup et organisation

### Version 2.1 (2026-01-19)

**Nouvelles fonctionnalités majeures:**
- 🤖 **10 nouveaux modèles** depuis [Upscale-Hub](https://github.com/Sirosky/Upscale-Hub) (AniToon, Ani4K v2, AniSD, OpenProteus, AniScale2)
- 🎞️ **Format intermédiaire frames vidéo** configurable (PNG 8/16-bit, JPEG quality)
- 🧪 **Test Image rapide** pour ajuster paramètres avant batch
- 🎯 **Accordéons UI** pour sections Upload, AI Model, Output Format
- 📁 **Upload multi-fichiers amélioré** - ajoutez autant de fichiers que vous voulez
- 🎬 **FPS par défaut = 0** (préserve FPS original automatiquement)

**Modèles remplacés:**
- ❌ Anciens modèles Kim2091 (AnimeSharpV4, AnimeSharpV4-Fast)
- ✅ Nouveaux modèles Upscale-Hub spécialisés par type de contenu
- 🏆 Ani4K v2 Compact recommandé par défaut (équilibre vitesse/qualité)

**Améliorations:**
- 📥 install.bat télécharge automatiquement les 10 modèles
- 📖 Documentation complète mise à jour
- 🎨 Interface réorganisée et plus claire

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
- 🔧 Interface réorganisée
- 💡 Tooltips et descriptions améliorées

### Version 1.0

- Version initiale
- Traitement batch images/vidéos
- Export multi-codec
- Support transparence

## 🙏 Crédits

### Modèles AI

Tous les modèles sont fournis par leurs créateurs respectifs et soumis à leurs licences :

#### [Upscale-Hub Models](https://github.com/Sirosky/Upscale-Hub)
- **Auteur** : [Sirosky](https://github.com/Sirosky)
- **Modèles inclus** : AniToon, Ani4K v2, AniSD, OpenProteus, AniScale2
- **Licence** : CC-BY-NC-SA-4.0 (Attribution - Non Commercial - Share Alike)
- **Usage** : Usage non-commercial uniquement, modifications autorisées si publiées sous même licence
- **Source** : [Upscale-Hub Repository](https://github.com/Sirosky/Upscale-Hub)
- **Détails** : Modèles spécialisés pour différents types d'anime (moderne, ancien, SD) avec architectures optimisées

**⚠️ Important** : Les modèles ne sont PAS inclus dans ce dépôt. Ils sont automatiquement téléchargés depuis les sources officielles lors de l'installation via `install.bat` ou au premier usage. Respectez les conditions de licence CC-BY-NC-SA-4.0 (usage non-commercial uniquement).

### Technologies

- [PyTorch](https://pytorch.org/) - Deep Learning framework (BSD License)
- [Gradio](https://gradio.app/) - Web UI framework (Apache 2.0)
- [Spandrel](https://github.com/chaiNNer-org/spandrel) - Universal model loader (MIT)
- [FFmpeg](https://ffmpeg.org/) - Video processing (LGPL/GPL)
- [OpenModelDB](https://openmodeldb.info/) - Model database

## 📄 License

**Application Code** : MIT License - Libre d'utilisation pour projets personnels et commerciaux.

**Modèles AI** : Tous les modèles inclus proviennent d'[Upscale-Hub](https://github.com/Sirosky/Upscale-Hub) et sont sous licence **CC-BY-NC-SA-4.0**.
- ✅ **Autorisé** : Usage non-commercial, modification, distribution
- ❌ **Non autorisé** : Usage commercial
- 📝 **Requis** : Attribution, partage sous même licence si modifié

Les images/vidéos upscalées sont soumises à la licence CC-BY-NC-SA-4.0 du modèle utilisé pour les créer.

## 🔗 Liens Utiles

- [OpenModelDB](https://openmodeldb.info/) - Base de données modèles
- [Gradio Documentation](https://gradio.app/docs/) - Framework UI
- [PyTorch CUDA Setup](https://pytorch.org/get-started/locally/) - Installation GPU
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html) - Traitement vidéo

---

**Développé avec ❤️ pour la communauté anime**

