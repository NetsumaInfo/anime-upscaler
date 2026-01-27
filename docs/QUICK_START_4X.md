# Guide Rapide : Utiliser un Modèle 4x

## 🎯 Objectif
Utiliser un modèle 4x (comme FaceUpSharpDAT) pour upscaler vos images/vidéos avec un facteur 4x au lieu de 2x.

## 📥 Étape 1 : Télécharger un Modèle 4x

### Option A : 4x-FaceUpSharpDAT (Recommandé pour visages/anime détaillé)
1. Allez sur https://openmodeldb.info/models/4x-FaceUpSharpDAT
2. Cliquez sur "Download Model" (147.5 MB)
3. Sauvegardez `4x-FaceUpSharpDAT.pth` dans vos téléchargements

### Option B : Autres modèles 4x sur OpenModelDB
- **4x-AnimeSharp** : Anime général
- **4x-NMKD-Siax** : Photos et textures
- **4x-UltraSharp** : Usage général

Visitez https://openmodeldb.info/ et filtrez par "4x" dans la barre de recherche.

## 📂 Étape 2 : Installer le Modèle

**Windows :**
```cmd
copy "%USERPROFILE%\Downloads\4x-FaceUpSharpDAT.pth" "S:\projet_app\app upscale\models\"
```

**Linux/Mac :**
```bash
cp ~/Downloads/4x-FaceUpSharpDAT.pth ./models/
```

**Ou via l'explorateur de fichiers :**
1. Ouvrez le dossier `models/` de l'application
2. Copiez-collez le fichier `.pth` téléchargé
3. C'est tout !

## 🚀 Étape 3 : Démarrer l'Application

```bash
run.bat  # Windows
python app.py  # Linux/Mac
```

**Au démarrage, vous verrez :**
```
📦 Scanning models...
⏳ Loading 4x-FaceUpSharpDAT...
⚠️ DAT architecture detected - FP16 disabled (incompatible)
   Using FP32 for stability
✅ 4xFaceUpSharpDAT loaded on cuda (FP32) - 4x upscale
🌐 Using port 7860
```

**Note :** 4x-FaceUpSharpDAT utilise l'architecture DAT qui n'est pas compatible FP16. L'app le détecte automatiquement et utilise FP32 pour éviter les erreurs. Cela utilise plus de VRAM mais garantit la stabilité.

## 🎨 Étape 4 : Utiliser le Modèle 4x

1. **Ouvrez l'interface** : http://localhost:7860
2. **Uploadez vos fichiers** dans "📁 Télécharger Images/Vidéos"
3. **Sélectionnez le modèle** : Choisissez "4x-FaceUpSharpDAT" dans "🤖 Modèle IA"
4. **Ajustez les paramètres** (recommandé pour 4x) :
   - **Taille de Tuile** : 256-384px (au lieu de 512px)
   - **Échelle finale** : ×2 (1 passe 4x puis downscale) ou ×4 (1 passe 4x direct)
5. **Lancez le traitement** : Cliquez sur "▶️ Lancer le Batch"

## ⚙️ Paramètres Recommandés pour Modèles 4x

| Paramètre | Valeur | Raison |
|-----------|--------|--------|
| **Taille de Tuile** | 256-384px | Réduit l'utilisation VRAM |
| **Chevauchement** | 32px | Bon équilibre qualité/vitesse |
| **Mode Précision** | FP16 | 50% moins de VRAM, plus rapide |
| **Échelle finale** | ×2 ou ×4 | ×2 = meilleure qualité (4x puis downscale) |

## 🎯 Cas d'Usage : Image 480p → 1080p

### Avec Modèle 2x (Ancien Comportement)
- **Passe 1** : 480p → 960p (2x)
- **Passe 2** : 960p → 1920p (2x)
- **Resize** : 1920p → 1080p
- **Total** : 2 passes

### Avec Modèle 4x (Nouveau !)
- **Passe 1** : 480p → 1920p (4x) ✅
- **Resize** : 1920p → 1080p
- **Total** : 1 passe seulement ! 🚀

**Résultat :** 2x plus rapide avec qualité égale ou supérieure !

## 🎬 Cas d'Usage : Vidéo 720p → 4K

### Avec Modèle 2x
- **Passe 1** : 720p → 1440p (2x)
- **Passe 2** : 1440p → 2880p (2x)
- **Resize** : 2880p → 2160p (4K)
- **Total** : 2 passes + encodage

### Avec Modèle 4x
- **Passe 1** : 720p → 2880p (4x) ✅
- **Resize** : 2880p → 2160p (4K)
- **Total** : 1 passe + encodage 🚀

## ❓ FAQ

### Q : Le modèle 4x est-il plus lent que le 2x ?
**R :** Par passe, oui (~2-3x plus lent). Mais comme il fait moins de passes au total, le temps final est souvent similaire ou plus rapide !

### Q : Quelle VRAM nécessaire pour un modèle 4x ?
**R :**
- **Pour modèles non-DAT** (avec FP16) :
  - Minimum : 6GB avec tiles 256px
  - Recommandé : 8GB avec tiles 384px
  - Confortable : 12GB+ avec tiles 512px
- **Pour modèles DAT** (4x-FaceUpSharpDAT utilise FP32 automatiquement) :
  - Minimum : 8GB avec tiles 256px
  - Recommandé : 12GB avec tiles 384px
  - Confortable : 16GB+ avec tiles 512px

### Q : Puis-je utiliser FP32 au lieu de FP16 ?
**R :** Oui, mais vous aurez besoin de 2x plus de VRAM. Changez le mode de précision dans "⚡ Avancé" → "Mode de Précision" → "FP32".

### Q : Le modèle 4x fonctionne-t-il avec les vidéos ?
**R :** Absolument ! La détection des frames dupliquées fonctionne aussi, ce qui accélère encore plus le traitement.

### Q : Erreur "CUDA out of memory" ?
**R :** Solutions :
1. Réduire la taille de tuile (256px → 128px)
2. Activer FP16 si désactivé
3. Fermer d'autres applications utilisant le GPU
4. Traiter les fichiers un par un au lieu de batch

### Q : Le modèle n'apparaît pas dans la liste ?
**R :** Vérifiez :
1. Le fichier est bien dans `models/` (pas dans un sous-dossier)
2. L'extension est `.pth` ou `.safetensors`
3. Vous avez redémarré l'application après l'ajout
4. Le fichier n'est pas corrompu (réessayez le téléchargement)

### Q : Pourquoi 4x-FaceUpSharpDAT utilise FP32 au lieu de FP16 ?
**R :** Ce modèle utilise l'architecture DAT (Dual Aggregation Transformer) qui a des composants internes incompatibles avec FP16. L'application le détecte automatiquement et utilise FP32 pour garantir la stabilité. Cela consomme environ 300MB de VRAM supplémentaire mais évite les erreurs de dtype.

### Q : Existe-t-il des modèles 4x compatibles FP16 ?
**R :** Oui ! Essayez :
- **4x-AnimeSharp** (architecture ESRGAN - compatible FP16)
- **4x-UltraSharp** (architecture RealESRGAN - compatible FP16)
- **4x-NMKD-Siax** (architecture ESRGAN - compatible FP16)

Téléchargez-les depuis https://openmodeldb.info/ et placez-les dans `models/`

## 🎉 Profitez !

Vous pouvez maintenant upscaler vos contenus avec des modèles 2x, 4x, ou même 8x+ ! L'application détecte automatiquement le facteur et optimise le traitement.

**Besoin d'aide ?** Consultez [ADDING_MODELS.md](ADDING_MODELS.md) pour plus de détails techniques.
