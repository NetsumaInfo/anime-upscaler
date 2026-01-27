# Guide : Ajouter n'importe quel modèle d'upscaling

L'application Anime Upscaler supporte maintenant **tous les facteurs d'upscaling** (2x, 4x, 8x, etc.) grâce à la détection automatique de Spandrel.

## ✨ Nouveautés (Version 2.4)

- **Auto-détection du facteur d'upscaling** : Plus besoin de spécifier manuellement si un modèle est 2x, 4x, ou autre
- **Support universel** : Tous les modèles compatibles Spandrel fonctionnent automatiquement
- **Multi-passes intelligents** : L'app calcule automatiquement le nombre de passes nécessaires

## 📥 Méthode 1 : Ajout manuel (Glisser-déposer) ⭐ RECOMMANDÉ

La méthode la plus simple pour ajouter un modèle :

1. **Téléchargez** un modèle depuis [OpenModelDB](https://openmodeldb.info/) ou [Upscale-Hub](https://github.com/Sirosky/Upscale-Hub)
2. **Copiez** le fichier `.pth` ou `.safetensors` dans le dossier `models/`
3. **Redémarrez** l'application
4. Le modèle apparaît automatiquement dans la liste avec son facteur d'upscaling détecté !

### Exemple : Ajouter 4x-FaceUpSharpDAT

**Étape 1 : Télécharger le modèle**
- Allez sur [OpenModelDB - 4x-FaceUpSharpDAT](https://openmodeldb.info/models/4x-FaceUpSharpDAT)
- Cliquez sur "Download Model" (147.5 MB)
- Sauvegardez le fichier `4x-FaceUpSharpDAT.pth`

**Étape 2 : Placer dans le dossier models/**
```bash
# Windows
copy "Downloads\4x-FaceUpSharpDAT.pth" "S:\projet_app\app upscale\models\"

# Linux/Mac
cp ~/Downloads/4x-FaceUpSharpDAT.pth ./models/
```

**Étape 3 : Redémarrer l'app**
```bash
run.bat  # Windows
python app.py  # Linux/Mac
```

**Résultat au démarrage :**
```
📦 Scanning models...
✅ 4x-FaceUpSharpDAT loaded on cuda (FP16) - 4x upscale
```

Le modèle apparaît maintenant dans la liste déroulante de l'interface !

## 🔧 Méthode 2 : Ajout au code (Auto-téléchargement)

Pour que le modèle soit téléchargé automatiquement au premier usage :

1. Ouvrez `app.py`
2. Ajoutez votre modèle dans le dictionnaire `DEFAULT_MODELS` (ligne ~300)

```python
DEFAULT_MODELS = {
    # ... modèles existants ...

    # Votre nouveau modèle
    "4x-YourModel.pth": {
        "url": "https://example.com/path/to/model.pth",
        "description": "Description de votre modèle",
        "display_name": "Nom Affiché dans l'UI"
    },
}
```

### Exemple réel

```python
"4x-FaceUpSharpDAT.pth": {
    "url": "https://example.com/4x-FaceUpSharpDAT.pth",
    "description": "4x upscaling for faces and detailed anime art",
    "display_name": "4x FaceUpSharp DAT"
},
```

## 🎯 Modèles compatibles

L'application fonctionne avec **tous les modèles Spandrel** :

- **2x models** : AniToon, Ani4K, AniSD, AniScale2, OpenProteus
- **4x models** : FaceUpSharpDAT, Ani4Kv3, NMKD-Siax, etc.
- **Architectures** : ESRGAN, RealESRGAN, SwinIR, HAT, OmniSR, PLKSR, DAT, etc.

Consultez [OpenModelDB](https://openmodeldb.info/) pour explorer des milliers de modèles.

## ⚙️ Fonctionnement automatique

### Détection du scale

Quand un modèle est chargé :

```python
model, actual_fp16, scale = load_model("4x FaceUpSharp DAT")
# scale = 4 (détecté automatiquement par Spandrel)
```

### Multi-passes intelligents

Si vous demandez un upscaling plus grand que le scale du modèle :

| Modèle | Scale demandé | Nombre de passes | Résultat final |
|--------|---------------|------------------|----------------|
| 2x | ×4 | 2 passes | 2x → 2x = 4x |
| 4x | ×8 | 2 passes | 4x → 4x = 16x puis resize |
| 4x | ×2 | 1 passe | 4x puis downscale à 2x |

### Exemple d'utilisation

```python
# Image 480p avec modèle 4x
# Target : 1080p

# Calcul automatique :
# - 480p × 4 = 1920p (> 1080p)
# - 1 seule passe nécessaire
# - Resize final vers 1080p

# Avec modèle 2x :
# - 480p × 2 = 960p (< 1080p)
# - 960p × 2 = 1920p (> 1080p)
# - 2 passes nécessaires
# - Resize final vers 1080p
```

## 📝 Notes techniques

### Ce qui a changé (v2.4)

**Avant :**
- Scale hardcodé dans `DEFAULT_MODELS` avec `"scale": 2`
- Fonction `extract_scale_from_filename()` basée sur regex
- Support uniquement des modèles 2x

**Après :**
- Scale détecté automatiquement par `model_descriptor.scale`
- Plus besoin de `"scale"` dans la config
- Support universel de tous les facteurs (2x, 3x, 4x, 8x, etc.)

### Architectures supportées

Spandrel supporte automatiquement :

- **ESRGAN** (RealESRGAN, BSRGAN, etc.)
- **SwinIR** et variantes
- **HAT** (Hybrid Attention Transformer)
- **OmniSR**
- **PLKSR** et dérivés (RealPLKSR, Compact PLKSR)
- **DAT** (Dual Aggregation Transformer)
- **CRAFT** et **DITN**
- Et bien d'autres...

## 🚀 Exemples de modèles recommandés

### Pour l'anime moderne (2x)
- `2x_Ani4Kv2_G6i2_Compact_107500.pth` ⭐ (Recommandé)
- `2x_Ani4Kv2_G6i2_UltraCompact_105K.pth` (Rapide)

### Pour l'anime ancien (2x)
- `2x_AniToon_RPLKSRL_280K.pth` (Meilleure qualité)
- `2x_AniSD_RealPLKSR_140K.pth` (Anime SD/VHS)

### Pour les visages / photos (4x)
- `4x-FaceUpSharpDAT.pth`
- `4x-NMKD-Siax-CX`

### Général purpose (4x)
- `4x-AnimeSharp`
- `4x-UltraSharp`

## 🔗 Ressources

- **OpenModelDB** : https://openmodeldb.info/ (Base de données de modèles)
- **Upscale-Hub** : https://github.com/Sirosky/Upscale-Hub (Modèles anime)
- **Spandrel** : https://github.com/chaiNNer-org/spandrel (Moteur de chargement)
- **chaiNNer** : https://github.com/chaiNNer-org/chaiNNer (App similaire)

## ⚠️ Limitations et Solutions

### Téléchargement Google Drive
- **Problème** : Les liens Google Drive ne fonctionnent pas pour le téléchargement automatique (retournent une page HTML)
- **Solution** : Télécharger manuellement depuis OpenModelDB et placer dans `models/`

### Modèles très lourds (4x, 8x)
- **Problème** : Certains modèles 4x font 150-200+ MB
- **Solution** :
  - Assurez-vous d'avoir assez de VRAM (8GB+ recommandé pour 4x)
  - Utilisez FP16 pour réduire l'utilisation VRAM de 50%

### Tile Size avec modèles 4x
- **Recommandations** :
  - **2x models** : 512px tile (par défaut)
  - **4x models** : 256-384px tile recommandé
  - **8x models** : 128-256px tile recommandé
- **Pourquoi** : Les modèles avec facteur plus élevé consomment plus de VRAM par tile

### Modèles DAT avec FP16
- **Symptôme** : Erreurs `RuntimeError: expected scalar type Half but found Float` ou `IndexError: tensors used as indices`
- **Cause** : Les modèles DAT (Dual Aggregation Transformer) ont des composants internes qui créent des mismatches de dtype avec FP16
- **Solution** : CORRIGÉ dans v2.4.1 - Les modèles DAT sont automatiquement détectés et utilisent FP32 au lieu de FP16
- **Impact** : Modèles DAT utilisent 2x plus de VRAM mais fonctionnent de manière stable
- **Note** : 4x-FaceUpSharpDAT est un modèle DAT et utilisera automatiquement FP32

### Performance FP16 vs FP32
- **FP16** :
  - ✅ 50% moins de VRAM
  - ✅ Plus rapide sur GPU NVIDIA récents (RTX series)
  - ⚠️ Légère perte de précision (négligeable pour upscaling)
- **FP32** :
  - ✅ Précision maximale
  - ❌ 2x plus de VRAM
  - ❌ Plus lent sur GPU modernes
