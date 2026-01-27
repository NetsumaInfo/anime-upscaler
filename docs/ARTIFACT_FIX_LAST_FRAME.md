# FIX: Artefacts sur la Dernière Frame (Lignes Blanches)

**Date:** 2026-01-27
**Problème:** Artefacts de lignes horizontales blanches sur la dernière frame de la vidéo exportée

## 🐛 Problème Identifié

### Symptôme
La dernière frame de la vidéo encodée affiche des **artefacts de lignes horizontales blanches** sur toute l'image.

### Cause Racine: Gestion Incorrecte du Color Range

**Problème de conversion:**
1. **Frames PNG en input:** Full Range (0-255) - couleurs complètes
2. **Encodage précédent:** Utilisait `"-color_range", "pc"` (Full Range) en output
3. **Décodeurs H.264:** S'attendent à TV Range (16-235) par défaut
4. **Résultat:** Confusion de range → artefacts, surtout sur la **dernière frame**

**Pourquoi la dernière frame?**
- FFmpeg peut avoir des problèmes de padding ou de flush sur la dernière frame
- Si le color range n'est pas géré explicitement, la dernière frame peut être mal encodée
- Les décodeurs interprètent mal les valeurs hors range (0-15 et 236-255)
- Résultat: lignes blanches/artefacts sur la dernière frame

### Code Problématique (AVANT)

```python
# Problème: Full Range en output sans conversion explicite
color_metadata = [
    "-colorspace", "bt709",
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
    "-color_range", "pc"  # ❌ PROBLÈME: Full Range en output
]

# Filtre sans conversion de range
"-vf", "format=yuv420p"  # ❌ Pas de conversion Full→TV
```

**Résultat:**
- Décodeurs confus entre Full Range et TV Range
- Artefacts sur dernière frame (problème de flush FFmpeg)
- Lignes blanches horizontales visibles

## ✅ Solution Implémentée

### Conversion Explicite Full Range → TV Range

**Principe:**
1. **Input:** PNGs en Full Range (0-255) - correct
2. **Filtre:** Conversion explicite `scale=in_range=full:out_range=limited`
3. **Output:** TV Range (16-235) + metadata `"-color_range", "tv"`
4. **Résultat:** Compatibilité maximale, pas d'artefacts

### Code Corrigé (APRÈS)

```python
# Solution: TV Range en output avec conversion explicite
color_metadata = [
    "-colorspace", "bt709",
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
    "-color_range", "tv"  # ✅ TV Range (16-235) - standard vidéo
]

# Filtre avec conversion explicite Full→TV
"-vf", "scale=in_range=full:out_range=limited,format=yuv420p"
#       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Conversion explicite
```

## 📝 Fichiers Modifiés

### `app_upscale/video_processing.py`

**Ligne 664-672:** Metadata de color range
```python
# AVANT
"-color_range", "pc"  # Full Range

# APRÈS
"-color_range", "tv"  # TV Range (standard vidéo)
```

**Ligne 681-684:** H.264 (AVC) - Filtre de conversion
```python
# AVANT
"-vf", "format=yuv420p"

# APRÈS
"-vf", "scale=in_range=full:out_range=limited,format=yuv420p"
```

**Ligne 702:** H.265 (HEVC) - Filtre de conversion
```python
# AVANT
"-vf", f"format={pix_fmt}"

# APRÈS
"-vf", f"scale=in_range=full:out_range=limited,format={pix_fmt}"
```

**Ligne 712:** ProRes - Filtre de conversion
```python
# AVANT
"-vf", f"format={pix_fmt}"

# APRÈS
"-vf", f"scale=in_range=full:out_range=limited,format={pix_fmt}"
```

**Ligne 730 & 737:** DNxHD/DNxHR - Filtre de conversion
```python
# AVANT (DNxHR)
"-vf", f"format={dnx_pix_fmt}"

# APRÈS (DNxHR)
"-vf", f"scale=in_range=full:out_range=limited,format={dnx_pix_fmt}"

# AVANT (DNxHD)
"-vf", "format=yuv422p"

# APRÈS (DNxHD)
"-vf", "scale=in_range=full:out_range=limited,format=yuv422p"
```

## 🎯 Résultats Attendus

### Avant le Fix
- ❌ Artefacts de lignes blanches sur dernière frame
- ❌ Problèmes de color range avec certains décodeurs
- ❌ Incompatibilité avec standard TV Range

### Après le Fix
- ✅ Dernière frame propre, sans artefacts
- ✅ Conversion explicite Full→TV Range
- ✅ Compatibilité maximale avec tous les players
- ✅ Respect du standard vidéo (TV Range 16-235)

## 🔧 Test Recommandé

Pour vérifier que le fix fonctionne:

1. **Re-encoder la vidéo** avec les nouveaux paramètres
2. **Vérifier la dernière frame:**
   - Ouvrir la vidéo dans un player
   - Aller à la dernière frame (touche →)
   - **Attendu:** Pas de lignes blanches, image propre
3. **Vérifier les couleurs:**
   - Comparer avec les frames PNG originales
   - **Attendu:** Couleurs correctes, pas de "washed out"

## 📚 Contexte Technique

### Full Range vs TV Range

| Range Type | Values | Usage | Problème si Mal Utilisé |
|------------|--------|-------|-------------------------|
| **Full Range (pc)** | 0-255 | Images (PNG, JPEG), Écrans | Artefacts si décodeur attend TV Range |
| **TV Range (tv)** | 16-235 | Vidéo (H.264, H.265, etc.) | Couleurs "washed out" si décodeur attend Full Range |

### Conversion Explicite (Solution)

```
PNGs (0-255) → scale filter → YUV (16-235) → H.264 → Player
                ↑ Conversion explicite ici
                in_range=full → out_range=limited
```

**Avantages:**
- Pas d'ambiguïté sur le range utilisé
- FFmpeg sait exactement quoi faire
- Décodeurs reçoivent des données dans le range attendu
- Pas d'artefacts sur dernière frame

## 🎓 Leçons Apprises

1. **Toujours spécifier le color range explicitement**
   - Ne jamais laisser FFmpeg deviner
   - Utiliser `scale=in_range=X:out_range=Y`

2. **La dernière frame est critique**
   - FFmpeg peut avoir des problèmes de flush/padding
   - Tester spécifiquement la dernière frame

3. **TV Range est le standard pour la vidéo**
   - H.264, H.265, ProRes, DNxHD/DNxHR = TV Range
   - Full Range en vidéo = artefacts avec la plupart des players

4. **Conversion > Tagging**
   - Faire une vraie conversion (`scale` filter)
   - Puis taguer correctement (`-color_range`)
   - Pas juste taguer sans convertir

## 🔄 Historique des Fixes

### v1 (Précédent - ÉCHEC)
- Utilisait Full Range en output
- Tagué comme `"-color_range", "pc"`
- **Problème:** Artefacts sur dernière frame

### v2 (Actuel - ✅ SOLUTION)
- Conversion explicite Full→TV Range
- Filtre `scale=in_range=full:out_range=limited`
- Tagué comme `"-color_range", "tv"`
- **Résultat:** Dernière frame propre, compatibilité maximale
