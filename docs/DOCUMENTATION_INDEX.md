# 📚 Index de la Documentation

Guide pour naviguer dans la documentation complète d'Anime Upscaler.

---

## 🎯 Documentation Principale

### Pour Utilisateurs

1. **[../README.md](../README.md)** ⭐ **COMMENCEZ ICI**
   - Installation rapide (Windows/Linux/macOS)
   - Guide d'utilisation en 5 étapes
   - Nouveautés version 2.7.1
   - Résolution problèmes courants

2. **[ADVANCED.md](ADVANCED.md)** 📖 **GUIDE COMPLET**
   - Tous les modèles en détail (10 modèles)
   - Multi-scale support technique
   - Mode précision FP16/FP32 expliqué
   - Post-processing professionnel
   - Export vidéo codecs détaillés
   - Optimisations performance

3. **[ADDING_MODELS.md](ADDING_MODELS.md)**
   - Comment ajouter vos propres modèles
   - Sources de modèles (Upscale-Hub, OpenModelDB)
   - Compatibilité et formats

### Pour Développeurs

4. **[../CLAUDE.md](../CLAUDE.md)** 🔬 **DOCUMENTATION DÉVELOPPEUR**
   - Architecture modulaire (10 modules)
   - Pipeline concurrent 4-étages
   - Dépendances entre modules
   - Guide de modification

5. **[CHANGELOG.md](CHANGELOG.md)**
   - Historique complet des versions
   - Détails techniques des changements
   - Notes de migration

6. **[PARALLEL_VIDEO_PROCESSING.md](PARALLEL_VIDEO_PROCESSING.md)**
   - Documentation pipeline concurrent
   - Architecture 4-étages
   - Performance et optimisations

7. **[CHANGELOG_OPTIMIZATIONS.md](CHANGELOG_OPTIMIZATIONS.md)**
   - Détails optimisations v2.4.2
   - Benchmarks détaillés
   - Modifications code

---

## 📂 Par Sujet

### Installation & Configuration
- **Installation:** [README.md § Installation](../README.md#-démarrage-rapide)
- **Prérequis système:** [README.md § Prérequis](../README.md#installation-windows)
- **Résolution problèmes:** [README.md § Troubleshooting](../README.md#-résolution-de-problèmes)

### Pipeline Concurrent (v2.7+)
- **Vue d'ensemble:** [README.md § Nouveautés](../README.md#-nouveautés-version-271)
- **Documentation technique:** [PARALLEL_VIDEO_PROCESSING.md](PARALLEL_VIDEO_PROCESSING.md)
- **Détails d'implémentation:** [CLAUDE.md § Pipeline](../CLAUDE.md#9-pipelinepy-740-lines---tier-4-new-in-v27)

### Modèles IA
- **Vue d'ensemble:** [README.md § Modèles](../README.md#3-choisir-un-modèle)
- **Descriptions complètes:** [ADVANCED.md § Modèles](ADVANCED.md#-modèles-ia-en-détail)
- **Ajouter modèles:** [ADDING_MODELS.md](ADDING_MODELS.md)

### Multi-Scale (×1, ×2, ×4, ×8, ×16)
- **Vue d'ensemble:** [README.md § Image Scale](../README.md#4-configurer-les-paramètres)
- **Technique détaillée:** [ADVANCED.md § Multi-Scale](ADVANCED.md#-multi-scale-support)

### Mode de Précision (FP16/FP32)
- **Utilisation basique:** [README.md § Précision](../README.md#mode-de-précision-avancé)
- **Explications détaillées:** [ADVANCED.md § Précision](ADVANCED.md#-mode-de-précision-fp16fp32)

### Export Vidéo
- **Codecs disponibles:** [README.md § Export Vidéo](../README.md#-export-vidéo)
- **Détails techniques:** [ADVANCED.md § Export Vidéo](ADVANCED.md#-export-vidéo-professionnel)

### Performance & Optimisation
- **Architecture modulaire:** [CLAUDE.md](../CLAUDE.md)
- **Optimisations CUDA:** [CHANGELOG_OPTIMIZATIONS.md](CHANGELOG_OPTIMIZATIONS.md)
- **Pipeline concurrent:** [PARALLEL_VIDEO_PROCESSING.md](PARALLEL_VIDEO_PROCESSING.md)

---

## 🔍 Par Question Fréquente

### "Comment démarrer rapidement?"
→ [README.md](../README.md) sections Installation et Guide d'utilisation

### "Quel modèle choisir pour mon anime?"
→ [ADVANCED.md § Modèles IA](ADVANCED.md#-modèles-ia-en-détail)

### "Comment avoir le traitement vidéo le plus rapide?"
→ [README.md § Pipeline Concurrent](../README.md#-nouveautés-version-271)
→ [PARALLEL_VIDEO_PROCESSING.md](PARALLEL_VIDEO_PROCESSING.md)

### "FP16 ou FP32? Quelle différence?"
→ [ADVANCED.md § Précision](ADVANCED.md#-mode-de-précision-fp16fp32)

### "Erreur Out of Memory (OOM)?"
→ [README.md § Troubleshooting](../README.md#erreur-out-of-memory-oom)
→ [ADVANCED.md § Tile Settings](ADVANCED.md#-tile-processing-system)

### "Quel codec vidéo utiliser?"
→ [ADVANCED.md § Export Vidéo](ADVANCED.md#-export-vidéo-professionnel)

### "Comment ajouter mes propres modèles?"
→ [ADDING_MODELS.md](ADDING_MODELS.md)

---

## 🗺️ Parcours Recommandés

### Parcours Débutant (15 min)
1. [README.md](../README.md) - Installation et guide complet
2. Tester l'application!

### Parcours Utilisateur Avancé (30-40 min)
1. [README.md](../README.md) - Révision rapide
2. [ADVANCED.md § Modèles](ADVANCED.md#-modèles-ia-en-détail) - Choisir meilleur modèle
3. [ADVANCED.md § Export Vidéo](ADVANCED.md#-export-vidéo-professionnel) - Codecs pro

### Parcours Développeur (60 min)
1. [CLAUDE.md](../CLAUDE.md) - Architecture complète
2. [CHANGELOG.md](CHANGELOG.md) - Historique technique
3. [PARALLEL_VIDEO_PROCESSING.md](PARALLEL_VIDEO_PROCESSING.md) - Pipeline concurrent
4. Code source avec nouveau contexte

---

## 📁 Fichiers Archivés

Les documents suivants sont archivés dans `docs/archive/` :
- Notes de correctifs v2.7.1 (intégrées dans CHANGELOG.md)
- Documentation de développement temporaire

---

**Version documentation:** 2.7.1
**Dernière mise à jour:** 2026-01-29
**Langue:** Français
