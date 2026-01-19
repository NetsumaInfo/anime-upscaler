@echo off
chcp 65001 >nul
title 🎨 Installation - Anime Upscaler

:: Change to script directory
cd /d "%~dp0"

echo.
echo ╔══════════════════════════════════════════════════════════════════════╗
echo           🎨 Anime Upscaler - Installation                         
echo           Optimisé pour NVIDIA CUDA                                    
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.

:: Check Python
echo 🔍 Vérification de Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python non trouvé!
    echo.
    echo Installez Python 3.10+ depuis https://www.python.org/downloads/
    echo ⚠️ IMPORTANT: Cochez "Add Python to PATH" lors de l'installation!
    echo.
    pause
    exit /b 1
)

:: Display Python version
for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ %PYTHON_VERSION% détecté

:: Check FFmpeg
echo.
echo 🔍 Vérification de FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️ FFmpeg non trouvé! Tentative d'installation...
    winget install --id Gyan.FFmpeg -e --silent
    if %errorlevel% neq 0 (
        echo.
        echo ❌ Installation automatique de FFmpeg échouée!
        echo.
        echo Téléchargez FFmpeg manuellement depuis: https://www.gyan.dev/ffmpeg/builds/
        echo 1. Téléchargez "ffmpeg-release-essentials.zip"
        echo 2. Extrayez le dossier
        echo 3. Ajoutez le dossier "bin" à votre PATH système
        echo.
        pause
        exit /b 1
    )
) else (
    echo ✅ FFmpeg détecté
)

:: Check FFprobe
ffprobe -version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ FFprobe non trouvé! Réinstallez FFmpeg.
    pause
    exit /b 1
) else (
    echo ✅ FFprobe détecté
)

:: Create virtual environment
echo.
echo 📦 Création de l'environnement virtuel...
if not exist "venv" (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ Erreur lors de la création du venv!
        pause
        exit /b 1
    )
    echo ✅ Environnement virtuel créé
) else (
    echo ℹ️ Environnement virtuel existant détecté
)

:: Activate venv
echo.
echo 🔧 Activation de l'environnement virtuel...
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Fichier d'activation introuvable! Supprimez le dossier "venv" et relancez l'installation.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
echo ✅ Environnement virtuel activé

:: Upgrade pip
echo.
echo ⬆️ Mise à jour de pip, setuptools et wheel...
python -m pip install --upgrade pip setuptools wheel --quiet
if %errorlevel% neq 0 (
    echo ❌ Erreur lors de la mise à jour de pip!
    pause
    exit /b 1
)
echo ✅ Outils mis à jour

:: Install PyTorch with CUDA
echo.
echo 🔥 Installation de PyTorch avec CUDA 12.1...
echo    (Cela peut prendre quelques minutes...)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
if %errorlevel% neq 0 (
    echo ❌ Erreur lors de l'installation de PyTorch!
    pause
    exit /b 1
)
echo ✅ PyTorch installé

:: Install other dependencies
echo.
echo 📚 Installation des dépendances...
echo    (Cela peut prendre quelques minutes...)
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo ❌ Erreur lors de l'installation des dépendances!
    pause
    exit /b 1
)
echo ✅ Dépendances installées

:: Create directories
echo.
echo 📁 Création des dossiers...
if not exist "models" (
    mkdir models
    echo ✅ Dossier "models" créé
) else (
    echo ℹ️ Dossier "models" existant
)

if not exist "output" (
    mkdir output
    echo ✅ Dossier "output" créé
) else (
    echo ℹ️ Dossier "output" existant
)

:: Download models from Upscale-Hub
echo.
echo 📥 Téléchargement des modèles AI depuis Upscale-Hub (https://github.com/Sirosky/Upscale-Hub)...
echo    (Les modèles peuvent aussi être ajoutés manuellement dans le dossier "models")
echo.

:: Model 1: AniToon Small (Fast, for old/low-quality anime)
if not exist "models\2x_AniToon_RPLKSRS_242500.pth" (
    echo [1/10] Téléchargement de AniToon Small... (~9 MB)
    curl -L --progress-bar -o "models\2x_AniToon_RPLKSRS_242500.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/AniToon/2x_AniToon_RPLKSRS_242500.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 1/10 téléchargé
    )
) else (
    echo ✅ [1/10] AniToon Small déjà présent
)

:: Model 2: AniToon (Balanced, for old/low-quality anime)
if not exist "models\2x_AniToon_RPLKSR_197500.pth" (
    echo [2/10] Téléchargement de AniToon... (~30 MB)
    curl -L --progress-bar -o "models\2x_AniToon_RPLKSR_197500.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/AniToon/2x_AniToon_RPLKSR_197500.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 2/10 téléchargé
    )
) else (
    echo ✅ [2/10] AniToon déjà présent
)

:: Model 3: AniToon Large (Best quality, for old/low-quality anime)
if not exist "models\2x_AniToon_RPLKSRL_280K.pth" (
    echo [3/10] Téléchargement de AniToon Large... (~66 MB)
    curl -L --progress-bar -o "models\2x_AniToon_RPLKSRL_280K.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/AniToon/2x_AniToon_RPLKSRL_280K.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 3/10 téléchargé
    )
) else (
    echo ✅ [3/10] AniToon Large déjà présent
)

:: Model 4: Ani4K v2 UltraCompact (Very fast, for modern anime)
if not exist "models\2x_Ani4Kv2_G6i2_UltraCompact_105K.pth" (
    echo [4/10] Téléchargement de Ani4K v2 UltraCompact... (~20 MB)
    curl -L --progress-bar -o "models\2x_Ani4Kv2_G6i2_UltraCompact_105K.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/Ani4K-v2/2x_Ani4Kv2_G6i2_UltraCompact_105K.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 4/10 téléchargé
    )
) else (
    echo ✅ [4/10] Ani4K v2 UltraCompact déjà présent
)

:: Model 5: Ani4K v2 Compact (RECOMMENDED - Balanced speed/quality)
if not exist "models\2x_Ani4Kv2_G6i2_Compact_107500.pth" (
    echo [5/10] Téléchargement de Ani4K v2 Compact RECOMMANDÉ... (~30 MB)
    curl -L --progress-bar -o "models\2x_Ani4Kv2_G6i2_Compact_107500.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/Ani4K-v2/2x_Ani4Kv2_G6i2_Compact_107500.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 5/10 téléchargé - RECOMMANDÉ
    )
) else (
    echo ✅ [5/10] Ani4K v2 Compact déjà présent - RECOMMANDÉ
)

:: Model 6: AniSD AC (For SD anime - clean sources)
if not exist "models\2x_AniSD_AC_RealPLKSR_127500.pth" (
    echo [6/10] Téléchargement de AniSD AC... (~30 MB)
    curl -L --progress-bar -o "models\2x_AniSD_AC_RealPLKSR_127500.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/AniSD-RealPLKSR/2x_AniSD_AC_RealPLKSR_127500.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 6/10 téléchargé
    )
) else (
    echo ✅ [6/10] AniSD AC déjà présent
)

:: Model 7: AniSD (For SD anime - general)
if not exist "models\2x_AniSD_RealPLKSR_140K.pth" (
    echo [7/10] Téléchargement de AniSD... (~30 MB)
    curl -L --progress-bar -o "models\2x_AniSD_RealPLKSR_140K.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/AniSD-RealPLKSR/2x_AniSD_RealPLKSR_140K.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 7/10 téléchargé
    )
) else (
    echo ✅ [7/10] AniSD déjà présent
)

:: Model 8: OpenProteus (Free alternative to Topaz Proteus)
if not exist "models\2x_OpenProteus_Compact_i2_70K.pth" (
    echo [8/10] Téléchargement de OpenProteus... (~30 MB)
    curl -L --progress-bar -o "models\2x_OpenProteus_Compact_i2_70K.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/OpenProteus/2x_OpenProteus_Compact_i2_70K.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 8/10 téléchargé
    )
) else (
    echo ✅ [8/10] OpenProteus déjà présent
)

:: Model 9: AniScale2 Compact (Fast general purpose)
if not exist "models\2x_AniScale2S_Compact_i8_60K.pth" (
    echo [9/10] Téléchargement de AniScale2 Compact... (~25 MB)
    curl -L --progress-bar -o "models\2x_AniScale2S_Compact_i8_60K.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/AniScale2/2x_AniScale2S_Compact_i8_60K.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 9/10 téléchargé
    )
) else (
    echo ✅ [9/10] AniScale2 Compact déjà présent
)

echo.
echo ℹ️ Total: 10 modèles configurés depuis Upscale-Hub
echo    Modèle recommandé: Ani4K v2 Compact (équilibre vitesse/qualité)

:: Test CUDA availability
echo.
echo 🔍 Vérification de CUDA...
python -c "import torch; print('✅ CUDA disponible:', torch.cuda.is_available()); print('   GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>nul
if %errorlevel% neq 0 (
    echo ⚠️ Impossible de vérifier CUDA - vérifiez que PyTorch est installé
)

echo.
echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║                    ✅ Installation terminée!                          ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.
echo 📝 Instructions:
echo    1. Lancez "run.bat" pour démarrer l'application
echo    2. L'interface web s'ouvrira automatiquement dans votre navigateur
echo    3. Ajoutez vos propres modèles dans le dossier "models" si nécessaire
echo       (formats supportés: .pth, .safetensors)
echo.
echo 🌐 Les modèles manquants seront téléchargés automatiquement au premier usage
echo.
pause
