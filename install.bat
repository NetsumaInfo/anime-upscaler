@echo off
chcp 65001 >nul
title 🎨 Installation - Anime Upscaler

echo.
echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║            🎨 Anime Upscaler - Installation                          ║
echo ║         Optimisé pour NVIDIA CUDA                                    ║
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

:: Download models
echo.
echo 📥 Téléchargement des modèles AI par défaut...
echo    (Les modèles peuvent aussi être ajoutés manuellement dans le dossier "models")
echo.

:: Model 1: AnimeSharpV4 RCAN
if not exist "models\2x-AnimeSharpV4_RCAN.safetensors" (
    echo [1/3] Téléchargement de 2x-AnimeSharpV4_RCAN... (~90 MB)
    curl -L --progress-bar -o "models\2x-AnimeSharpV4_RCAN.safetensors" "https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV4/2x-AnimeSharpV4_RCAN.safetensors"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 1/3 téléchargé
    )
) else (
    echo ✅ [1/3] 2x-AnimeSharpV4_RCAN déjà présent
)

:: Model 2: AnimeSharpV4 Fast
if not exist "models\2x-AnimeSharpV4_Fast_RCAN_PU.safetensors" (
    echo [2/3] Téléchargement de 2x-AnimeSharpV4_Fast_RCAN_PU... (~90 MB)
    curl -L --progress-bar -o "models\2x-AnimeSharpV4_Fast_RCAN_PU.safetensors" "https://github.com/Kim2091/Kim2091-Models/releases/download/2x-AnimeSharpV4/2x-AnimeSharpV4_Fast_RCAN_PU.safetensors"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 2/3 téléchargé
    )
) else (
    echo ✅ [2/3] 2x-AnimeSharpV4_Fast_RCAN_PU déjà présent
)

:: Model 3: Ani4VK v2 Compact
if not exist "models\2x_Ani4Kv2_Compact.pth" (
    echo [3/3] Téléchargement de 2x_Ani4Kv2_Compact... (~30 MB)
    curl -L --progress-bar -o "models\2x_Ani4Kv2_Compact.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/Ani4K-v2/2x_Ani4Kv2_G6i2_Compact_107500.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 3/3 téléchargé
    )
) else (
    echo ✅ [3/3] 2x_Ani4Kv2_Compact déjà présent
)

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
