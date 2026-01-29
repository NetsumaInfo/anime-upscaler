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

:: ============================================================================
:: STEP 1: Find compatible Python (3.10, 3.11, or 3.12)
:: ============================================================================
echo 🔍 Recherche d'une version Python compatible (3.10-3.12)...

set PYTHON_CMD=

:: Try py launcher with Python 3.12 first (preferred)
py -3.12 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.12
    for /f "tokens=*" %%i in ('py -3.12 --version') do echo ✅ %%i détecté via py -3.12
    goto :python_found
)

:: Try py launcher with Python 3.11
py -3.11 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.11
    for /f "tokens=*" %%i in ('py -3.11 --version') do echo ✅ %%i détecté via py -3.11
    goto :python_found
)

:: Try py launcher with Python 3.10
py -3.10 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3.10
    for /f "tokens=*" %%i in ('py -3.10 --version') do echo ✅ %%i détecté via py -3.10
    goto :python_found
)

:: Try default python command and check version
python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2 delims= " %%v in ('python --version') do set PY_VER=%%v
    for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
        set PY_MAJOR=%%a
        set PY_MINOR=%%b
    )
    :: Check if default python is compatible (3.10-3.12)
    if %PY_MAJOR% EQU 3 if %PY_MINOR% GEQ 10 if %PY_MINOR% LEQ 12 (
        set PYTHON_CMD=python
        for /f "tokens=*" %%i in ('python --version') do echo ✅ %%i détecté
        goto :python_found
    )
    echo ⚠️ Python %PY_VER% détecté mais non compatible avec PyTorch
)

:: No compatible Python found - try to install
echo ❌ Aucune version Python compatible trouvée (3.10-3.12 requis)
echo.
echo 🔧 Tentative d'installation automatique de Python 3.12...
winget install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements
if %errorlevel% neq 0 (
    echo.
    echo ❌ Installation automatique échouée!
    echo.
    echo Téléchargez Python 3.12 manuellement:
    echo https://www.python.org/downloads/release/python-3120/
    echo.
    echo ⚠️ IMPORTANT: Cochez "Add Python to PATH" lors de l'installation!
    echo.
    pause
    exit /b 1
)
echo.
echo ✅ Python 3.12 installé avec succès!
echo.
echo ⚠️ IMPORTANT: Fermez cette fenêtre et relancez install.bat
echo.
pause
exit /b 0

:python_found
echo    Commande Python: %PYTHON_CMD%

:: ============================================================================
:: STEP 2: Check FFmpeg
:: ============================================================================
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
    echo ✅ FFmpeg installé
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

:: ============================================================================
:: STEP 3: Create virtual environment
:: ============================================================================
echo.
echo 📦 Création de l'environnement virtuel...
if exist "venv" (
    echo ℹ️ Environnement virtuel existant détecté - suppression pour réinstallation propre...
    rmdir /s /q venv
)

%PYTHON_CMD% -m venv venv
if %errorlevel% neq 0 (
    echo ❌ Erreur lors de la création du venv!
    pause
    exit /b 1
)
echo ✅ Environnement virtuel créé avec %PYTHON_CMD%

:: ============================================================================
:: STEP 4: Activate venv
:: ============================================================================
echo.
echo 🔧 Activation de l'environnement virtuel...
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Fichier d'activation introuvable! Supprimez le dossier "venv" et relancez l'installation.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
echo ✅ Environnement virtuel activé

:: ============================================================================
:: STEP 5: Upgrade pip
:: ============================================================================
echo.
echo ⬆️ Mise à jour de pip, setuptools et wheel...
python -m pip install --upgrade pip setuptools wheel --quiet
if %errorlevel% neq 0 (
    echo ❌ Erreur lors de la mise à jour de pip!
    pause
    exit /b 1
)
echo ✅ Outils mis à jour

:: ============================================================================
:: STEP 6: Install PyTorch with CUDA
:: ============================================================================
echo.
echo 🔥 Installation de PyTorch avec CUDA 12.1...
echo    (Cela peut prendre plusieurs minutes...)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (
    echo ❌ Erreur lors de l'installation de PyTorch!
    pause
    exit /b 1
)
echo ✅ PyTorch installé

:: ============================================================================
:: STEP 7: Install other dependencies
:: ============================================================================
echo.
echo 📚 Installation des dépendances...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Erreur lors de l'installation des dépendances!
    pause
    exit /b 1
)
echo ✅ Dépendances installées

:: ============================================================================
:: STEP 8: Verify critical packages
:: ============================================================================
echo.
echo 🔍 Vérification des packages critiques...
python -c "import torch; print(f'   ✅ torch {torch.__version__}')" 2>nul || echo    ❌ torch manquant!
python -c "import torchvision; print(f'   ✅ torchvision {torchvision.__version__}')" 2>nul || echo    ❌ torchvision manquant!
python -c "import gradio; print(f'   ✅ gradio {gradio.__version__}')" 2>nul || echo    ❌ gradio manquant!
python -c "import spandrel; print(f'   ✅ spandrel {spandrel.__version__}')" 2>nul || echo    ❌ spandrel manquant!
python -c "import PIL; print(f'   ✅ pillow {PIL.__version__}')" 2>nul || echo    ❌ pillow manquant!
python -c "import numpy; print(f'   ✅ numpy {numpy.__version__}')" 2>nul || echo    ❌ numpy manquant!
python -c "import cv2; print(f'   ✅ opencv {cv2.__version__}')" 2>nul || echo    ❌ opencv manquant!

:: ============================================================================
:: STEP 9: Create directories
:: ============================================================================
echo.
echo 📁 Création des dossiers...
if not exist "models" mkdir models && echo ✅ Dossier "models" créé
if not exist "output" mkdir output && echo ✅ Dossier "output" créé

:: ============================================================================
:: STEP 10: Download models
:: ============================================================================
echo.
echo 📥 Téléchargement des modèles AI...
echo.

:: Model 1: Ani4K v2 Compact (RECOMMENDED)
if not exist "models\2x_Ani4Kv2_G6i2_Compact_107500.pth" (
    echo [1/5] Téléchargement de Ani4K v2 Compact RECOMMANDE... (~30 MB)
    curl -L --progress-bar -o "models\2x_Ani4Kv2_G6i2_Compact_107500.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/Ani4K-v2/2x_Ani4Kv2_G6i2_Compact_107500.pth" 2>nul
    if exist "models\2x_Ani4Kv2_G6i2_Compact_107500.pth" (echo ✅ Modèle 1/5 téléchargé) else (echo ⚠️ Échec - sera téléchargé au premier lancement)
) else (
    echo ✅ [1/5] Ani4K v2 Compact déjà présent
)

:: Model 2: AniToon
if not exist "models\2x_AniToon_RPLKSR_197500.pth" (
    echo [2/5] Téléchargement de AniToon... (~30 MB)
    curl -L --progress-bar -o "models\2x_AniToon_RPLKSR_197500.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/AniToon/2x_AniToon_RPLKSR_197500.pth" 2>nul
    if exist "models\2x_AniToon_RPLKSR_197500.pth" (echo ✅ Modèle 2/5 téléchargé) else (echo ⚠️ Échec - sera téléchargé au premier lancement)
) else (
    echo ✅ [2/5] AniToon déjà présent
)

:: Model 3: OpenProteus
if not exist "models\2x_OpenProteus_Compact_i2_70K.pth" (
    echo [3/5] Téléchargement de OpenProteus... (~30 MB)
    curl -L --progress-bar -o "models\2x_OpenProteus_Compact_i2_70K.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/OpenProteus/2x_OpenProteus_Compact_i2_70K.pth" 2>nul
    if exist "models\2x_OpenProteus_Compact_i2_70K.pth" (echo ✅ Modèle 3/5 téléchargé) else (echo ⚠️ Échec - sera téléchargé au premier lancement)
) else (
    echo ✅ [3/5] OpenProteus déjà présent
)

:: Model 4: AniSD
if not exist "models\2x_AniSD_RealPLKSR_140K.pth" (
    echo [4/5] Téléchargement de AniSD... (~30 MB)
    curl -L --progress-bar -o "models\2x_AniSD_RealPLKSR_140K.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/AniSD-RealPLKSR/2x_AniSD_RealPLKSR_140K.pth" 2>nul
    if exist "models\2x_AniSD_RealPLKSR_140K.pth" (echo ✅ Modèle 4/5 téléchargé) else (echo ⚠️ Échec - sera téléchargé au premier lancement)
) else (
    echo ✅ [4/5] AniSD déjà présent
)

:: Model 5: Ani4K v2 UltraCompact
if not exist "models\2x_Ani4Kv2_G6i2_UltraCompact_105K.pth" (
    echo [5/5] Téléchargement de Ani4K v2 UltraCompact... (~20 MB)
    curl -L --progress-bar -o "models\2x_Ani4Kv2_G6i2_UltraCompact_105K.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/Ani4K-v2/2x_Ani4Kv2_G6i2_UltraCompact_105K.pth" 2>nul
    if exist "models\2x_Ani4Kv2_G6i2_UltraCompact_105K.pth" (echo ✅ Modèle 5/5 téléchargé) else (echo ⚠️ Échec - sera téléchargé au premier lancement)
) else (
    echo ✅ [5/5] Ani4K v2 UltraCompact déjà présent
)

:: ============================================================================
:: STEP 11: Test CUDA
:: ============================================================================
echo.
echo 🔍 Vérification de CUDA...
python -c "import torch; cuda_ok = torch.cuda.is_available(); print('✅ CUDA disponible:', cuda_ok); print('   GPU:', torch.cuda.get_device_name(0) if cuda_ok else 'N/A')" 2>nul
if %errorlevel% neq 0 (
    echo ⚠️ Impossible de vérifier CUDA
)

:: ============================================================================
:: DONE
:: ============================================================================
echo.
echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║                    ✅ Installation terminée!                          ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.
echo 📝 Lancez "run.bat" pour démarrer l'application
echo.
pause
