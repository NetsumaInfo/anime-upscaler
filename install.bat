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
    echo Installez Python 3.10, 3.11 ou 3.12 depuis https://www.python.org/downloads/
    echo ⚠️ IMPORTANT: Cochez "Add Python to PATH" lors de l'installation!
    echo.
    pause
    exit /b 1
)

:: Display Python version and check compatibility
for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo ✅ %PYTHON_VERSION% détecté

:: Check Python version compatibility (PyTorch requires 3.8-3.12)
for /f "tokens=2 delims= " %%v in ('python --version') do set PY_VER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)

:: Python 3.13+ is NOT supported by PyTorch yet
if %PY_MAJOR% GEQ 3 if %PY_MINOR% GEQ 13 (
    echo.
    echo ⚠️ ════════════════════════════════════════════════════════════════════
    echo ⚠️  ATTENTION: Python %PY_VER% n'est PAS compatible avec PyTorch!
    echo ⚠️  PyTorch supporte actuellement Python 3.8 à 3.12 uniquement.
    echo ⚠️  
    echo ⚠️  Veuillez installer Python 3.10, 3.11 ou 3.12:
    echo ⚠️  https://www.python.org/downloads/release/python-3120/
    echo ⚠️ ════════════════════════════════════════════════════════════════════
    echo.
    pause
    exit /b 1
)

:: Python 3.8-3.9 works but 3.10+ recommended
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 (
    echo ⚠️ Python %PY_VER% fonctionne mais Python 3.10+ est recommandé
)

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
    echo    Tentative avec options de secours...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    if %errorlevel% neq 0 (
        echo ❌ Échec définitif de l'installation de PyTorch!
        pause
        exit /b 1
    )
)
echo ✅ PyTorch installé

:: Install other dependencies
echo.
echo 📚 Installation des dépendances principales...
echo    (Cela peut prendre quelques minutes...)
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo ⚠️ Installation silencieuse échouée, nouvelle tentative avec logs...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo ❌ Erreur lors de l'installation des dépendances!
        pause
        exit /b 1
    )
)
echo ✅ Dépendances principales installées

:: Install spandrel extras (for model compatibility)
echo.
echo 🔧 Installation de spandrel avec extras (compatibilité modèles)...
pip install "spandrel[opencv,pillow]" --quiet 2>nul
if %errorlevel% neq 0 (
    echo ℹ️ Extras non disponibles (non critique)
)

:: Verify critical packages
echo.
echo 🔍 Vérification des packages critiques...
python -c "import torch; print(f'   ✅ torch {torch.__version__}')" 2>nul || echo    ❌ torch manquant!
python -c "import torchvision; print(f'   ✅ torchvision {torchvision.__version__}')" 2>nul || echo    ❌ torchvision manquant!
python -c "import gradio; print(f'   ✅ gradio {gradio.__version__}')" 2>nul || echo    ❌ gradio manquant!
python -c "import spandrel; print(f'   ✅ spandrel {spandrel.__version__}')" 2>nul || echo    ❌ spandrel manquant!
python -c "import PIL; print(f'   ✅ pillow {PIL.__version__}')" 2>nul || echo    ❌ pillow manquant!
python -c "import numpy; print(f'   ✅ numpy {numpy.__version__}')" 2>nul || echo    ❌ numpy manquant!
python -c "import cv2; print(f'   ✅ opencv {cv2.__version__}')" 2>nul || echo    ❌ opencv manquant!
python -c "import tqdm; print(f'   ✅ tqdm {tqdm.__version__}')" 2>nul || echo    ❌ tqdm manquant!
python -c "import safetensors; print(f'   ✅ safetensors (installed)')" 2>nul || echo    ❌ safetensors manquant!
python -c "import einops; print(f'   ✅ einops (installed)')" 2>nul || echo    ❌ einops manquant!
python -c "import requests; print(f'   ✅ requests {requests.__version__}')" 2>nul || echo    ❌ requests manquant!
python -c "import gradio_imageslider; print(f'   ✅ gradio_imageslider (installed)')" 2>nul || echo    ❌ gradio_imageslider manquant!

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

:: Download models from OpenModelDB / Upscale-Hub
echo.
echo 📥 Téléchargement des modèles AI...
echo    (Les modèles peuvent aussi être ajoutés manuellement dans le dossier "models")
echo.

:: Model 1: AniToon Small (Fast, for old/low-quality anime)
if not exist "models\2x_AniToon_RPLKSRS_242500.pth" (
    echo [1/9] Téléchargement de AniToon Small... (~9 MB)
    curl -L --progress-bar -o "models\2x_AniToon_RPLKSRS_242500.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/AniToon/2x_AniToon_RPLKSRS_242500.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 1/9 téléchargé
    )
) else (
    echo ✅ [1/9] AniToon Small déjà présent
)

:: Model 2: AniToon (Balanced, for old/low-quality anime)
if not exist "models\2x_AniToon_RPLKSR_197500.pth" (
    echo [2/9] Téléchargement de AniToon... (~30 MB)
    curl -L --progress-bar -o "models\2x_AniToon_RPLKSR_197500.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/AniToon/2x_AniToon_RPLKSR_197500.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 2/9 téléchargé
    )
) else (
    echo ✅ [2/9] AniToon déjà présent
)

:: Model 3: AniToon Large (Best quality, for old/low-quality anime)
if not exist "models\2x_AniToon_RPLKSRL_280K.pth" (
    echo [3/9] Téléchargement de AniToon Large... (~66 MB)
    curl -L --progress-bar -o "models\2x_AniToon_RPLKSRL_280K.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/AniToon/2x_AniToon_RPLKSRL_280K.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 3/9 téléchargé
    )
) else (
    echo ✅ [3/9] AniToon Large déjà présent
)

:: Model 4: Ani4K v2 UltraCompact (Very fast, for modern anime)
if not exist "models\2x_Ani4Kv2_G6i2_UltraCompact_105K.pth" (
    echo [4/9] Téléchargement de Ani4K v2 UltraCompact... (~20 MB)
    curl -L --progress-bar -o "models\2x_Ani4Kv2_G6i2_UltraCompact_105K.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/Ani4K-v2/2x_Ani4Kv2_G6i2_UltraCompact_105K.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 4/9 téléchargé
    )
) else (
    echo ✅ [4/9] Ani4K v2 UltraCompact déjà présent
)

:: Model 5: Ani4K v2 Compact (RECOMMENDED - Balanced speed/quality)
if not exist "models\2x_Ani4Kv2_G6i2_Compact_107500.pth" (
    echo [5/9] Téléchargement de Ani4K v2 Compact RECOMMANDE... (~30 MB)
    curl -L --progress-bar -o "models\2x_Ani4Kv2_G6i2_Compact_107500.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/Ani4K-v2/2x_Ani4Kv2_G6i2_Compact_107500.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 5/9 téléchargé - RECOMMANDE
    )
) else (
    echo ✅ [5/9] Ani4K v2 Compact déjà présent - RECOMMANDE
)

:: Model 6: AniSD AC (For SD anime - clean sources)
if not exist "models\2x_AniSD_AC_RealPLKSR_127500.pth" (
    echo [6/9] Téléchargement de AniSD AC... (~30 MB)
    curl -L --progress-bar -o "models\2x_AniSD_AC_RealPLKSR_127500.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/AniSD-RealPLKSR/2x_AniSD_AC_RealPLKSR_127500.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 6/9 téléchargé
    )
) else (
    echo ✅ [6/9] AniSD AC déjà présent
)

:: Model 7: AniSD (For SD anime - general)
if not exist "models\2x_AniSD_RealPLKSR_140K.pth" (
    echo [7/9] Téléchargement de AniSD... (~30 MB)
    curl -L --progress-bar -o "models\2x_AniSD_RealPLKSR_140K.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/AniSD-RealPLKSR/2x_AniSD_RealPLKSR_140K.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 7/9 téléchargé
    )
) else (
    echo ✅ [7/9] AniSD déjà présent
)

:: Model 8: OpenProteus (Free alternative to Topaz Proteus)
if not exist "models\2x_OpenProteus_Compact_i2_70K.pth" (
    echo [8/9] Téléchargement de OpenProteus... (~30 MB)
    curl -L --progress-bar -o "models\2x_OpenProteus_Compact_i2_70K.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/OpenProteus/2x_OpenProteus_Compact_i2_70K.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 8/9 téléchargé
    )
) else (
    echo ✅ [8/9] OpenProteus déjà présent
)

:: Model 9: AniScale2 Compact (Fast general purpose)
if not exist "models\2x_AniScale2S_Compact_i8_60K.pth" (
    echo [9/9] Téléchargement de AniScale2 Compact... (~25 MB)
    curl -L --progress-bar -o "models\2x_AniScale2S_Compact_i8_60K.pth" "https://github.com/Sirosky/Upscale-Hub/releases/download/AniScale2/2x_AniScale2S_Compact_i8_60K.pth"
    if %errorlevel% neq 0 (
        echo ⚠️ Échec du téléchargement - le modèle sera téléchargé au premier lancement
    ) else (
        echo ✅ Modèle 9/9 téléchargé
    )
) else (
    echo ✅ [9/9] AniScale2 Compact déjà présent
)

echo.
echo ℹ️ Total: 9 modèles configurés
echo    Modèle recommandé: Ani4K v2 Compact (équilibre vitesse/qualité)

:: Test CUDA availability
echo.
echo 🔍 Vérification de CUDA...
python -c "import torch; cuda_ok = torch.cuda.is_available(); print('✅ CUDA disponible:', cuda_ok); print('   GPU:', torch.cuda.get_device_name(0) if cuda_ok else 'N/A'); print('   VRAM:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if cuda_ok else 'N/A')" 2>nul
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
