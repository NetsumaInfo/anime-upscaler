@echo off
chcp 65001 >nul
title 🎨 Anime Upscaler

echo.
echo ╔══════════════════════════════════════════════════════════════════════╗
echo ║                     🎨 Anime Upscaler                                 ║
echo ╚══════════════════════════════════════════════════════════════════════╝
echo.

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Environnement virtuel non trouvé!
    echo.
    echo Lancez d'abord "install.bat" pour installer l'application.
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment
echo 🔧 Activation de l'environnement virtuel...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ❌ Erreur lors de l'activation du venv!
    echo.
    echo Essayez de supprimer le dossier "venv" et relancez "install.bat"
    echo.
    pause
    exit /b 1
)

:: Check if app.py exists
if not exist "app.py" (
    echo ❌ Fichier app.py introuvable!
    echo.
    pause
    exit /b 1
)

:: Run the application
echo ✅ Environnement prêt
echo.
echo 🚀 Lancement de l'application...
echo    (L'interface web s'ouvrira automatiquement dans votre navigateur)
echo.
echo ⚠️ NE FERMEZ PAS cette fenêtre tant que vous utilisez l'application!
echo    Pour arrêter l'application, fermez cette fenêtre ou appuyez sur Ctrl+C
echo.
echo ════════════════════════════════════════════════════════════════════════
echo.

python app.py

if %errorlevel% neq 0 (
    echo.
    echo ════════════════════════════════════════════════════════════════════════
    echo.
    echo ❌ L'application s'est arrêtée avec une erreur!
    echo.
    echo Si l'erreur persiste:
    echo    1. Vérifiez que les dépendances sont installées: install.bat
    echo    2. Vérifiez que FFmpeg est installé: ffmpeg -version
    echo    3. Consultez les messages d'erreur ci-dessus
    echo.
)

pause
