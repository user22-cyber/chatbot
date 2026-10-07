@echo off
chcp 65001 > nul
setlocal

rem 이 배치 파일이 있는 프로젝트 폴더로 이동합니다.
cd /d "%~dp0"

rem uv가 사용하는 캐시와 Python 런타임을 프로젝트 내부에 둡니다.
set "UV_CACHE_DIR=%~dp0.uv-cache"
set "UV_PYTHON_INSTALL_DIR=%~dp0.uv-python"
set "STREAMLIT_CONFIG_DIR=%~dp0.streamlit"
set "PYTHONUTF8=1"

rem 기본 설치 경로를 먼저 확인하고, 없으면 PATH에서 uv를 찾습니다.
set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
if not exist "%UV_EXE%" set "UV_EXE=uv"

echo RAG 챗봇을 실행합니다...
echo 브라우저 주소: http://localhost:8501
echo 이 창을 닫으면 챗봇이 종료됩니다.
echo.

"%UV_EXE%" run streamlit run app.py --server.headless false --server.port 8501 --browser.gatherUsageStats false

if errorlevel 1 (
    echo.
    echo 실행 중 오류가 발생했습니다. uv 설치와 .env의 OPENAI_API_KEY를 확인하세요.
    pause
)

endlocal
