@echo off
chcp 65001 >nul
echo ========================================
echo   智能体检报告系统 - 全新安装
echo   Python 3.11 + 零冲突依赖
echo ========================================
echo.

echo [1/3] 升级 pip...
python -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
echo.

echo [2/3] 安装项目依赖（约 5-10 分钟）...
echo 使用清华镜像加速...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir
if errorlevel 1 (
    echo.
    echo ❌ 依赖安装失败！请检查网络或重试
    pause
    exit /b 1
)
echo ✅ 依赖安装成功
echo.

echo [3/3] 验证安装...
python -c "import fastapi, gradio, easyocr, fitz; print('✅ 所有核心库导入成功')"
if errorlevel 1 (
    echo ❌ 验证失败！请检查错误信息
    pause
    exit /b 1
)
echo.

echo ========================================
echo   ✅ 全新安装完成！
echo ========================================
echo.
echo 快速启动：
echo   后端: python app.py
echo   前端: python frontend\gradio_app.py
echo.
echo 或使用一键启动脚本：
echo   start-backend.bat 和 start-frontend.bat
echo.
pause
