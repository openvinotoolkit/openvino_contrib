@echo off
set OV=c:\Users\VAIBHA~1\OneDrive\DOCUME~1\MASTER\OPENVI~1\OPENVI~1\modules\OLLAMA~1\OPENVI~1\OPENVI~1.0_X\runtime
set CGO_ENABLED=1
set CGO_CFLAGS=-I%OV%\include
set CGO_CXXFLAGS=-I%OV%\include
set CGO_LDFLAGS=-L%OV%\lib\intel64\Release
set PATH=C:\msys64\ucrt64\bin;%PATH%
set CC=C:\msys64\mingw64\bin\gcc.exe
set CXX=C:\msys64\mingw64\bin\g++.exe
set CGO_LDFLAGS=%CGO_LDFLAGS% -static-libgcc -static-libstdc++ -Wl,-Bstatic -lpthread -Wl,-Bdynamic

echo OV_ROOT = %OV%
echo CGO_CFLAGS = %CGO_CFLAGS%
echo CGO_LDFLAGS = %CGO_LDFLAGS%
echo CC = %CC%
echo.

echo Building ollama_genai_runner.exe ...
go build -v -o ollama_genai_runner.exe .\cmd\genai
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo GENAI RUNNER BUILD FAILED
    exit /b %ERRORLEVEL%
)
echo GENAI RUNNER BUILD OK

echo.
echo Building ollama.exe ...
go build -trimpath -ldflags "-s -w" -o ollama_new.exe .
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo OLLAMA BUILD FAILED
    exit /b %ERRORLEVEL%
)
echo OLLAMA BUILD OK

echo.
echo Done. Binaries:
dir ollama_genai_runner.exe ollama_new.exe

echo.
echo Backing up and deploying to Ollama install dir...
set OLLAMA_INSTALL=C:\Users\VAIBHA~1\AppData\Local\Programs\Ollama
set OV_BIN=c:\Users\VAIBHA~1\OneDrive\DOCUME~1\MASTER\OPENVI~1\OPENVI~1\modules\OLLAMA~1\OPENVI~1\OPENVI~1.0_X\runtime\bin\intel64\Release

copy /y %OLLAMA_INSTALL%\ollama.exe %OLLAMA_INSTALL%\ollama.exe.bak
copy /y ollama_new.exe %OLLAMA_INSTALL%\ollama.exe

echo Copying OpenVINO DLLs to Ollama install dir...
copy /y %OV_BIN%\openvino.dll %OLLAMA_INSTALL%\
copy /y %OV_BIN%\openvino_c.dll %OLLAMA_INSTALL%\
copy /y %OV_BIN%\openvino_genai.dll %OLLAMA_INSTALL%\
copy /y %OV_BIN%\openvino_genai_c.dll %OLLAMA_INSTALL%\
copy /y %OV_BIN%\openvino_tokenizers.dll %OLLAMA_INSTALL%\
copy /y %OV_BIN%\openvino_intel_cpu_plugin.dll %OLLAMA_INSTALL%\
copy /y %OV_BIN%\openvino_intel_gpu_plugin.dll %OLLAMA_INSTALL%\
copy /y %OV_BIN%\openvino_intel_npu_plugin.dll %OLLAMA_INSTALL%\
copy /y %OV_BIN%\openvino_ir_frontend.dll %OLLAMA_INSTALL%\
copy /y %OV_BIN%\openvino_auto_plugin.dll %OLLAMA_INSTALL%\
copy /y %OV_BIN%\openvino_auto_batch_plugin.dll %OLLAMA_INSTALL%\
copy /y %OV_BIN%\openvino_hetero_plugin.dll %OLLAMA_INSTALL%\
copy /y %OV_BIN%\icudt70.dll %OLLAMA_INSTALL%\
copy /y %OV_BIN%\icuuc70.dll %OLLAMA_INSTALL%\

echo.
echo Deployment complete. Run: ollama run deepseek-r1 "hello world"
