初始化openvino genai环境：

set CGO_CPPFLAGS=-I%OpenVINOGenAI_DIR%\..\include
set CGO_CFLAGS=-I%OpenVINO_DIR%\..\include
set CGO_LDFLAGS=-L%OpenVINOGenAI_DIR%\..\lib\intel64\Release