1. All steps as for java samples buld
2. To compile:
```bash
kotlinc -cp ".:${OpenCV_DIR}/bin/opencv-460.jar:${OV_JAVA_DIR}/java_api.jar"  hello.kt -include-runtime -d hello.jar
```
3. To run:
```bash
java -cp ".:${OpenCV_DIR}/bin/*:${OV_JAVA_DIR}/java_api.jar:hello.jar" Main
```
