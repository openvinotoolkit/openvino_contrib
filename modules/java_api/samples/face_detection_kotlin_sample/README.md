# Face Detection Kotlin Samples

## Build
1. Set OpenVINO environment variables 
    ```bash
    source <openvino_install>/setupvars.sh 
    ```
2. Add the following environment variable with the OpenCV installation or build path:
    ```bash
    export OpenCV_DIR=/path/to/opencv/ 
    ```
3. Use Gradle to build `openvino-x-x-x.jar` with OpenVINO Java bindings and `face_detection_kotlin_sample.jar` file with samples:
    ```bash
    cd openvino_contrib/modules/java_api
    gradle build -Pbuild_kotlin_samples=true
    ```

## Running
To run these samples, you need to specify a model and image. To perform inference of **image.jpg** using **face-detection-adas-0001** model:

* For OpenCV installation path
  ```bash
  export LD_LIBRARY_PATH=${OpenCV_DIR}/share/java/opencv4/:$LD_LIBRARY_PATH
  ```

  To run sample use:
  ```bash
  java -cp ".:${OV_JAVA_DIR}/openvino-x-x-x.jar:${OpenCV_DIR}/share/java/opencv4/*:samples/<sample_name>/build/libs/<sample_name>.jar" Main -m face-detection-adas-0001.xml -i image.jpg
  ```

* For OpenCV build path
  ```bash
  export LD_LIBRARY_PATH=${OpenCV_DIR}/lib:$LD_LIBRARY_PATH
  ```
  To run sample use:
  ```bash
  java -cp ".:${OV_JAVA_DIR}/openvino-x-x-x.jar:${OpenCV_DIR}/bin/*:samples/<sample_name>/build/libs/<sample_name>.jar" Main -m face-detection-adas-0001.xml -i image.jpg
  ```

## Running in IntelliJ IDEA

- Import the project in IntelliJ IDEA. See [here](../README.md#import-to-intellij-idea) for instructions.
- In **Run/Debug Configurations** dropdown, click on **Edit Configurations**.
- Click on **Add New Configuration** and select **Gradle** from the dropdown menu.
- Give the configuration an appropriate name: "FaceDetectionKotlinSample", and enter the following command in the **Tasks and arguments** input box.
    ```bash
    :samples:face_detection_kotlin_sample:run --args='-m <path-to-model> -i <path-to-image>' -Pbuild_kotlin_samples=true
    ```
- Under **Environment Variables**, select **Edit environment variables** and add the following environment variables:

  `INTEL_OPENVINO_DIR=<path-to-openvino_install>`

  `OpenCV_DIR=<path-to-opencv>`
- Under **System Environment Variables**, edit the **Path** variable and append the library path to opencv library
- Click on **OK** to save the configuration.
- Select the saved configuration from the **Run/Debug Configurations**. Click on the **Run** button to run or the **Debug** button to run in debug mode.