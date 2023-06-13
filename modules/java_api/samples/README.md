# Benchmark Application

This guide describes how to run the benchmark applications.

## How It Works

Upon start-up, the application reads command-line parameters and loads a network to the Inference Engine plugin, which is chosen depending on a specified device. The number of infer requests and execution approach depend on the mode defined with the `-api` command-line parameter.

## Build Benchmark Application
Set environment OpenVINO variables:
```bash
source <openvino_install>/setupvars.sh
``` 

Use Gradle to build `openvino-x-x-x.jar` with OpenVINO Java bindings and `benchmark_app.jar` file with Benchmark Application:
```bash
cd openvino_contrib/modules/java_api
gradle build -Pbuild_benchmark_app=true
```

## Running
Create an environment variable with path to directory with the `openvino-x-x-x.jar` file:
```bash
export OV_JAVA_DIR=/path/to/openvino_contrib/modules/java_api/build/libs
```

To get `benchmark_app` help use:
```bash
java -cp ".:${OV_JAVA_DIR}/openvino-x-x-x.jar:samples/benchmark_app/build/libs/benchmark_app.jar" Main --help
```

To run `benchmark_app` use:
```bash
java -cp ".:${OV_JAVA_DIR}/openvino-x-x-x.jar:samples/benchmark_app/build/libs/benchmark_app.jar" Main -m /path/to/model
```

## Running in Idea IntelliJ
- Import the project in IntelliJ IDEA. See [here](../README.md#import-to-intellij-idea) for instructions.
- In **Run/Debug Configurations** dropdown, click on **Edit Configurations**.
- Click on **Add New Configuration** and select **Gradle** from the dropdown menu.
- Give the configuration an appropriate name: "BenchmarkApp", and enter the following command in the **Tasks and arguments** input box.
    ```bash
    :samples:benchmark_app:run --args='-m <path-to-model>' -Pbuild_benchmark_app=true
    ```
- Under **Environment Variables**, select **Edit environment variables** and add the following environment variable: `INTEL_OPENVINO_DIR=<path-to-openvino_install>`
- Click on **OK** to save the configuration. 
- Select the saved configuration from the **Run/Debug Configurations**. Click on the **Run** button to run or the **Debug** button to run in debug mode. 

## Application Output

The application outputs the number of executed iterations, total duration of execution, latency, and throughput.

Below is fragment of application output for CPU device: 

```
[Step 10/11] Measuring performance (Start inference asyncronously, 4 inference requests using 4 streams for CPU, limits: 60000 ms duration)
[Step 11/11] Dumping statistics report
Count:      8904 iterations
Duration:   60045.87 ms
Latency:    27.03 ms
Throughput: 148.29 FPS
```

# Face Detection Java Samples

## How It Works

Upon start-up the sample application reads command line parameters and loads a network and an image to the Inference
Engine device. When inference is done, the application creates an output image/video.

Download the model from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) using the OpenVINO Model Downloader:
```
omz_downloader --name face-detection-adas-0001 --output_dir .
```

## Build
Build and run steps are similar to `benchmark_app`, but you need to add an environment variable with OpenCV installation or build path before building:
```bash
export OpenCV_DIR=/path/to/opencv/
```

Use Gradle to build `openvino-x-x-x.jar` with OpenVINO Java bindings and `sample_name.jar` files with samples:
```bash
cd openvino_contrib/modules/java_api
gradle build -Pbuild_java_samples=true
```

## Running
Add library path for opencv library and for openvino java library before running:

* For OpenCV installation path
```bash
export LD_LIBRARY_PATH=${OpenCV_DIR}/share/java/opencv4/:$LD_LIBRARY_PATH
```
To run sample use:
```bash
java -cp ".:${OV_JAVA_DIR}/openvino-x-x-x.jar:${OpenCV_DIR}/share/java/opencv4/*:samples/sample_name/build/libs/sample_name.jar" Main -m /path/to/model -i /path/to/image
```

* For OpenCV build path
```bash
export LD_LIBRARY_PATH=${OpenCV_DIR}/lib:$LD_LIBRARY_PATH
```
To run sample use:
```bash
java -cp ".:${OV_JAVA_DIR}/openvino-x-x-x.jar:${OpenCV_DIR}/bin/*:${OV_JAVA_DIR}/java_api.jar:samples/sample_name/build/libs/sample_name.jar" Main -m /path/to/model -i /path/to/image
```

## Running in Idea IntelliJ
- Import the project in IntelliJ IDEA. See [here](../README.md#import-to-intellij-idea) for instructions.
- In **Run/Debug Configurations** dropdown, click on **Edit Configurations**.
- Click on **Add New Configuration** and select **Gradle** from the dropdown menu.
- Give the configuration an appropriate name: "FaceDetectionSample", and enter the following command in the **Tasks and arguments** input box for `face_detection_java_sample`.
    ```bash
    :samples:face_detection_java_sample:run --args='-m <path-to-model> -i <path-to-image>' -Pbuild_java_samples=true
    ```
  For `face_detection_sample_async`, use the following command:
    ```bash
    :samples:face_detection_sample_async:run --args='-m <path-to-model> -i <path-to-image>' -Pbuild_java_samples=true
    ```
- Under **Environment Variables**, select **Edit environment variables** and add the following environment variables: 

  `INTEL_OPENVINO_DIR=<path-to-openvino_install>`
  
  `OpenCV_DIR=<path-to-opencv>`
- Under **System Environment Variables**, edit the **Path** variable and append the library path to opencv library
- Click on **OK** to save the configuration.
- Select the saved configuration from the **Run/Debug Configurations**. Click on the **Run** button to run or the **Debug** button to run in debug mode.

## Sample Output

### For ```face_detection_java_sample```
The application will show the image with detected objects enclosed in rectangles in new window. It outputs the confidence value and the coordinates of the rectangle to the standard output stream.

### For ```face_detection_sample_async```
The application will show the video with detected objects enclosed in rectangles in new window.
