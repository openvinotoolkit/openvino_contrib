# Samples

The OpenVINO samples are simple console applications that show how to utilize specific OpenVINO API capabilities within an application. The following samples are available
- [Face Detection Java samples](#face-detection-java-samples)
- [Face Detection Kotlin sample](./face_detection_kotlin_sample/README.md)
- [Hello Query Device](#hello-query-device-sample)

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
To run the sample, you need to specify a model. You can use [public](https://docs.openvino.ai/2022.3/omz_models_group_public.html#doxid-omz-models-group-public) or [Intel's](https://docs.openvino.ai/2022.3/omz_models_group_intel.html#doxid-omz-models-group-intel)
pretrained models from the Open Model Zoo. The models can be downloaded using the Model Downloader.

1. Create an environment variable with path to directory with the **openvino-x-x-x.jar** file:
    ```bash
    export OV_JAVA_DIR=/path/to/openvino_contrib/modules/java_api/build/libs
    ```

2. Install the **openvino-dev** Python package to use Open Model Zoo Tools:
    ```bash
    python -m pip install openvino-dev[caffe]
    ```

3. Download a pre-trained model using:
    ```bash
    omz_downloader --name googlenet-v1
    ```

4. If a model is not in the IR or ONNX format, it must be converted. You can do this using the model converter:
    ```bash
    omz_converter --name googlenet-v1
    ```

5. To get **benchmark_app** help use:
    ```bash
    java -cp ".:${OV_JAVA_DIR}/openvino-x-x-x.jar:samples/benchmark_app/build/libs/benchmark_app.jar" Main --help
    ```

6. To performing benchmarking using the **googlenet-v1** model on a **CPU**, use:
    ```bash
    java -cp ".:${OV_JAVA_DIR}/openvino-x-x-x.jar:samples/benchmark_app/build/libs/benchmark_app.jar" Main -m googlenet-v1.xml
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
Build and run steps are similar to **benchmark_app**, but you need to add an environment variable with OpenCV installation or build path:
```bash
export OpenCV_DIR=/path/to/opencv/
```

Use Gradle to build **openvino-x-x-x.jar** with OpenVINO Java bindings in `java_api/build/libs` and **sample_name.jar** files with samples in `java_api/samples/<sample_name>/build/libs`:
```bash
cd openvino_contrib/modules/java_api
gradle build -Pbuild_java_samples=true
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

## Running in Idea IntelliJ
- Import the project in IntelliJ IDEA. See [here](../README.md#import-to-intellij-idea) for instructions.
- In **Run/Debug Configurations** dropdown, click on **Edit Configurations**.
- Click on **Add New Configuration** and select **Gradle** from the dropdown menu.
- Give the configuration an appropriate name: "FaceDetectionSample", and enter the following command in the **Tasks and arguments** input box for **face_detection_java_sample**.
    ```bash
    :samples:face_detection_java_sample:run --args='-m <path-to-model> -i <path-to-image>' -Pbuild_java_samples=true
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


# Hello Query Device Sample

## How It Works

This sample demonstrates how to show OpenVINO Runtime devices and print their metrics and default configuration values using the Query Device API feature.

## Build

Use Gradle to build **openvino-x-x-x.jar** with OpenVINO Java bindings in `java_api/build/libs` and **hello_query_device.jar** in `java_api/samples/hello_query_device/build/libs`:
```bash
cd openvino_contrib/modules/java_api
gradle build -Pbuild_hello_query_device=true
```

## Running

To run the sample use:
```bash
java -cp ".:${OV_JAVA_DIR}/openvino-x-x-x.jar:samples/hello_query_device/build/libs/hello_query_device.jar" Main
```

## Running in Idea IntelliJ
- Import the project in IntelliJ IDEA. See [here](../README.md#import-to-intellij-idea) for instructions.
- In **Run/Debug Configurations** dropdown, click on **Edit Configurations**.
- Click on **Add New Configuration** and select **Gradle** from the dropdown menu.
- Give the configuration an appropriate name: "HelloQueryDeviceSample", and enter the following command in the **Tasks and arguments** input box.
    ```bash
    :samples:hello_query_device:run -Pbuild_hello_query_device=true
    ```
- Under **Environment Variables**, select **Edit environment variables** and add the following environment variables:

  `INTEL_OPENVINO_DIR=<path-to-openvino_install>`
- Click on **OK** to save the configuration.
- Select the saved configuration from the **Run/Debug Configurations**. Click on the **Run** button to run or the **Debug** button to run in debug mode.

## Sample Output

Below is a sample output for CPU device:

```
[INFO] Available devices:
[INFO] CPU:
[INFO]  SUPPORTED_PROPERTIES:
[INFO]          AVAILABLE_DEVICES:
[INFO]          RANGE_FOR_ASYNC_INFER_REQUESTS: 1 1 1
[INFO]          RANGE_FOR_STREAMS: 1 20
[INFO]          FULL_DEVICE_NAME: 12th Gen Intel(R) Core(TM) i7-12700H
[INFO]          OPTIMIZATION_CAPABILITIES: FP32 FP16 INT8 BIN EXPORT_IMPORT
[INFO]          CACHING_PROPERTIES: FULL_DEVICE_NAME
[INFO]          NUM_STREAMS: 1
[INFO]          AFFINITY: HYBRID_AWARE
[INFO]          INFERENCE_NUM_THREADS: 0
[INFO]          PERF_COUNT: NO
[INFO]          INFERENCE_PRECISION_HINT: f32
[INFO]          PERFORMANCE_HINT: LATENCY
[INFO]          EXECUTION_MODE_HINT: PERFORMANCE
[INFO]          PERFORMANCE_HINT_NUM_REQUESTS: 0
[INFO]          ENABLE_CPU_PINNING: YES
[INFO]          SCHEDULING_CORE_TYPE: ANY_CORE
[INFO]          ENABLE_HYPER_THREADING: YES
[INFO]          DEVICE_ID:
```