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

Upon the start-up the sample application reads command line parameters and loads a network and an image to the Inference
Engine device. When inference is done, the application creates an output image/video.

Download model from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) using the following command:
```
python downloader.py --name face-detection-adas-0001 --output_dir .
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

## Sample Output

### For ```face_detection_java_sample```
The application will show the image with detected objects enclosed in rectangles in new window. It outputs the confidence value and the coordinates of the rectangle to the standard output stream.

### For ```face_detection_sample_async```
The application will show the video with detected objects enclosed in rectangles in new window.
