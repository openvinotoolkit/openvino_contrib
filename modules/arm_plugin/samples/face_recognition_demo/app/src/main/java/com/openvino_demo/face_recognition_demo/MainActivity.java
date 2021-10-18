package com.openvino_demo.face_recognition_demo;

import androidx.annotation.NonNull;
import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.widget.Toast;

import org.intel.openvino.*;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
public class MainActivity extends CameraActivity implements CvCameraViewListener2 {
    private void copyFiles() {
        String[] fileNames = {MODEL_BIN, MODEL_XML, PLUGINS_XML};
        for (String fileName : fileNames) {
            String outputFilePath = modelDir + "/" + fileName;
            File outputFile = new File(outputFilePath);
            if (!outputFile.exists()) {
                try {
                    InputStream inputStream = getApplicationContext().getAssets().open(fileName);
                    OutputStream outputStream = new FileOutputStream(outputFilePath);
                    byte[] buffer = new byte[5120];
                    int length = inputStream.read(buffer);
                    while (length > 0) {
                        outputStream.write(buffer, 0, length);
                        length = inputStream.read(buffer);
                    }
                    outputStream.flush();
                    outputStream.close();
                    inputStream.close();
                } catch (Exception e) {
                    Log.e("CopyError", "Copying model has failed.");
                    System.exit(1);
                }
            }
        }
    }
    private void processNetwork() {
        // Set up camera listener.
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraPermissionGranted();
        // Initialize model
        copyFiles();
        IECore core = new IECore(modelDir + "/" + PLUGINS_XML);
        CNNNetwork net = core.ReadNetwork(modelDir + "/" + MODEL_XML);
        Map<String, InputInfo> inputsInfo = net.getInputsInfo();
        inputName = new ArrayList<String>(inputsInfo.keySet()).get(0);
        InputInfo inputInfo = inputsInfo.get(inputName);
        inputInfo.getPreProcess().setResizeAlgorithm(ResizeAlgorithm.RESIZE_BILINEAR);
        inputInfo.setPrecision(Precision.U8);
        ExecutableNetwork executableNetwork = core.LoadNetwork(net, DEVICE_NAME);
        inferRequest = executableNetwork.CreateInferRequest();
        Map<String, Data> outputsInfo = net.getOutputsInfo();
        outputName = new ArrayList<>(outputsInfo.keySet()).get(0);
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try {
            System.loadLibrary(OPENCV_LIBRARY_NAME);
            System.loadLibrary(IECore.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            Log.e(
                    "UnsatisfiedLinkError",
                    "Failed to load native OpenVINO libraries\n" + e.toString());
            System.exit(1);
        }
        modelDir = this.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS).getAbsolutePath();
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
           requestPermissions(new String[]{Manifest.permission.CAMERA}, 0);
        } else {
            processNetwork();
        }
    }
    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults.length > 0 && grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            Log.e("PermissionError", "The application can't work without camera permissions");
            System.exit(1);
        }
        processNetwork();
    }
    @Override
    public void onResume() {
        super.onResume();
        mOpenCvCameraView.enableView();
    }
    @Override
    public void onCameraViewStarted(int width, int height) {}
    @Override
    public void onCameraViewStopped() {}
    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat frame = inputFrame.rgba();
        Mat frameBGR = new Mat();
        Imgproc.cvtColor(frame, frameBGR, Imgproc.COLOR_RGBA2RGB);
        int[] dimsArr = {1, frameBGR.channels(), frameBGR.height(), frameBGR.width()};
        TensorDesc tDesc = new TensorDesc(Precision.U8, dimsArr, Layout.NHWC);
        Blob imgBlob = new Blob(tDesc, frameBGR.dataAddr());
        inferRequest.SetBlob(inputName, imgBlob);
        inferRequest.Infer();
        Blob outputBlob = inferRequest.GetBlob(outputName);
        float[] scores = new float[outputBlob.size()];
        outputBlob.rmap().get(scores);
        int numDetections = outputBlob.size() / 7;
        int confidentDetections = 0;
        for (int i = 0; i < numDetections; i++) {
            float confidence = scores[i * 7 + 2];
            if (confidence > CONFIDENCE_THRESHOLD) {
                float xmin = scores[i * 7 + 3] * frameBGR.cols();
                float ymin = scores[i * 7 + 4] * frameBGR.rows();
                float xmax = scores[i * 7 + 5] * frameBGR.cols();
                float ymax = scores[i * 7 + 6] * frameBGR.rows();
                Imgproc.rectangle(
                        frame,
                        new Point(xmin, ymin),
                        new Point(xmax, ymax),
                        new Scalar(0, 0, 255),
                        6);
                confidentDetections++;
            }
        }
        Imgproc.putText(
                frame,
                String.valueOf(confidentDetections),
                new Point(10, 40),
                Imgproc.FONT_HERSHEY_COMPLEX,
                1.8,
                new Scalar(0, 255, 0),
                6);
        return frame;
    }
    private CameraBridgeViewBase mOpenCvCameraView;
    private InferRequest inferRequest;
    private String inputName;
    private String outputName;
    private String modelDir;
    public static final double CONFIDENCE_THRESHOLD = 0.5;
    public static final String OPENCV_LIBRARY_NAME = "opencv_java4";
    public static final String PLUGINS_XML = "plugins.xml";
    public static final String MODEL_XML = "face-detection-adas-0001.xml";
    public static final String MODEL_BIN = "face-detection-adas-0001.bin";
    public static final String DEVICE_NAME = "CPU";
}
