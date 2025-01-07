package org.intel.openvino;

import android.Manifest;
import android.app.ActivityManager;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TickMeter;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * This is the object detection demo for ARM CPUs Android (for OpenVINO Java API 2.0).
 *
 * <p>The demo loads a network (including SSD, Pelee, EfficientDet) and read image from camera to
 * Inference Engine device. The screen will show the inference result and speed in frame.
 */
public class MainActivity extends CameraActivity implements CvCameraViewListener2 {
    private CameraBridgeViewBase mOpenCvCameraView;
    private InferRequest inferRequest;
    private String modelDir;
    public TickMeter tm;
    public Scalar[] randomColor;

    private static final String APPTAG = MainActivity.class.getName();
    public static final float CONFIDENCE_THRESHOLD = 0.6F;
    public static final float NMS_THRESHOLD = 0.6F;
    public static final String OPENCV_LIBRARY_NAME = "opencv_java4";
    public static final String MODEL_XML = "ssd_mobilenet_v1_coco.xml";
    public static final String COCO_LABELS = "labels.txt";
    public static final String MODEL_BIN = "ssd_mobilenet_v1_coco.bin";
    public static final String DEVICE_NAME = "CPU";
    public static String[] COCO_CLASSES_91;

    private void copyFiles() {
        String[] fileNames = {MODEL_BIN, MODEL_XML, COCO_LABELS};
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

    private void createRandomColor(int length) {
        Random r = new Random(999);
        randomColor = new Scalar[length];

        for (int i = 0; i < length; i++) {
            randomColor[i] =
                    new Scalar(r.nextDouble() * 255, r.nextDouble() * 255, r.nextDouble() * 255);
        }
    }

    private void processNetwork() {
        // Set up camera listener
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraPermissionGranted();
        mOpenCvCameraView.enableFpsMeter();
        mOpenCvCameraView.setMaxFrameSize(320, 240);

        // Initialize model
        copyFiles();

        List<String> listOfStrings = new ArrayList<String>();
        try {
            BufferedReader bf = null;
            bf = new BufferedReader(new FileReader(modelDir + "/" + COCO_LABELS));
            String line = bf.readLine();
            while (line != null) {
                listOfStrings.add(line);
                line = bf.readLine();
            }
            bf.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        COCO_CLASSES_91 = listOfStrings.toArray(new String[0]);

        Core core = new Core();
        Model net = core.read_model(modelDir + "/" + MODEL_XML);
        Log.i(APPTAG, "load ok...");

        // Set config of the network
        PrePostProcessor p = new PrePostProcessor(net);
        p.input().tensor().set_element_type(ElementType.u8).set_layout(new Layout("NCHW"));

        p.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR);
        p.input().model().set_layout(new Layout("NCHW"));
        p.build();

        CompiledModel compiledModel = core.compile_model(net, DEVICE_NAME);
        inferRequest = compiledModel.create_infer_request();

        // System info
        ActivityManager activityManager = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo info = new ActivityManager.MemoryInfo();
        activityManager.getMemoryInfo(info);
        Log.i(APPTAG, "residue memory : " + (info.availMem >> 20) + "M");

        tm = new TickMeter();
        createRandomColor(COCO_CLASSES_91.length);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try {
            System.loadLibrary(OPENCV_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            Log.e(
                    "UnsatisfiedLinkError",
                    "Failed to load native OpenCV libraries\n" + e.toString());
            System.exit(1);
        }
        modelDir = this.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS).getAbsolutePath();
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[] {Manifest.permission.CAMERA}, 0);
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
        // Use TickMeter to calculate fps
        tm.start();
        Mat frame = inputFrame.rgba();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
        Mat currentFrame = frame.clone();
        frame.release();

        // Preprocess the frame
        Imgproc.resize(currentFrame, currentFrame, new Size(300, 300));
        int[] dimsArr = {1, currentFrame.rows(), currentFrame.cols(), 3};
        Tensor input_tensor = new Tensor(ElementType.u8, dimsArr, currentFrame.dataAddr());

        // Set input data
        inferRequest.set_input_tensor(input_tensor);
        inferRequest.infer();

        // Get output data from model
        Tensor output_tensor = inferRequest.get_output_tensor();
        Imgproc.resize(currentFrame, currentFrame, new Size(320, 240));

        float[] detection = output_tensor.data();
        int maxProposalCount = detection.length / 7;

        // Construct the input to the NMS algorithm
        List<Rect2d> rect2dList = new ArrayList<>();
        List<Float> confList = new ArrayList<>();
        List<Integer> objIndexList = new ArrayList<>();

        for (int i = 0; i < maxProposalCount; i++) {
            float label = detection[i * 7 + 1];
            float conf = detection[i * 7 + 2];
            float xMin = detection[i * 7 + 3] * currentFrame.cols();
            float yMin = detection[i * 7 + 4] * currentFrame.rows();
            float xMax = detection[i * 7 + 5] * currentFrame.cols();
            float yMax = detection[i * 7 + 6] * currentFrame.rows();

            confList.add(conf);
            objIndexList.add((int) label);
            rect2dList.add(new Rect2d(xMin, yMin, (xMax - xMin), (yMax - yMin)));
        }

        MatOfInt indices = new MatOfInt();
        MatOfRect2d boxes = new MatOfRect2d(rect2dList.toArray(new Rect2d[0]));
        float[] confArr = new float[confList.size()];
        for (int i = 0; i < confList.size(); i++) {
            confArr[i] = confList.get(i);
        }
        MatOfFloat confs = new MatOfFloat(confArr);
        Dnn.NMSBoxes(boxes, confs, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

        if (indices.empty()) {
            tm.stop();
            // Fps for inference
            double worksFps = tm.getAvgTimeSec();
            String inferFps = "Inference fps: " + String.format("%.3f", worksFps);
            Imgproc.putText(
                    currentFrame, inferFps, new Point(10, 15), 0, 0.5, new Scalar(0, 255, 0), 1);

            Log.i(APPTAG, "No boxes here!");
            return currentFrame;
        }

        int[] idxes = indices.toArray();
        for (int idx : idxes) {
            Rect2d rect2d = rect2dList.get(idx);
            Integer obj = objIndexList.get(idx);
            Float conf = confList.get(idx);
            Imgproc.rectangle(
                    currentFrame,
                    new Point(rect2d.x, rect2d.y),
                    new Point((rect2d.x + rect2d.width), (rect2d.y + rect2d.height)),
                    randomColor[obj],
                    1);
            Imgproc.putText(
                    currentFrame,
                    COCO_CLASSES_91[obj] + " " + conf,
                    new Point(rect2d.x, rect2d.y - 10),
                    Imgproc.FONT_HERSHEY_COMPLEX,
                    0.5,
                    randomColor[obj],
                    1);
        }

        tm.stop();
        // Fps for inference
        double avgTime = tm.getAvgTimeSec() * 1000;
        double worksFps = tm.getFPS();
        String inferAvgTime = "Inference average time: " + String.format("%.3f", avgTime);
        String inferFps = "Inference fps: " + String.format("%.3f", worksFps);
        Imgproc.putText(
                currentFrame, inferAvgTime, new Point(10, 15), 0, 0.3, new Scalar(0, 255, 0), 1);
        Imgproc.putText(
                currentFrame, inferFps, new Point(10, 25), 0, 0.3, new Scalar(0, 255, 0), 1);

        return currentFrame;
    }
}
