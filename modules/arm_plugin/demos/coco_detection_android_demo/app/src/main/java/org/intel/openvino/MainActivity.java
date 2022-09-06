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
import org.intel.openvino.*;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Vector;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public class MainActivity extends CameraActivity implements CvCameraViewListener2 {
    private CameraBridgeViewBase mOpenCvCameraView;
    private String inputName;
    private String outputName;
    private String modelDir;
    private Mat currentFrame;

    public static final String OPENCV_LIBRARY_NAME = "opencv_java4";
    public static final String PLUGINS_XML = "plugins.xml";
    public static final String MODEL_XML = "ssdlite_mobilenet_v2.xml";
    public static final String MODEL_BIN = "ssdlite_mobilenet_v2.bin";
    public static final String DEVICE_NAME = "CPU";
    public static int waitingTime = 1; // Waiting for 1 second
    public static int inferRequestsSize = 4;
    public static int warmupNum = inferRequestsSize * 2;
    public TickMeter tm = null;

    public static BlockingQueue<Mat> processedFramesQueue = new LinkedBlockingQueue<Mat>();
    public static BlockingQueue<float[]> detectionOutput = new LinkedBlockingQueue<float[]>(); // Can't add null object
    public static BlockingQueue<Mat> framesQueue = new LinkedBlockingQueue<Mat>();

    public static Queue<Integer> startedRequestsIds = new LinkedList<Integer>();
    public static Vector<InferRequest> inferRequests = new Vector<InferRequest>();
    public static Vector<Boolean> asyncInferIsFree;

    public static int framesCounter = 0;
    public static int resultCounter = 0;

    public static final String[] COCO_CLASSES_80 = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    public static final String[] COCO_CLASSES_91 = {
            "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
            "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
            "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
            "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush", "hair brush"
    };

    private Blob imageToBlob(Mat image) {
        int[] dimsArr = {1, image.channels(), image.height(), image.width()};
        TensorDesc tDesc = new TensorDesc(Precision.U8, dimsArr, Layout.NHWC);

        return new Blob(tDesc, image.dataAddr());
    }

    private void processInferRequets(WaitMode wait) {
        int size = 0;
        float[] res = null;

        while (!startedRequestsIds.isEmpty()) {
            int requestId = startedRequestsIds.peek();
            InferRequest inferRequest = inferRequests.get(requestId);

            if (inferRequest.Wait(wait) != StatusCode.OK)
                return; // STATUS_ONLY => StatusCode OK

            if (size == 0 && res == null) {
                size = inferRequest.GetBlob(outputName).size();
                res = new float[size];
            }

            inferRequest.GetBlob(outputName).rmap().get(res);
            detectionOutput.add(res);

            resultCounter++;

            asyncInferIsFree.setElementAt(true, requestId);
            startedRequestsIds.remove();
        }
    }

    private void copyFiles() {
        String[] fileNames = {MODEL_BIN, MODEL_XML, PLUGINS_XML};
        for (String fileName: fileNames) {
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
        mOpenCvCameraView.enableFpsMeter();
        mOpenCvCameraView.setMaxFrameSize(640, 480);
        copyFiles();

        IECore core = new IECore(modelDir + "/" + PLUGINS_XML);
        CNNNetwork net = core.ReadNetwork(modelDir + "/" + MODEL_XML);
        System.out.println("load ok...");

        Map<String, InputInfo> inputsInfo = net.getInputsInfo();
        inputName = new ArrayList<String>(inputsInfo.keySet()).get(0);
        InputInfo inputInfo = inputsInfo.get(inputName);

        inputInfo.getPreProcess().setResizeAlgorithm(ResizeAlgorithm.RESIZE_BILINEAR);
        inputInfo.setPrecision(Precision.U8);

        outputName = new ArrayList<String>(net.getOutputsInfo().keySet()).get(0);

        ExecutableNetwork executableNetwork = core.LoadNetwork(net, DEVICE_NAME);

        asyncInferIsFree = new Vector<Boolean>(inferRequestsSize); // Add flags for the infer process

        // Init multi requests
        for (int i = 0; i < inferRequestsSize; i++) {
            inferRequests.add(executableNetwork.CreateInferRequest());
            asyncInferIsFree.add(true);
        }

        // System info
        String tag = "APPActivity";
        ActivityManager activityManager = (ActivityManager) getSystemService(ACTIVITY_SERVICE);
        ActivityManager.MemoryInfo info = new ActivityManager.MemoryInfo();
        activityManager.getMemoryInfo(info);

        Log.i(tag, "residue memory : " + (info.availMem >> 20) + "M");
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try{
            System.loadLibrary(OPENCV_LIBRARY_NAME);
            System.loadLibrary(IECore.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            Log.e("UnsatisfiedLinkError",
                    "Failed to load native OpenVINO libraries\n" + e.toString());
            System.exit(1);
        }
        modelDir = this.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS).getAbsolutePath();
        if(checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 0);
        } else {
            processNetwork();
        }

        tm = new TickMeter();

        // Infer Thread
        Runnable infer = () -> {
            try {
                // Wait for several seconds
                // Thread.sleep(5000);
                while (true) { // !framesQueue.isEmpty()!framesQueue.isEmpty()
                    if (Thread.interrupted()) break;

                    processInferRequets(WaitMode.STATUS_ONLY);
                    for (int i = 0; i < inferRequestsSize; i++) {
                        if (!asyncInferIsFree.get(i)) continue; // Skip for the result

                        if (framesQueue.size() != 0) {
                            System.out.println("INFER REQUEST :" + framesQueue.size());
                        }

                        Mat frame = framesQueue.poll(0, TimeUnit.SECONDS);

                        if (frame == null) break;

                        InferRequest request = inferRequests.get(i);

                        asyncInferIsFree.setElementAt(false, i); // Get Infer lock

                        Imgproc.resize(frame, frame, new Size(300, 300));
                        processedFramesQueue.add(frame);

                        Blob imgBlob = imageToBlob(frame);
                        request.SetBlob(inputName, imgBlob);

                        startedRequestsIds.add(i);
                        request.StartAsync();
                    }
                }
                processInferRequets(WaitMode.RESULT_READY); // No Use
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        };

        Thread inferThread = new Thread(infer, "Infer Thread");
        inferThread.start();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
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
        // Check resources
        Mat frame = inputFrame.rgba();
        // Memory leak
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
        currentFrame = frame.clone();
        frame.release();

        framesCounter++;
        framesQueue.add(currentFrame);

        try {
            currentFrame = processedFramesQueue.poll(waitingTime * 10L, TimeUnit.SECONDS);
            Imgproc.resize(currentFrame, currentFrame, new Size(640, 480));

            float[] detection = detectionOutput.poll(waitingTime, TimeUnit.SECONDS);
            if (detection == null) return currentFrame;
            int maxProposalCount = detection.length / 7;

            // Details for NMS
            List<Rect2d> rect2dList = new ArrayList<>();
            List<Float> confList = new ArrayList<>();
            List<Integer> objIndexList = new ArrayList<>();

            for (int i = 0; i < maxProposalCount; i ++) {
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
            Dnn.NMSBoxes(boxes, confs, 0.6F, 0.6F, indices);

            if (indices.empty()) {
                System.out.println("No boxes here");
                return currentFrame;
            }

            int[] idxes = indices.toArray();
            for (int idx : idxes) {
                Rect2d rect2d = rect2dList.get(idx);
                Integer obj = objIndexList.get(idx);
                Float conf = confList.get(idx);
                Imgproc.rectangle(currentFrame, new Point(rect2d.x, rect2d.y),
                        new Point((rect2d.x + rect2d.width), (rect2d.y + rect2d.height)),
                        new Scalar(0, 255, 0), 1);
                Imgproc.putText(currentFrame, COCO_CLASSES_91[obj] + " " + conf, new Point(rect2d.x, rect2d.y - 10),
                        Imgproc.FONT_HERSHEY_COMPLEX, 0.5, new Scalar(0, 255, 0), 1);
            }

            // Using TickMeter to calculate fps
            if (resultCounter == warmupNum) {
                tm.start();
            } else if (resultCounter > warmupNum) {
                tm.stop();
                // Fps for inference
                double worksFps = ((double) (resultCounter - warmupNum)) / tm.getTimeSec();
                tm.start();

                String inferFps = "Inference fps: " + String.format("%.3f", worksFps);

                Imgproc.putText(currentFrame, inferFps, new Point(10, 15), 0, 0.5, new Scalar(0, 255, 0), 1);
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        return currentFrame;
    }
}