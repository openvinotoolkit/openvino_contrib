package org.intel.openvino;

import androidx.annotation.NonNull;
import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.widget.Toast;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.intel.openvino.*;
// import org.intel.openvino.compatibility.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.sql.Time;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Map;

public class MainActivity extends CameraActivity implements CvCameraViewListener2 {
    private CameraBridgeViewBase mOpenCvCameraView;
    private InferRequest inferRequest;
    private String inputName;
    private String outputName;
    private InferRequest re_inferRequest;
    private String re_inputName;
    private String re_outputName;
    private String modelDir;
    public static final String OPENCV_LIBRARY_NAME = "opencv_java4";
    public static final String PLUGINS_XML = "plugins.xml";
    public static final String MODEL_XML = "vehicle-detection-0200.xml";
    public static final String MODEL_BIN = "vehicle-detection-0200.bin";
    public static final String RE_MODEL_XML = "vehicle-attributes-recognition-barrier-0039.xml";
    public static final String RE_MODEL_BIN = "vehicle-attributes-recognition-barrier-0039.bin";
    public static final String DEVICE_NAME = "CPU";
    public static final String TEST_IMAGE = "cars.png";
    public static final Integer RECOGNITION_INPUT_SIZE = 72;
    public static final String[] COLORS = {"White", "Gray", "Yellow", "Red", "Green", "Blue", "Black"};
    public static final String[] TYPES = {"Car", "Bus", "Truck", "Van"};

    private void copyFiles() {
        String[] fileNames = {MODEL_BIN, MODEL_XML, RE_MODEL_XML, RE_MODEL_BIN, PLUGINS_XML, TEST_IMAGE};
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
        // Initialize model
        copyFiles();
        IECore core = new IECore(modelDir + "/" + PLUGINS_XML);
        CNNNetwork net = core.ReadNetwork(modelDir + "/" + MODEL_XML);
        System.out.println("load ok...");

        CNNNetwork re_net = core.ReadNetwork(modelDir + "/" + RE_MODEL_XML);
        System.out.println("load ok...");

        Map<String, InputInfo> inputsInfo = net.getInputsInfo();
        inputName = new ArrayList<String>(inputsInfo.keySet()).get(0);
        InputInfo inputInfo = inputsInfo.get(inputName);
        inputInfo.getPreProcess().setResizeAlgorithm(ResizeAlgorithm.RESIZE_BILINEAR);
        inputInfo.setPrecision(Precision.U8);
        ExecutableNetwork executableNetwork = core.LoadNetwork(net, DEVICE_NAME);
        inferRequest = executableNetwork.CreateInferRequest();
        Map<String, Data> outputsInfo = net.getOutputsInfo();
        outputName = new ArrayList<>(outputsInfo.keySet()).get(0);

        Map<String, InputInfo> re_inputsInfo = re_net.getInputsInfo();
        re_inputName = new ArrayList<String>(re_inputsInfo.keySet()).get(0);
        InputInfo re_inputInfo = re_inputsInfo.get(re_inputName);
        re_inputInfo.getPreProcess().setResizeAlgorithm(ResizeAlgorithm.RESIZE_BILINEAR);
        re_inputInfo.setPrecision(Precision.U8);
        ExecutableNetwork re_executableNetwork = core.LoadNetwork(re_net, DEVICE_NAME);
        re_inferRequest = re_executableNetwork.CreateInferRequest();
        Map<String, Data> re_outputsInfo = re_net.getOutputsInfo();
        System.out.println("re_outputInfo : " + re_outputsInfo);
        re_outputName = new ArrayList<>(re_outputsInfo.keySet()).get(0);
        re_outputName2 = new ArrayList<>(re_outputsInfo.keySet()).get(1);
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try{
            System.loadLibrary(OPENCV_LIBRARY_NAME);
            System.out.println("-----------------------");
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
        long start = System.currentTimeMillis();

        Mat frame = inputFrame.rgba();
        Mat frameBGR = new Mat();
        Imgproc.cvtColor(frame, frameBGR, Imgproc.COLOR_RGBA2RGB);

        // Test image
        String image_path = modelDir + "/" + TEST_IMAGE;
        Mat imread = Imgcodecs.imread(image_path);
        Mat newImread = new Mat();
        Imgproc.resize(imread, frameBGR, new Size(frameBGR.width(), frameBGR.height()));
        Imgproc.cvtColor(frameBGR, frameBGR, Imgproc.COLOR_RGB2BGR);

        int[] dimsArr = {1, 3, imread.height(), imread.width()};
        TensorDesc tDesc = new TensorDesc(Precision.U8, dimsArr, Layout.NHWC);
        Blob imgBlob = new Blob(tDesc, imread.dataAddr());
        inferRequest.SetBlob(inputName, imgBlob);
        inferRequest.Infer();
        Blob outputBlob = inferRequest.GetBlob(outputName);
        float[] scores = new float[outputBlob.size()];
        outputBlob.rmap().get(scores);
        int numDetections = outputBlob.size() / 7;

        // outputBlob.size() : 1400
        System.out.println("Outputblob : " + outputBlob.size());
        System.out.println("Scores : " + scores);

        for (int i = 0; i < numDetections; i++) {
            float confidence = scores[i * 7 + 2];
            if (confidence > CONFIDENCE_THRESHOLD) {
                float xmin = scores[i * 7 + 3] * frameBGR.cols();
                float ymin = scores[i * 7 + 4] * frameBGR.rows();
                float xmax = scores[i * 7 + 5] * frameBGR.cols();
                float ymax = scores[i * 7 + 6] * frameBGR.rows();
                Imgproc.rectangle(frameBGR, new Point(xmin, ymin), new Point(xmax, ymax), new Scalar(0, 0, 255), 6);

                // crop image
                Mat car = new Mat(imread, new Rect((int)(scores[i * 7 + 3] * imread.cols()),
                        (int)(scores[i * 7 + 4] * imread.rows()),
                        (int)(scores[i * 7 + 5] * imread.cols() - scores[i * 7 + 3] * imread.cols()),
                        (int)(scores[i * 7 + 6] * imread.rows() - scores[i * 7 + 4] * imread.rows())));


                // Resize to recognition model input dimension.
                Imgproc.resize(car, car, new Size(RECOGNITION_INPUT_SIZE, RECOGNITION_INPUT_SIZE));
                int[] re_dimsArr = {1, 3, car.height(), car.width()};
                TensorDesc re_tDesc = new TensorDesc(Precision.U8, re_dimsArr, Layout.NHWC);
                Blob re_imgBlob = new Blob(re_tDesc, car.dataAddr());
                re_inferRequest.SetBlob(re_inputName, re_imgBlob);
                // re_inferRequest.Infer();
                re_inferRequest.StartAsync();
                // Test async
                try {
                    Thread.sleep(900);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                re_inferRequest.Wait(WaitMode.RESULT_READY);


                Blob re_outputBlob = re_inferRequest.GetBlob(re_outputName);
                float[] re_scores = new float[re_outputBlob.size()];
                re_outputBlob.rmap().get(re_scores);
                // Find max confidence for 7 colors.
                int flag = 0;
                for (int m = 0; m < re_scores.length - 1; m++) {
                    if (re_scores[flag] < re_scores[m + 1]) {
                        flag = m + 1;
                    }
                }

                // Find max type for 3 types
                Blob re_outputBlob2 = re_inferRequest.GetBlob(re_outputName2);
                float[] re_scores2 = new float[re_outputBlob2.size()];
                re_outputBlob2.rmap().get(re_scores2);
                // Find max type for 3 types
                int flag2 = 0;
                for (int m = 0; m < re_scores2.length - 1; m++) {
                    if (re_scores2[flag2] < re_scores2[m + 1]) {
                        flag2 = m + 1;
                    }
                }
                Imgproc.putText(frameBGR, String.valueOf(COLORS[flag] + " " +TYPES[flag2]), new Point(xmin, ymin),
                        Imgproc.FONT_HERSHEY_COMPLEX, 1.8, new Scalar(0, 255, 0), 6);
            }
        }

        long end = System.currentTimeMillis();

        Imgproc.putText(frameBGR, String.valueOf(1000 / (end - start)) + " FPS", new Point(10, 100),
                Imgproc.FONT_HERSHEY_COMPLEX, 1.8, new Scalar(0, 255, 0), 6);


        return frameBGR;
    }
}