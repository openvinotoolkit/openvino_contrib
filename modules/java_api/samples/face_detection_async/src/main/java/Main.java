import org.openvino.*;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.*;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Vector;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public class Main {

    // API 2.0: Blob is now Tensor
    public static Tensor imageToTensor(Mat image) {
        int[] dimsArr = {1, image.height(), image.width(), image.channels()};
        // In API 2.0, we use ElementType and pass the memory address
        return new Tensor(ElementType.U8, dimsArr, image.dataAddr());
    }

    static void processInferRequests() {
        int size = 0;
        float[] res = null;

        while (!startedRequestsIds.isEmpty()) {
            int requestId = startedRequestsIds.peek();
            InferRequest inferRequest = inferRequests.get(requestId);

            // API 2.0: Wait() is now wait()
            inferRequest.wait();

            if (size == 0 && res == null) {
                // API 2.0: GetBlob is now get_tensor
                size = (int) inferRequest.get_tensor(outputName).get_size();
                res = new float[size];
            }

            // Get the data array from the tensor
            float[] tensorData = inferRequest.get_tensor(outputName).data();
            System.arraycopy(tensorData, 0, res, 0, size);
            
            detectionOutput.add(res);
            resultCounter++;
            asyncInferIsFree.setElementAt(true, requestId);
            startedRequestsIds.remove();
        }
    }

    public static void main(String[] args) {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load OpenCV library\n" + e);
            System.exit(1);
        }

        // Parse arguments (assuming ArgumentParser is available in common samples)
        String imgsPath = "0"; // Default camera
        String xmlPath = "face-detection-retail-0004.xml"; // Example model
        String device = "CPU";
        int inferRequestsSize = 2;

        int warmupNum = inferRequestsSize * 2;
        BlockingQueue<Mat> framesQueue = new LinkedBlockingQueue<Mat>();

        Runnable capture = new Runnable() {
            @Override
            public void run() {
                Mat frame = new Mat();
                VideoCapture cam = new VideoCapture();
                try {
                    cam.open(Integer.valueOf(imgsPath));
                } catch (NumberFormatException exception) {
                    cam.open(imgsPath);
                }

                while (cam.read(frame) && !Thread.interrupted()) {
                    framesCounter++;
                    framesQueue.add(frame.clone());
                }
            }
        };
        Thread captureThread = new Thread(capture);

        Runnable infer = new Runnable() {
            @Override
            public void run() {
                try {
                    // API 2.0: Core instead of IECore
                    org.openvino.Core core = new org.openvino.Core();
                    
                    // API 2.0: Model instead of CNNNetwork
                    org.openvino.Model model = core.read_model(xmlPath);
                    outputName = model.outputs().get(0).get_any_name();

                    // API 2.0: CompiledModel instead of ExecutableNetwork
                    org.openvino.CompiledModel compiledModel = core.compile_model(model, device);

                    asyncInferIsFree = new Vector<Boolean>(inferRequestsSize);

                    for (int i = 0; i < inferRequestsSize; i++) {
                        // API 2.0: create_infer_request()
                        inferRequests.add(compiledModel.create_infer_request());
                        asyncInferIsFree.add(true);
                    }

                    while (captureThread.isAlive() || !framesQueue.isEmpty()) {
                        if (Thread.interrupted()) break;

                        processInferRequests();
                        
                        for (int i = 0; i < inferRequestsSize; i++) {
                            if (!asyncInferIsFree.get(i)) continue;

                            Mat frame = framesQueue.poll(0, TimeUnit.SECONDS);
                            if (frame == null) break;

                            InferRequest request = inferRequests.get(i);
                            asyncInferIsFree.setElementAt(false, i);
                            processedFramesQueue.add(frame);

                            // API 2.0: SetBlob -> set_tensor
                            Tensor imgTensor = imageToTensor(frame);
                            request.set_tensor(model.inputs().get(0).get_any_name(), imgTensor);

                            startedRequestsIds.add(i);
                            
                            // API 2.0: StartAsync() -> start_async()
                            request.start_async();
                        }
                    }
                    processInferRequests();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        };
        Thread inferThread = new Thread(infer);

        captureThread.start();
        inferThread.start();

        // Standard OpenCV drawing logic
        TickMeter tm = new TickMeter();
        Scalar color = new Scalar(0, 255, 0);
        try {
            while (inferThread.isAlive() || !detectionOutput.isEmpty()) {
                float[] detection = detectionOutput.poll(waitingTime, TimeUnit.SECONDS);
                if (detection == null) continue;

                Mat img = processedFramesQueue.poll(waitingTime, TimeUnit.SECONDS);
                int maxProposalCount = detection.length / 7;

                for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
                    int imageId = (int) detection[curProposal * 7];
                    if (imageId < 0) break;
                    float confidence = detection[curProposal * 7 + 2];

                    if (confidence < CONFIDENCE_THRESHOLD) continue;

                    int xmin = (int) (detection[curProposal * 7 + 3] * img.cols());
                    int ymin = (int) (detection[curProposal * 7 + 4] * img.rows());
                    int xmax = (int) (detection[curProposal * 7 + 5] * img.cols());
                    int ymax = (int) (detection[curProposal * 7 + 6] * img.rows());

                    Imgproc.rectangle(img, new Point(xmin, ymin), new Point(xmax, ymax), color, 2);
                }

                HighGui.imshow("Detection API 2.0", img);
                if (HighGui.waitKey(1) != -1) {
                    inferThread.interrupt();
                    captureThread.interrupt();
                    break;
                }
            }
            HighGui.destroyAllWindows();
            captureThread.join();
            inferThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    static final float CONFIDENCE_THRESHOLD = 0.7f;
    static int waitingTime = 1;
    static BlockingQueue<Mat> processedFramesQueue = new LinkedBlockingQueue<Mat>();
    static BlockingQueue<float[]> detectionOutput = new LinkedBlockingQueue<float[]>();
    static String outputName;
    static Queue<Integer> startedRequestsIds = new LinkedList<Integer>();
    static Vector<InferRequest> inferRequests = new Vector<InferRequest>();
    static Vector<Boolean> asyncInferIsFree;
    static int framesCounter = 0;
    static int resultCounter = 0;
}