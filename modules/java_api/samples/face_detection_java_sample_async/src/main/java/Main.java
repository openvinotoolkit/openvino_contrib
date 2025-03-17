import org.intel.openvino.*;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.*;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
import java.util.Vector;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/*
This is async face detection Java sample (for OpenVINO Java API 2.0).
Upon the start-up the sample application reads command line parameters and loads a network
and an image to the Inference Engine device. When inference is done, the application will show
the image with detected objects enclosed in rectangles in new window. It also outputs the
confidence value and the coordinates of the rectangle to the standard output stream.
To get the list of command line parameters run the application with `--help` parameter.
*/
public class Main {
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

    public static Tensor imageToTensor(Mat image) {
        int[] dimsArr = {1, image.rows(), image.cols(), 3};
        return new Tensor(ElementType.u8, dimsArr, image.dataAddr());
    }

    static void processInferRequests() {
        while (!startedRequestsIds.isEmpty()) {
            int requestId = startedRequestsIds.peek();
            InferRequest inferRequest = inferRequests.get(requestId);

            if (!inferRequest.wait_for(0)) return;

            Tensor outputTensor = inferRequest.get_output_tensor();
            float[] res = outputTensor.data();

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

        ArgumentParser parser = new ArgumentParser("This is async face detection sample");
        parser.addArgument("-i", "path to video");
        parser.addArgument("-m", "path to model .xml");
        parser.addArgument("-d", "device");
        parser.addArgument("-nireq", "number of infer requests");
        parser.parseArgs(args);

        String imgsPath = parser.get("-i", null);
        String xmlPath = parser.get("-m", null);
        String device = parser.get("-d", "CPU");
        int inferRequestsSize = parser.getInteger("-nireq", 2);

        if (imgsPath == null) {
            System.out.println("Error: Missed argument: -i");
            return;
        }
        if (xmlPath == null) {
            System.out.println("Error: Missed argument: -m");
            return;
        }

        int warmupNum = inferRequestsSize * 2;

        BlockingQueue<Mat> framesQueue = new LinkedBlockingQueue<Mat>();

        Runnable capture =
                new Runnable() {
                    @Override
                    public void run() {
                        Mat frame = new Mat();

                        VideoCapture cam = new VideoCapture();
                        try {
                            int idx = Integer.valueOf(imgsPath);
                            cam.open(idx);
                        } catch (NumberFormatException exception) {
                            cam.open(imgsPath);
                        }

                        while (cam.read(frame) && !Thread.interrupted()) {
                            framesCounter++;
                            framesQueue.add(frame.clone());
                        }
                        if (framesCounter == 0) {
                            System.err.println("ERROR: Can't get any video frame!");
                            System.exit(1);
                        }
                    }
                };
        Thread captureThread = new Thread(capture);

        Runnable infer =
                new Runnable() {
                    @Override
                    public void run() {
                        try {
                            Core core = new Core();
                            Model net = core.read_model(xmlPath);

                            PrePostProcessor p = new PrePostProcessor(net);
                            p.input()
                                    .tensor()
                                    .set_element_type(ElementType.u8)
                                    .set_layout(new Layout("NHWC"));
                            p.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR);
                            p.input().model().set_layout(new Layout("NCHW"));
                            p.build();

                            CompiledModel compiledModel = core.compile_model(net, device);

                            asyncInferIsFree = new Vector<Boolean>(inferRequestsSize);

                            for (int i = 0; i < inferRequestsSize; i++) {
                                inferRequests.add(compiledModel.create_infer_request());
                                asyncInferIsFree.add(true);
                            }

                            boolean isRunning = true;

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

                                    Tensor imgTensor = imageToTensor(frame);
                                    request.set_input_tensor(imgTensor);

                                    startedRequestsIds.add(i);
                                    request.start_async();
                                }
                            }
                            processInferRequests();
                        } catch (InterruptedException e) {
                            e.printStackTrace();

                            for (Thread t : Thread.getAllStackTraces().keySet())
                                if (t.getState() == Thread.State.RUNNABLE) t.interrupt();
                        }
                    }
                };
        Thread inferThread = new Thread(infer);

        captureThread.start();
        inferThread.start();

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

                    // Drawing only objects with >70% probability
                    if (confidence < CONFIDENCE_THRESHOLD) continue;

                    int xmin = (int) (detection[curProposal * 7 + 3] * img.cols());
                    int ymin = (int) (detection[curProposal * 7 + 4] * img.rows());
                    int xmax = (int) (detection[curProposal * 7 + 5] * img.cols());
                    int ymax = (int) (detection[curProposal * 7 + 6] * img.rows());

                    // Draw rectangle around detected object.
                    Point lt = new Point(xmin, ymin);
                    Point br = new Point(xmax, ymax);
                    Imgproc.rectangle(img, lt, br, color, 2);
                }

                if (resultCounter == warmupNum) {
                    tm.start();
                } else if (resultCounter > warmupNum) {
                    tm.stop();
                    double worksFps = ((double) (resultCounter - warmupNum)) / tm.getTimeSec();
                    double readFps = ((double) (framesCounter - warmupNum)) / tm.getTimeSec();
                    tm.start();

                    String label = "Reading fps: " + String.format("%.3f", readFps);
                    String label1 = "Inference fps: " + String.format("%.3f", worksFps);

                    Imgproc.putText(img, label, new Point(10, 50), 0, 0.7, color, 1);
                    Imgproc.putText(img, label1, new Point(10, 80), 0, 0.7, color, 1);
                }
                HighGui.imshow("Detection", img);

                if (HighGui.waitKey(1) != -1) {
                    inferThread.interrupt();
                    captureThread.interrupt();
                    break;
                }
            }

            HighGui.waitKey(1);
            HighGui.destroyAllWindows();

            captureThread.join();
            inferThread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
            for (Thread t : Thread.getAllStackTraces().keySet())
                if (t.getState() == Thread.State.RUNNABLE) t.interrupt();
        }
    }
}