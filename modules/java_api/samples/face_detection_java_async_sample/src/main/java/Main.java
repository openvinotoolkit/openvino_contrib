// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import org.intel.openvino.*;
import org.intel.openvino.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;

/*
This is an asynchronous face detection Java sample for OpenVINO Java API 2.0.

The sample keeps a small pool of InferRequest objects and overlaps video capture with
inference using start_async()/wait_async(). While one request is running on the device,
another request can be prepared with the next frame.

Command-line options:
  -m          path to model .xml (required)
  -i          path to video/image, or camera index such as 0 (required)
  -d          inference device, default CPU
  -nireq      number of async infer requests, default 2
  -t          detection confidence threshold, default 0.7
  -no_show    disable GUI visualization (useful for headless runs)
*/
public class Main {
    private static final class Detection {
        final int label;
        final float confidence;
        final int xmin;
        final int ymin;
        final int xmax;
        final int ymax;

        Detection(int label, float confidence, int xmin, int ymin, int xmax, int ymax) {
            this.label = label;
            this.confidence = confidence;
            this.xmin = xmin;
            this.ymin = ymin;
            this.xmax = xmax;
            this.ymax = ymax;
        }
    }

    private static final class FramePacket {
        final Mat frame;
        final long frameId;

        FramePacket(Mat frame, long frameId) {
            this.frame = frame;
            this.frameId = frameId;
        }
    }

    private static final class ResultPacket {
        final Mat frame;
        final long frameId;
        final List<Detection> detections;

        ResultPacket(Mat frame, long frameId, List<Detection> detections) {
            this.frame = frame;
            this.frameId = frameId;
            this.detections = detections;
        }
    }

    private static final class RequestSlot {
        final InferRequest request;
        Mat occupiedFrame;
        Tensor occupiedTensor;
        long occupiedFrameId = -1;
        boolean busy;

        RequestSlot(InferRequest request) {
            this.request = request;
        }
    }

    public static void main(String[] args) {
        try {
            System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load OpenCV library\n" + e);
            System.exit(1);
        }

        ArgumentParser parser = new ArgumentParser("Face detection async sample (OpenVINO Java API 2.0)");
        parser.addArgument("-i", "path to video/image or camera index (e.g. 0)");
        parser.addArgument("-m", "path to model .xml");
        parser.addArgument("-d", "device to infer on (default: CPU)");
        parser.addArgument("-nireq", "number of async infer requests (default: 2)");
        parser.addArgument("-t", "confidence threshold (default: 0.7)");
        parser.addArgument("-no_show", "disable GUI (true/false, default: false)");
        parser.parseArgs(args);

        String inputPath = parser.get("-i", null);
        String xmlPath = parser.get("-m", null);
        String device = parser.get("-d", "CPU");
        int numRequests = Math.max(1, parser.getInteger("-nireq", 2));
        double threshold = Double.parseDouble(parser.get("-t", "0.7"));
        boolean noShow = parser.getBoolean("-no_show", false);

        if (inputPath == null) {
            System.out.println("Error: Missed argument: -i");
            return;
        }
        if (xmlPath == null) {
            System.out.println("Error: Missed argument: -m");
            return;
        }

        VideoCapture capture = openCapture(inputPath);
        if (!capture.isOpened()) {
            System.err.println("Error: failed to open input: " + inputPath);
            return;
        }

        Mat firstFrame = new Mat();
        if (!capture.read(firstFrame) || firstFrame.empty()) {
            System.err.println("Error: failed to read the first frame from: " + inputPath);
            capture.release();
            return;
        }

        int frameHeight = firstFrame.rows();
        int frameWidth = firstFrame.cols();

        Core core = new Core();
        Model model = core.read_model(xmlPath);

        PrePostProcessor p = new PrePostProcessor(model);
        p.input()
                .tensor()
                .set_element_type(ElementType.u8)
                .set_layout(new Layout("NHWC"))
                .set_spatial_static_shape(frameHeight, frameWidth);
        p.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR);
        p.input().model().set_layout(new Layout("NCHW"));
        p.build();

        CompiledModel compiledModel = core.compile_model(model, device);
        RequestSlot[] slots = new RequestSlot[numRequests];
        for (int i = 0; i < numRequests; i++) {
            slots[i] = new RequestSlot(compiledModel.create_infer_request());
        }

        BlockingQueue<FramePacket> inputQueue = new ArrayBlockingQueue<>(numRequests * 2);
        BlockingQueue<ResultPacket> outputQueue = new ArrayBlockingQueue<>(numRequests * 2);
        AtomicBoolean captureDone = new AtomicBoolean(false);
        AtomicBoolean stop = new AtomicBoolean(false);

        // Put the already-read first frame into the pipeline.
        inputQueue.offer(new FramePacket(firstFrame, 0));

        Thread captureThread =
                new Thread(
                        () -> {
                            long frameId = 1;
                            try {
                                while (!stop.get()) {
                                    Mat frame = new Mat();
                                    if (!capture.read(frame) || frame.empty()) {
                                        frame.release();
                                        break;
                                    }
                                    if (frame.rows() != frameHeight || frame.cols() != frameWidth) {
                                        System.err.println(
                                                "Warning: skipping frame with unexpected size "
                                                        + frame.cols()
                                                        + "x"
                                                        + frame.rows());
                                        frame.release();
                                        continue;
                                    }
                                    inputQueue.put(new FramePacket(frame, frameId++));
                                }
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                            } finally {
                                captureDone.set(true);
                            }
                        },
                        "capture-thread");
        captureThread.start();

        Thread inferenceThread =
                new Thread(
                        () -> {
                            try {
                                boolean draining = false;
                                while (!stop.get()) {
                                    // Collect completed requests first so slots become free ASAP.
                                    for (RequestSlot slot : slots) {
                                        if (!slot.busy) {
                                            continue;
                                        }
                                        slot.request.wait_async();
                                        List<Detection> detections =
                                                parseDetections(
                                                        slot.request.get_output_tensor(),
                                                        slot.occupiedFrame.cols(),
                                                        slot.occupiedFrame.rows(),
                                                        threshold);
                                        outputQueue.put(
                                                new ResultPacket(
                                                        slot.occupiedFrame,
                                                        slot.occupiedFrameId,
                                                        detections));
                                        slot.occupiedFrame = null;
                                        slot.occupiedTensor = null;
                                        slot.occupiedFrameId = -1;
                                        slot.busy = false;
                                    }

                                    // Schedule new work onto free slots.
                                    boolean scheduled = false;
                                    for (RequestSlot slot : slots) {
                                        if (slot.busy) {
                                            continue;
                                        }
                                        FramePacket packet = inputQueue.poll();
                                        if (packet == null) {
                                            break;
                                        }
                                        int[] dimsArr = {
                                            1, packet.frame.rows(), packet.frame.cols(), 3
                                        };
                                        Tensor inputTensor =
                                                new Tensor(
                                                        ElementType.u8,
                                                        dimsArr,
                                                        packet.frame.dataAddr());
                                        slot.request.set_input_tensor(inputTensor);
                                        slot.request.start_async();
                                        slot.occupiedFrame = packet.frame;
                                        slot.occupiedTensor = inputTensor;
                                        slot.occupiedFrameId = packet.frameId;
                                        slot.busy = true;
                                        scheduled = true;
                                    }

                                    if (!scheduled) {
                                        if (captureDone.get() && inputQueue.isEmpty() && allIdle(slots)) {
                                            draining = true;
                                            break;
                                        }
                                        Thread.sleep(1);
                                    }
                                }

                                // Drain any in-flight requests on shutdown.
                                if (draining || stop.get()) {
                                    for (RequestSlot slot : slots) {
                                        if (!slot.busy) {
                                            continue;
                                        }
                                        slot.request.wait_async();
                                        List<Detection> detections =
                                                parseDetections(
                                                        slot.request.get_output_tensor(),
                                                        slot.occupiedFrame.cols(),
                                                        slot.occupiedFrame.rows(),
                                                        threshold);
                                        outputQueue.put(
                                                new ResultPacket(
                                                        slot.occupiedFrame,
                                                        slot.occupiedFrameId,
                                                        detections));
                                        slot.occupiedFrame = null;
                                        slot.occupiedTensor = null;
                                        slot.occupiedFrameId = -1;
                                        slot.busy = false;
                                    }
                                }
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                            }
                        },
                        "inference-thread");
        inferenceThread.start();

        long processed = 0;
        long detectedObjects = 0;
        try {
            while (true) {
                ResultPacket result = outputQueue.poll();
                if (result == null) {
                    if (!inferenceThread.isAlive() && outputQueue.isEmpty()) {
                        break;
                    }
                    Thread.sleep(1);
                    continue;
                }

                processed++;
                detectedObjects += result.detections.size();
                for (Detection det : result.detections) {
                    System.out.println(
                            "[frame "
                                    + result.frameId
                                    + ", det "
                                    + det.label
                                    + "] prob = "
                                    + det.confidence
                                    + "    ("
                                    + det.xmin
                                    + ","
                                    + det.ymin
                                    + ")-("
                                    + det.xmax
                                    + ","
                                    + det.ymax
                                    + ")");
                    Imgproc.rectangle(
                            result.frame,
                            new Point(det.xmin, det.ymin),
                            new Point(det.xmax, det.ymax),
                            new Scalar(0, 255, 0),
                            2);
                }

                if (!noShow) {
                    HighGui.imshow("Detection Async", result.frame);
                    int key = HighGui.waitKey(1);
                    if (key == 27 || key == 'q' || key == 'Q') {
                        stop.set(true);
                    }
                }

                result.frame.release();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            stop.set(true);
            try {
                captureThread.join(2000);
                inferenceThread.join(5000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            capture.release();
            if (!noShow) {
                HighGui.destroyAllWindows();
            }
            for (RequestSlot slot : slots) {
                slot.request.release();
            }
        }

        System.out.println(
                "Async sample finished. processed_frames="
                        + processed
                        + " detections="
                        + detectedObjects
                        + " device="
                        + device
                        + " nireq="
                        + numRequests);
    }

    private static boolean allIdle(RequestSlot[] slots) {
        for (RequestSlot slot : slots) {
            if (slot.busy) {
                return false;
            }
        }
        return true;
    }

    private static VideoCapture openCapture(String inputPath) {
        try {
            int cameraIndex = Integer.parseInt(inputPath);
            VideoCapture capture = new VideoCapture(cameraIndex);
            capture.set(Videoio.CAP_PROP_FRAME_WIDTH, 1280);
            capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, 720);
            return capture;
        } catch (NumberFormatException ignored) {
            return new VideoCapture(inputPath);
        }
    }

    private static List<Detection> parseDetections(
            Tensor outputTensor, int imageWidth, int imageHeight, double threshold) {
        List<Detection> detections = new ArrayList<>();
        float[] detection = outputTensor.data();
        int[] dims = outputTensor.get_shape();
        int maxProposalCount = dims[2];

        for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
            int imageId = (int) detection[curProposal * 7];
            if (imageId < 0) {
                break;
            }

            float confidence = detection[curProposal * 7 + 2];
            if (confidence < threshold) {
                continue;
            }

            int label = (int) detection[curProposal * 7 + 1];
            int xmin = (int) (detection[curProposal * 7 + 3] * imageWidth);
            int ymin = (int) (detection[curProposal * 7 + 4] * imageHeight);
            int xmax = (int) (detection[curProposal * 7 + 5] * imageWidth);
            int ymax = (int) (detection[curProposal * 7 + 6] * imageHeight);
            detections.add(new Detection(label, confidence, xmin, ymin, xmax, ymax));
        }
        return detections;
    }
}
