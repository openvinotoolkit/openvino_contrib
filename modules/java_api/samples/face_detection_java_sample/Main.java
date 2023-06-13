// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import org.intel.openvino.*;
import org.intel.openvino.Core;
import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.*;
import org.opencv.imgproc.Imgproc;

/*
This is face detection Java sample (for OpenVINO Java API 2.0).
Upon the start-up the sample application reads command line parameters and loads a network
and an image to the Inference Engine device. When inference is done, the application will show
the image with detected objects enclosed in rectangles in new window.It also outputs the
confidence value and the coordinates of the rectangle to the standard output stream.
To get the list of command line parameters run the application with `--help` paramether.
*/
public class Main {
    public static void main(String[] args) {
        final double THRESHOLD = 0.7;
        try {
            System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load OpenCV library\n" + e);
            System.exit(1);
        }

        ArgumentParser parser = new ArgumentParser("This is face detection sample");
        parser.addArgument("-i", "path to image");
        parser.addArgument("-m", "path to model .xml");
        parser.parseArgs(args);

        String imgPath = parser.get("-i", null);
        String xmlPath = parser.get("-m", null);

        if (imgPath == null) {
            System.out.println("Error: Missed argument: -i");
            return;
        }
        if (xmlPath == null) {
            System.out.println("Error: Missed argument: -m");
            return;
        }

        Mat image = Imgcodecs.imread(imgPath);

        Core core = new Core();
        Model net = core.read_model(xmlPath);

        /* The source image is also used at the end of the program to display the detection results,
        therefore the Mat object won't be destroyed by Garbage Collector while the network is
        running. */
        int[] dimsArr = {1, image.rows(), image.cols(), 3};
        Tensor input_tensor = new Tensor(ElementType.u8, dimsArr, image.dataAddr());

        PrePostProcessor p = new PrePostProcessor(net);
        p.input()
                .tensor()
                .set_element_type(ElementType.u8)
                .set_layout(new Layout("NHWC"))
                .set_spatial_static_shape(image.rows(), image.cols());

        p.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR);
        p.input().model().set_layout(new Layout("NCHW"));
        p.build();

        CompiledModel compiledModel = core.compile_model(net, "CPU");
        InferRequest inferRequest = compiledModel.create_infer_request();

        inferRequest.set_input_tensor(input_tensor);
        inferRequest.infer();

        Tensor output_tensor = inferRequest.get_output_tensor();
        float detection[] = output_tensor.data();

        int dims[] = output_tensor.get_shape();
        int maxProposalCount = dims[2];

        for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
            int image_id = (int) detection[curProposal * 7];
            if (image_id < 0) break;

            float confidence = detection[curProposal * 7 + 2];

            // Drawing only objects with >70% probability
            if (confidence < THRESHOLD) continue;

            int label = (int) (detection[curProposal * 7 + 1]);
            int xmin = (int) (detection[curProposal * 7 + 3] * image.cols());
            int ymin = (int) (detection[curProposal * 7 + 4] * image.rows());
            int xmax = (int) (detection[curProposal * 7 + 5] * image.cols());
            int ymax = (int) (detection[curProposal * 7 + 6] * image.rows());

            String result = "[" + curProposal + "," + label + "] element, prob = " + confidence;
            result += "    (" + xmin + "," + ymin + ")-(" + xmax + "," + ymax + ")";

            System.out.print(result);
            System.out.println(" - WILL BE PRINTED!");

            // Draw rectangle around detected object.
            Imgproc.rectangle(
                    image, new Point(xmin, ymin), new Point(xmax, ymax), new Scalar(0, 255, 0));
        }

        HighGui.namedWindow("Detection", HighGui.WINDOW_AUTOSIZE);
        HighGui.imshow("Detection", image);
        HighGui.waitKey(0);
        HighGui.destroyAllWindows();
    }
}
