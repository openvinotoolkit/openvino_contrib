// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import org.intel.openvino.*
import org.intel.openvino.Core
import org.opencv.core.*
import org.opencv.highgui.HighGui
import org.opencv.imgcodecs.*
import org.opencv.imgproc.Imgproc
import kotlin.system.exitProcess

/*
This is face detection Kotlin sample (for OpenVINO Java API 2.0).
Upon the start-up the sample application reads command line parameters and loads a network
and an image to the Inference Engine device. When inference is done, the application will show
the image with detected objects enclosed in rectangles in new window.It also outputs the
confidence value and the coordinates of the rectangle to the standard output stream.
To get the list of command line parameters run the application with `--help` paramether.
*/
object Main {

    @JvmStatic
    fun main(args: Array<String>) {

        val THRESHOLD = 0.7

        try {
            System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME)
        } catch (e: UnsatisfiedLinkError) {
            System.err.println("Failed to load OpenCV library\n$e")
            exitProcess(1)
        }
        Core.loadNativeLibs()

        val parser = ArgumentParser("This is face detection sample")
        parser.addArgument("-i", "path to image")
        parser.addArgument("-m", "path to model .xml")
        parser.parseArgs(args)

        val imgPath = parser.get("-i", null)
        val xmlPath = parser.get("-m", null)

        if (imgPath == null) {
            println("Error: Missed argument: -i")
            return
        }
        if (xmlPath == null) {
            println("Error: Missed argument: -m")
            return
        }

        val image = Imgcodecs.imread(imgPath)

        val core = Core()
        val net = core.read_model(xmlPath)

        /* The source image is also used at the end of the program to display the detection results,
        therefore the Mat object won't be destroyed by Garbage Collector while the network is
        running. */
        val dimsArr = intArrayOf(1, image.rows(), image.cols(), 3)
        val input_tensor = Tensor(ElementType.u8, dimsArr, image.dataAddr())

        val p = PrePostProcessor(net)
        p.input()
            .tensor()
            .set_element_type(ElementType.u8)
            .set_layout(Layout("NHWC"))
            .set_spatial_static_shape(image.rows(), image.cols())

        p.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)
        p.input().model().set_layout(Layout("NCHW"))
        p.build()

        val compiledModel = core.compile_model(net, "CPU")
        val inferRequest = compiledModel.create_infer_request()

        inferRequest.set_input_tensor(input_tensor)
        inferRequest.infer()

        val output_tensor = inferRequest.get_output_tensor()
        val detection = output_tensor.data()

        val dims = output_tensor.get_shape()
        val maxProposalCount = dims[2]

        for (curProposal in 0 until maxProposalCount) {
            val image_id = detection[curProposal * 7].toInt()
            if (image_id < 0) break

            val confidence = detection[curProposal * 7 + 2]

            // Drawing only objects with > 70% probability
            if (confidence < THRESHOLD) continue

            val label = detection[curProposal * 7 + 1].toInt()
            val xmin = ((detection[curProposal * 7 + 3] * image.cols())).toInt()
            val ymin = ((detection[curProposal * 7 + 4] * image.rows())).toInt()
            val xmax = ((detection[curProposal * 7 + 5] * image.cols())).toInt()
            val ymax = ((detection[curProposal * 7 + 6] * image.rows())).toInt() //as Int

            println("[$curProposal,$label] element, prob = $confidence    ($xmin,$ymin)-($xmax,$ymax) - WILL BE PRINTED!")

            // Draw rectangle around detected object.
            Imgproc.rectangle(
                image, Point(xmin.toDouble(), ymin.toDouble()), Point(xmax.toDouble(), ymax.toDouble()), Scalar(0.0, 255.0, 0.0)
            )
        }

        HighGui.namedWindow("Detection", HighGui.WINDOW_AUTOSIZE)
        HighGui.imshow("Detection", image)
        HighGui.waitKey(0)
        HighGui.destroyAllWindows()
    }
}
