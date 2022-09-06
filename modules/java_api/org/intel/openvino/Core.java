// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.file.Files;
import java.util.logging.Logger;

/**
 * This class represents an OpenVINO runtime Core entity.
 * <p>
 * User applications can create several Core class instances, but in this case the underlying plugins
 * are created multiple times and not shared between several Core instances. The recommended way is to have
 * a single Core instance per application.
 */
public class Core extends Wrapper {
    public static final String NATIVE_LIBRARY_NAME = "inference_engine_java_api";
    private static final Logger logger = Logger.getLogger(Core.class.getName());

    public Core() {
        super(GetCore());
    }

    private static String getLibraryName(String name, String linux_ver) {
        String osName = System.getProperty("os.name").toLowerCase();
        if (osName.contains("win")) {
            return name + ".dll";
        } else if (osName.contains("mac")) {
            return "lib" + name + ".dylib";
        } else {
            name = "lib" + name + ".so";
            if (linux_ver != null) {
                name += "." + linux_ver;
            }
        }
        return name;
    }

    // Use this method to initialize native libraries and other files like
    // plugins.xml and *.mvcmd in case of the JAR package os OpenVINO.
    public static void loadNativeLibs() {
        // A set of required libraries which are listed in dependency order.
        final String[] nativeLibs = {"tbb", "tbbmalloc", "openvino", "inference_engine_java_api"};

        InputStream resources_list = null;
        try {
            // Get a list of all native resources (libraries, plugins and other files).
            resources_list = Core.class.getClassLoader().getResourceAsStream("resources_list.txt");
            BufferedReader r = new BufferedReader(new InputStreamReader(resources_list));

            // Create a temporal folder to unpack native files.
            File tmpDir = Files.createTempDirectory("openvino-native").toFile();
            tmpDir.deleteOnExit();

            String file;
            while ((file = r.readLine()) != null) {
                // Check that file name is valid
                if (!file.chars()
                        .allMatch(
                                c ->
                                        Character.isLetterOrDigit(c)
                                                || c == '.'
                                                || c == '_'
                                                || c == '-')) {
                    throw new IOException("Invalid file path: " + file);
                }

                URL url = Core.class.getClassLoader().getResource(file);
                if (url == null) {
                    logger.warning("Resource not found: " + file);
                    continue;
                }
                File nativeLibTmpFile = new File(tmpDir, file);
                nativeLibTmpFile.deleteOnExit();
                try (InputStream in = url.openStream()) {
                    Files.copy(in, nativeLibTmpFile.toPath());
                }
            }

            // Load native libraries.
            for (String lib : nativeLibs) {
                // On Linux, TBB and GNA libraries has .so.2 soname
                String version = lib.startsWith("tbb") || lib.equals("gna") ? "2" : null;
                lib = getLibraryName(lib, version);
                File nativeLibTmpFile = new File(tmpDir, lib);
                try {
                    System.load(nativeLibTmpFile.getAbsolutePath());
                } catch (UnsatisfiedLinkError ex) {
                    logger.warning("Failed to load library " + file + ": " + ex);
                }
            }
        } catch (IOException ex) {
            logger.warning("Failed to load native Inference Engine libraries: " + ex.getMessage());
        } finally {
            if (resources_list != null) {
                try {
                    resources_list.close();
                } catch (IOException ex) {
                    logger.warning("Failed to read native libraries list");
                }
            }
        }
    }

    /**
     * Same as {@link Core#read_model(String, String)} but with empty weights path
     */
    public Model read_model(final String modelPath) {
        return new Model(ReadModel(nativeObj, modelPath));
    }

    /**
     * Reads models from IR/ONNX/PDPD formats.
     * @param modelPath Path to a model.
     * @param weightPath Path to a data file.
     * <p>
     * For IR format (*.bin):
     * <ul>
     * <li>if path is empty, will try to read a bin file with the same name as xml and</li>
     * <li>if the bin file with the same name is not found, will load IR without weights.</li>
     * </ul>
     * For ONNX format (*.onnx):
     * <ul>
     * <li>the bin_path parameter is not used.</li>
     * </ul>
     * For PDPD format (*.pdmodel)
     * <ul>
     * <li>the bin_path parameter is not used.</li>
     * </ul>
     * @return A model.
     */
    public Model read_model(final String modelPath, final String weightPath) {
        return new Model(ReadModel1(nativeObj, modelPath, weightPath));
    }

    /**
     * Creates a compiled model from a source model object.
     * <p>
     * Users can create as many compiled models as they need and use
     * them simultaneously (up to the limitation of the hardware resources).
     *
     * @param model Model object acquired from {@link Core#read_model}.
     * @param device Name of a device to load a model to.
     * @return A compiled model.
     */
    public CompiledModel compile_model(Model model, final String device) {
        return new CompiledModel(CompileModel(nativeObj, net.getNativeObjAddr(), device));
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long ReadModel(long core, final String modelPath);

    private static native long ReadModel1(
            long core, final String modelPath, final String weightPath);

    private static native long CompileModel(long core, long net, final String device);

    private static native long GetCore();

    @Override
    protected native void delete(long nativeObj);
}
