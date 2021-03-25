package org.intel.openvino;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.file.Files;
import java.util.Map;
import java.util.logging.Logger;

public class IECore extends IEWrapper {
    public static final String NATIVE_LIBRARY_NAME = "inference_engine_java_api";
    private static final Logger logger = Logger.getLogger(IECore.class.getName());

    public IECore() {
        super(GetCore());
    }

    public IECore(String xmlConfigFile) {
        super(GetCore1(xmlConfigFile));
    }

    private static String getLibraryName(String name, String linux_ver) {
        final String osName = System.getProperty("os.name").toLowerCase();
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
        final String[] nativeLibs = {
            "tbb",
            "tbbmalloc",
            "ngraph",
            "inference_engine_transformations",
            "inference_engine",
            "inference_engine_legacy",
            "inference_engine_lp_transformations",
            "onnx_importer",
            "inference_engine_java_api"
        };

        try {
            // Get a list of all native resources (libraries, plugins and other files).
            InputStream resources_list =
                    IECore.class.getClassLoader().getResourceAsStream("resources_list.txt");
            BufferedReader r = new BufferedReader(new InputStreamReader(resources_list));

            // Create a temporal folder to unpack native files.
            File tmpDir = Files.createTempDirectory("openvino-native").toFile();
            tmpDir.deleteOnExit();

            String file;
            while ((file = r.readLine()) != null) {
                URL url = IECore.class.getClassLoader().getResource(file);
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
            resources_list.close();

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
        }
    }

    public CNNNetwork ReadNetwork(final String modelPath, final String weightPath) {
        return new CNNNetwork(ReadNetwork1(nativeObj, modelPath, weightPath));
    }

    public CNNNetwork ReadNetwork(final String modelFileName) {
        return new CNNNetwork(ReadNetwork(nativeObj, modelFileName));
    }

    public ExecutableNetwork LoadNetwork(CNNNetwork net, final String device) {
        return new ExecutableNetwork(LoadNetwork(nativeObj, net.getNativeObjAddr(), device));
    }

    public ExecutableNetwork LoadNetwork(
            CNNNetwork net, final String device, final Map<String, String> config) {
        long network = LoadNetwork1(nativeObj, net.getNativeObjAddr(), device, config);
        return new ExecutableNetwork(network);
    }

    public void RegisterPlugin(String pluginName, String deviceName) {
        RegisterPlugin(nativeObj, pluginName, deviceName);
    }

    public void RegisterPlugin(String xmlConfigFile) {
        RegisterPlugins(nativeObj, xmlConfigFile);
    }

    public void UnregisterPlugin(String deviceName) {
        UnregisterPlugin(nativeObj, deviceName);
    }

    public void AddExtension(String extension) {
        AddExtension(nativeObj, extension);
    }

    public void AddExtension(String extension, String deviceName) {
        AddExtension1(nativeObj, extension, deviceName);
    }

    public void SetConfig(Map<String, String> config, String deviceName) {
        SetConfig(nativeObj, config, deviceName);
    }

    public void SetConfig(Map<String, String> config) {
        SetConfig1(nativeObj, config);
    }

    public Parameter GetConfig(String deviceName, String name) {
        return new Parameter(GetConfig(nativeObj, deviceName, name));
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long ReadNetwork(long core, final String modelFileName);

    private static native long ReadNetwork1(
            long core, final String modelPath, final String weightPath);

    private static native long LoadNetwork(long core, long net, final String device);

    private static native long LoadNetwork1(
            long core, long net, final String device, final Map<String, String> config);

    private static native void RegisterPlugin(long core, String pluginName, String deviceName);

    private static native void RegisterPlugins(long core, String xmlConfigFile);

    private static native void UnregisterPlugin(long core, String deviceName);

    private static native void AddExtension(long core, String extension);

    private static native void AddExtension1(long core, String extension, String deviceName);

    private static native void SetConfig(long core, Map<String, String> config, String deviceName);

    private static native void SetConfig1(long core, Map<String, String> config);

    private static native long GetConfig(long core, String deviceName, String name);

    private static native long GetCore();

    private static native long GetCore1(String xmlConfigFile);

    @Override
    protected native void delete(long nativeObj);
}
