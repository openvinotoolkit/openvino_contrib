// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * This class represents an OpenVINO runtime Core entity.
 *
 * <p>User applications can create several Core class instances, but in this case the underlying
 * plugins are created multiple times and not shared between several Core instances. The recommended
 * way is to have a single Core instance per application.
 */
public class Core extends Wrapper {

    private static final Logger logger = Logger.getLogger(Core.class.getName());

    public Core() {
        super(GetCore());
    }

    public Core(String xmlConfigFile) {
        super(GetCore1(xmlConfigFile));
    }

    /** Same as {@link Core#read_model(String, String)} but with empty weights path */
    public Model read_model(final String modelPath) {
        return new Model(ReadModel(nativeObj, modelPath));
    }

    /**
     * Reads models from IR/ONNX/PDPD formats.
     *
     * @param modelPath Path to a model.
     * @param weightPath Path to a data file.
     *     <p>For IR format (*.bin):
     *     <ul>
     *       <li>if path is empty, will try to read a bin file with the same name as xml and
     *       <li>if the bin file with the same name is not found, will load IR without weights.
     *     </ul>
     *     For ONNX format (*.onnx):
     *     <ul>
     *       <li>the bin_path parameter is not used.
     *     </ul>
     *     For PDPD format (*.pdmodel)
     *     <ul>
     *       <li>the bin_path parameter is not used.
     *     </ul>
     *
     * @return A model.
     */
    public Model read_model(final String modelPath, final String weightPath) {
        return new Model(ReadModel1(nativeObj, modelPath, weightPath));
    }

    /**
     * Reads and loads a compiled model from the IR/ONNX/PDPD file to the default OpenVINO device
     * selected by the AUTO plugin.
     *
     * <p>This can be more efficient, especially when caching is enabled and a cached model is
     * available, than using the read_model followed by compile_model flow.
     *
     * @param path Path to a model.
     * @return A compiled model.
     */
    public CompiledModel compile_model(final String path) {
        return new CompiledModel(CompileModel1(nativeObj, path));
    }

    /**
     * Reads and loads a compiled model from the IR/ONNX/PDPD file.
     *
     * <p>This can be more efficient, especially when caching is enabled and a cached model is
     * available, than using the read_model followed by compile_model flow.
     *
     * @param path Path to a model.
     * @param device Name of a device to load a model to.
     * @return A compiled model.
     */
    public CompiledModel compile_model(final String path, final String device) {
        return new CompiledModel(CompileModel2(nativeObj, path, device));
    }

    /**
     * Reads and loads a compiled model from the IR/ONNX/PDPD file.
     *
     * <p>This can be more efficient, especially when caching is enabled and a cached model is
     * available, than using the read_model followed by compile_model flow.
     *
     * @param path Path to a model.
     * @param device Name of a device to load a model to.
     * @param properties Map of pairs: (property name, property value) relevant only for this load
     *     operation.
     * @return A compiled model.
     */
    public CompiledModel compile_model(
            final String path, final String device, final Map<String, String> properties) {
        return new CompiledModel(CompileModel3(nativeObj, path, device, properties));
    }

    /**
     * Creates a compiled model from a source model object.
     *
     * <p>Users can create as many compiled models as they need and use them simultaneously (up to
     * the limitation of the hardware resources).
     *
     * @param model Model object acquired from {@link Core#read_model}.
     * @param device Name of a device to load a model to.
     * @return A compiled model.
     */
    public CompiledModel compile_model(Model model, final String device) {
        return new CompiledModel(CompileModel(nativeObj, model.getNativeObjAddr(), device));
    }

    /**
     * Creates a compiled model from a source model object.
     *
     * <p>Users can create as many compiled models as they need and use them simultaneously (up to
     * the limitation of the hardware resources).
     *
     * @param model Model object acquired from {@link Core#read_model}.
     * @param device Name of a device to load a model to.
     * @param properties properties – Map of pairs: (property name, property value) relevant only
     *     for this load operation.
     * @return A compiled model.
     */
    public CompiledModel compile_model(
            Model model, final String device, final Map<String, String> properties) {
        return new CompiledModel(
                CompileModel4(nativeObj, model.getNativeObjAddr(), device, properties));
    }

    /**
     * Gets properties related to device behaviour.
     *
     * <p>The method extracts information that can be set via the set_property method.
     *
     * @param device Name of a device to get a property value.
     * @param name {@link Property} name.
     * @return Value of a property corresponding to the property name.
     */
    public Any get_property(final String device, final String name) {
        return new Any(GetProperty(nativeObj, device, name));
    }

    /**
     * Sets properties for a device, acceptable keys can be found in
     * openvino/runtime/properties.hpp.
     *
     * @param device Name of a device to get a property value.
     * @param prop Map of pairs: (property name, property value).
     */
    public void set_property(final String device, final Map<String, String> prop) {
        SetProperty(nativeObj, device, prop);
    }

    /**
     * Returns the devices available for inference.
     *
     * @return List of devices.
     *     <p>The devices are returned as { CPU, GPU.0, GPU.1, GNA }. If there is more than one
     *     device of a specific type, they are enumerated with the .# suffix. Such enumerated device
     *     can later be used as a device name in all Core methods like {@link Core#compile_model},
     *     {@link Core#set_property} and so on.
     */
    public List<String> get_available_devices() {
        return GetAvailableDevices(nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/

    private static native long GetCore();

    private static native long GetCore1(String xmlConfigFile);

    private static native long ReadModel(long core, final String modelPath);

    private static native long ReadModel1(
            long core, final String modelPath, final String weightPath);

    private static native long CompileModel(long core, long net, final String device);

    private static native long CompileModel1(long core, final String device);

    private static native long CompileModel2(
            long core, final String modelPath, final String device);

    private static native long CompileModel4(
            long core, long net, final String device, final Map<String, String> properties);

    private static native long CompileModel3(
            long core,
            final String modelPath,
            final String device,
            final Map<String, String> props);

    private static native long GetProperty(long core, final String device, final String name);

    private static native void SetProperty(
            long core, final String device, final Map<String, String> prop);

    private static native List<String> GetAvailableDevices(long core);

    @Override
    protected native void delete(long nativeObj);
}
