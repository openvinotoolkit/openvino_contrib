// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package org.intel.openvino;

/**
 * Layout represents the text information of tensor's dimensions/axes. E.g. layout `NCHW` means that
 * 4D tensor `{-1, 3, 480, 640}` will have:
 *
 * <ul>
 *   <li>0: `N = -1`: batch dimension is dynamic
 *   <li>1: `C = 3`: number of channels is '3'
 *   <li>2: `H = 480`: image height is 480
 *   <li>3: `W = 640`: image width is 640
 * </ul>
 *
 * Examples: `Layout` can be specified for:
 *
 * <ul>
 *   <li>Preprocessing purposes. E.g.
 *       <ul>
 *         <li>To apply normalization (means/scales) it is usually required to set 'C' dimension in
 *             a layout.
 *         <li>To resize the image to specified width/height it is needed to set 'H' and 'W'
 *             dimensions in a layout
 *         <li>To transpose image - source and target layout can be set (see
 *             `ov::preprocess::PreProcessSteps::convert_layout`)
 *       </ul>
 *   <li>To set/get model's batch (see {@link Model#get_batch}/'set_batch') it is required in
 *       general to specify 'N' dimension
 * </ul>
 *
 * in layout for appropriate inputs
 */
public class Layout extends Wrapper {

    public Layout(String str) {
        super(GetLayout(str));
    }

    /** Returns 'height' dimension index. */
    public static int height_idx(Layout layout) {
        return HeightIdx(layout.nativeObj);
    }

    /** Returns 'width' dimension index. */
    public static int width_idx(Layout layout) {
        return WidthIdx(layout.nativeObj);
    }

    /*----------------------------------- native methods -----------------------------------*/
    private static native long GetLayout(String str);

    private static native int HeightIdx(long layout);

    private static native int WidthIdx(long layout);

    @Override
    protected native void delete(long nativeObj);
}
