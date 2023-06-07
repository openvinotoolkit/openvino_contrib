package org.intel.openvino;

import static org.junit.Assert.*;

import org.junit.Test;

public class TensorTests extends OVTest {
    int[] dimsArr = {1, 3, 2, 2};
    float[] data = {0.0f, 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f, 1.1f, 2.2f};

    @Test
    public void testGetTensorFromFloat() {
        Tensor tensor = new Tensor(dimsArr, data);

        assertArrayEquals(tensor.get_shape(), dimsArr);
        assertArrayEquals(tensor.data(), data, 0.0f);
    }
}
