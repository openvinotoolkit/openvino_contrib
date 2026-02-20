package org.intel.openvino;

import static org.junit.Assert.*;

import org.junit.Test;

import java.util.Arrays;

public class TensorTests extends OVTest {
    int[] dimsArr = {1, 3, 2, 2};
    float[] data = {0.0f, 1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f, 1.1f, 2.2f};

    @Test
    public void testGetTensorFromFloat() {
        Tensor tensor = new Tensor(dimsArr, data);

        assertArrayEquals(tensor.get_shape(), dimsArr);
        assertArrayEquals(tensor.data(), data, 0.0f);
        assertEquals(ElementType.f32, tensor.get_element_type());
    }

    @Test
    public void testGetTensorFromInt() {
        int size = Arrays.stream(dimsArr).reduce((i, j) -> i * j).orElse(1);
        int[] inputData = new int[size];
        Arrays.fill(inputData, 1);

        Tensor tensor = new Tensor(dimsArr, inputData);

        assertArrayEquals(dimsArr, tensor.get_shape());
        assertArrayEquals(inputData, tensor.as_int());
        assertEquals(size, tensor.get_size());
        assertEquals(ElementType.i32, tensor.get_element_type());
    }

    @Test
    public void testGetTensorFromLong() {
        int size = Arrays.stream(dimsArr).reduce((i, j) -> i * j).orElse(1);
        long[] inputData = new long[size];
        Arrays.fill(inputData, 1L);

        Tensor tensor = new Tensor(dimsArr, inputData);

        assertArrayEquals(dimsArr, tensor.get_shape());
        assertEquals(size, tensor.get_size());
        assertEquals(ElementType.i64, tensor.get_element_type());
    }
}
