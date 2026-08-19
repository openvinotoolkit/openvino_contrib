// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
    public void testGetTensorFromByte() {
        int size = Arrays.stream(dimsArr).reduce((i, j) -> i * j).orElse(1);
        byte[] inputData = new byte[size];
        for (int i = 0; i < size; i++) {
            inputData[i] = (byte) (i * 10);
        }

        Tensor tensor = new Tensor(ElementType.u8, dimsArr, inputData);

        // A u8 tensor is not int/float pointer-representable, so it can't be read back through
        // as_int()/data(); as_byte() round-trips the raw bytes and verifies the JNI copy path.
        assertArrayEquals(dimsArr, tensor.get_shape());
        assertEquals(size, tensor.get_size());
        assertEquals(ElementType.u8, tensor.get_element_type());
        assertArrayEquals(inputData, tensor.as_byte());
    }

    @Test
    public void testGetTensorFromByteSignedRoundTrip() {
        int size = Arrays.stream(dimsArr).reduce((i, j) -> i * j).orElse(1);
        byte[] inputData = new byte[size];
        for (int i = 0; i < size; i++) {
            // Mix of negative and positive values to exercise the i8 (signed) copy path.
            inputData[i] = (byte) (i % 2 == 0 ? -i : i);
        }

        Tensor tensor = new Tensor(ElementType.i8, dimsArr, inputData);

        assertEquals(ElementType.i8, tensor.get_element_type());
        assertArrayEquals(inputData, tensor.as_byte());
    }

    @Test
    public void testAsByteWrongType() {
        Tensor tensor = new Tensor(dimsArr, data);
        try {
            tensor.as_byte();
            fail("Expected an exception when reading a non-byte tensor as bytes");
        } catch (Exception e) {
            assertTrue(e.getMessage().contains("only u8 and i8"));
        }
    }

    @Test
    public void testGetTensorFromByteWrongLength() {
        byte[] inputData = new byte[] {1, 2, 3};
        try {
            new Tensor(ElementType.u8, dimsArr, inputData);
            fail("Expected an exception for mismatched data length");
        } catch (Exception e) {
            assertTrue(e.getMessage().contains("does not match the tensor shape"));
        }
    }

    @Test
    public void testGetTensorFromByteWrongType() {
        int size = Arrays.stream(dimsArr).reduce((i, j) -> i * j).orElse(1);
        byte[] inputData = new byte[size];
        try {
            new Tensor(ElementType.f32, dimsArr, inputData);
            fail("Expected an exception for a non-byte element type");
        } catch (Exception e) {
            assertTrue(e.getMessage().contains("only u8 and i8"));
        }
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
