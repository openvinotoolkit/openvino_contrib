package tests;

import static org.junit.Assert.*;

import org.intel.openvino.*;
import org.junit.Test;

public class TensorTests extends OVTest {
    @Test
    public void testGetTensorFromFloat() {
        int[] dimsArr = {1, 1, 2, 2};
        float[] data = {0.0f, 1.1f, 2.2f, 3.3f};

        Tensor tensor = new Tensor(dimsArr, data);

        assertArrayEquals(tensor.get_shape(), dimsArr);
        assertArrayEquals(tensor.data(), data, 0.0f);
    }
}
