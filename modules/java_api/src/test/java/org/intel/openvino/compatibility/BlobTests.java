package org.intel.openvino.compatibility;

import org.junit.Assert;
import org.junit.Test;

public class BlobTests extends IETest {

    @Test
    public void testGetBlob() {
        int[] dimsArr = {1, 3, 200, 200};
        TensorDesc tDesc = new TensorDesc(Precision.U8, dimsArr, Layout.NHWC);

        Blob blob = new Blob(tDesc);

        Assert.assertArrayEquals(blob.getTensorDesc().getDims(), dimsArr);
    }

    @Test
    public void testGetBlobFromFloat() {
        int[] dimsArr = {1, 1, 2, 2};
        TensorDesc tDesc = new TensorDesc(Precision.FP32, dimsArr, Layout.NHWC);

        float[] data = {0.0f, 1.1f, 2.2f, 3.3f};

        Blob blob = new Blob(tDesc, data);

        float detection[] = new float[blob.size()];
        blob.rmap().get(detection);

        Assert.assertArrayEquals(blob.getTensorDesc().getDims(), dimsArr);
        Assert.assertArrayEquals(data, detection, 0.0f);
    }

    @Test
    public void testGetBlobFromInt() {
        int[] dimsArr = {2, 1, 2, 1};
        TensorDesc tDesc = new TensorDesc(Precision.I32, dimsArr, Layout.NHWC);

        int[] data = {55, 12, -8, 0};

        Blob blob = new Blob(tDesc, data);

        int[] out = new int[blob.size()];
        blob.rmap().get(out);

        Assert.assertArrayEquals(blob.getTensorDesc().getDims(), dimsArr);
        Assert.assertArrayEquals("Data", data, out);
    }

    @Test
    public void testGetBlobFromLong() {
        int[] dimsArr = {1, 2, 1, 2};
        TensorDesc tDesc = new TensorDesc(Precision.I64, dimsArr, Layout.NHWC);

        long[] data = {-99, 1, 111, -2};

        Blob blob = new Blob(tDesc, data);

        long[] out = new long[blob.size()];
        blob.rmap().get(out);

        Assert.assertArrayEquals(blob.getTensorDesc().getDims(), dimsArr);
        Assert.assertArrayEquals("Data", data, out);
    }
}
