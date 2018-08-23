package test;

import main.GPU;
import org.jocl.blast.CLBlastTranspose;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class GPUTest {
	@Test
	void sgemm() {
		float[] a = new float[]{1, 2, 3, 4, 5, 6};
		float[] b = new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
		float[] c = new float[]{1, 2, 1, 2, 1, 2};

		assertArrayEquals(new float[]{31, 38, 43, 68, 82, 98}, GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo,
			CLBlastTranspose.CLBlastTransposeNo, 2, 3, 3, a, 3, b, 3, c, 3));
	}

	@Test
	void sger() {
		float[] x = new float[]{1, 2, 3, 4, 5};
		float[] y = new float[]{1, 2, 3};
		float[] a = new float[15];

		assertArrayEquals(new float[]{1, 2, 3, 2, 4, 6, 3, 6, 9, 4, 8, 12, 5, 10, 15}, GPU.sger(5, 3, x, y, a, 3));
	}
}