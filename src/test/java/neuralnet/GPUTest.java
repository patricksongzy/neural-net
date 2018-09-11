package neuralnet;

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
	void saxpy() {
		float[] a = new float[]{1, 2, 3, 4, 5};
		float[] b = new float[]{5, 4, 3, 2, 1};

		assertArrayEquals(new float[]{5.5f, 5, 4.5f, 4, 3.5f}, GPU.saxpy(5, 0.5f, a, b));
	}
}