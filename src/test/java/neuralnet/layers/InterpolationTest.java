package neuralnet.layers;

import org.junit.jupiter.api.Test;

class InterpolationTest {
	@Test
	void forward() {
		Interpolation interpolation = new Interpolation.Builder().outputHeight(10).outputWidth(10).build();
		interpolation.setDimensions(new int[]{1, 2, 2}, null);

		float[] output = interpolation.forward(new float[]{5, 1, 1, 5}, 1);
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 10; j++) {
				System.out.print(String.format("%02f ", output[j + 10 * i]));
			}

			System.out.println();
		}
	}
}