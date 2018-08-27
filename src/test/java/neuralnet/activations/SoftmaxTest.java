package neuralnet.activations;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class SoftmaxTest {
	private final Activation SOFTMAX = ActivationType.SOFTMAX.create();

	@Test
	void activation() {
		float[] input = new float[]{-1000, -2000, -3000};
		SOFTMAX.activation(input, 1);

		assertArrayEquals(new float[]{1, 0, 0}, input);
	}
}