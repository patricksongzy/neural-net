package test.neuralnet.activations;

import main.neuralnet.activations.Activation;
import main.neuralnet.activations.ActivationType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class SoftmaxTest {
	private Activation softmax = ActivationType.SOFTMAX.create();

	@Test
	void activation() {
		float[] input = new float[]{-1000, -2000, -3000};
		softmax.activation(input);

		assertArrayEquals(new float[]{1, 0, 0}, input);
	}
}