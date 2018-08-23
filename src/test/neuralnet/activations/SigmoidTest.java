package test.neuralnet.activations;

import main.neuralnet.activations.Activation;
import main.neuralnet.activations.ActivationType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class SigmoidTest {
	private Activation sigmoid = ActivationType.SIGMOID.create();

	@Test
	void activation() {
		float[] input = new float[]{-1000, -2000, -3000};

		sigmoid.activation(input);

		assertArrayEquals(new float[]{0, 0, 0}, input);
	}

	@Test
	void derivative() {

		float[][] input = new float[][]{{0, 0.5f, 1}};

		input = sigmoid.derivative(input);

		assertArrayEquals(new float[][]{{0, 0.25f, 0}}, input);
	}
}