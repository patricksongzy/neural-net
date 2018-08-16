package test.neuralnet.activations;

import main.neuralnet.activations.Activation;
import main.neuralnet.activations.ActivationType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SigmoidTest {
	private Activation sigmoid = ActivationType.SIGMOID.create();

	@Test
	void activation() {
		double[] input = new double[] {-1000, -2000, -3000};

		sigmoid.activation(input);

		assertArrayEquals(new double[] {0, 0, 0}, input);
	}

	@Test
	void derivative() {

		double[][] input = new double[][] {{0, 0.5, 1}};

		input = sigmoid.derivative(input);

		assertArrayEquals(new double[][] {{0, 0.25, 0}}, input);
	}
}