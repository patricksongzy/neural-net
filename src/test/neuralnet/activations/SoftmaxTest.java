package test.neuralnet.activations;

import main.neuralnet.activations.Activation;
import main.neuralnet.activations.ActivationType;
import main.neuralnet.activations.Softmax;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class SoftmaxTest {
	private Activation softmax = ActivationType.SOFTMAX.create();

	@Test
	void activation() {
		double[] input = new double[] {-1000, -2000, -3000};
		softmax.activation(input);

		assertArrayEquals(new double[] {1, 0, 0}, input);
	}
}