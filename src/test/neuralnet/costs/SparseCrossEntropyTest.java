package test.neuralnet.costs;

import main.neuralnet.activations.ActivationType;
import main.neuralnet.costs.Cost;
import main.neuralnet.costs.CostType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class SparseCrossEntropyTest {
	private Cost crossEntropy = CostType.SPARSE_CROSS_ENTROPY.create();

	@Test
	void cost() {
		float cost = crossEntropy.cost(new float[]{0.2f, 0.6f, 0.1f, 0.0f, 0.1f}, new float[]{2});

		assertEquals(2.30258509299f, cost, 1e-8f);
	}

	@Test
	void derivative() {
		float[][] delta = crossEntropy.derivative(new float[][]{{0.2f, 0.6f, 0.1f, 0.0f, 0.1f}}, new float[][]{{2}},
			ActivationType.SIGMOID.create());

		assertArrayEquals(new float[]{0, 0, -0.9f, 0, 0}, delta[0], 1e-8f);
	}

	@Test
	void derivativeSoftmax() {
		// this is just a general test
		float[][] delta = crossEntropy.derivative(new float[][]{{0.2f, 0.6f, 0.1f, 0.0f, 0.1f}}, new float[][]{{2}},
			ActivationType.SOFTMAX.create());

		assertArrayEquals(new float[]{0.2f, 0.6f, -0.9f, 0.0f, 0.1f}, delta[0]);
	}
}