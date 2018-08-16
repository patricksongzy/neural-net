package test.neuralnet.costs;

import main.neuralnet.activations.ActivationType;
import main.neuralnet.costs.Cost;
import main.neuralnet.costs.CostType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SparseCrossEntropyTest {
	private Cost crossEntropy = CostType.SPARSE_CROSS_ENTROPY.create();

	@Test
	void cost() {
		double cost = crossEntropy.cost(new double[] {0.2, 0.6, 0.1, 0.0, 0.1}, new double[] {2});

		assertEquals(2.30258509299, cost, 1e-8);
	}

	@Test
	void derivative() {
		double[][] delta = crossEntropy.derivative(new double[][] {{0.2, 0.6, 0.1, 0.0, 0.1}}, new double[][] {{2}}, ActivationType.SIGMOID.create());

		assertArrayEquals(new double[] {0, 0, -0.9, 0, 0}, delta[0], 1e-8);
	}

	@Test
	void derivativeSoftmax() {
		// this is just a general test
		double[][] delta = crossEntropy.derivative(new double[][] {{0.2, 0.6, 0.1, 0.0, 0.1}}, new double[][] {{2}}, ActivationType.SOFTMAX.create());

		assertArrayEquals(new double[] {0.2, 0.6, -0.9, 0.0, 0.1}, delta[0]);
	}
}