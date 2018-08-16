package test.neuralnet.costs;

import main.neuralnet.activations.ActivationType;
import main.neuralnet.costs.Cost;
import main.neuralnet.costs.CostType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MeanSquareErrorTest {
	private Cost mse = CostType.MEAN_SQUARE_ERROR.create();

	@Test
	void cost() {
		assertEquals(0.095, mse.cost(new double[] {0.3, 0.2, 0.6, 0.4, 0.7}, new double[] {0.2, 0.1, 0.6, 0.3, 0.3}), 1e-8);
	}

	@Test
	void derivative() {
		assertArrayEquals(new double[] {0.1, 0.1, 0.0, 0.1, 0.4},
			mse.derivative(new double[][] {{0.3, 0.2, 0.6, 0.4, 0.7}},
			new double[][] {{0.2, 0.1, 0.6, 0.3, 0.3}}, ActivationType.IDENTITY.create())[0], 1e-8);
	}
}