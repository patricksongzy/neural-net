package test.neuralnet.costs;

import main.neuralnet.activations.ActivationType;
import main.neuralnet.costs.Cost;
import main.neuralnet.costs.CostType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class MeanSquareErrorTest {
	private Cost mse = CostType.MEAN_SQUARE_ERROR.create();

	@Test
	void cost() {
		assertEquals(0.095, mse.cost(new float[]{0.3f, 0.2f, 0.6f, 0.4f, 0.7f}, new float[]{0.2f, 0.1f, 0.6f, 0.3f, 0.3f}), 1e-5f);
	}

	@Test
	void derivative() {
		assertArrayEquals(new float[]{0.1f, 0.1f, 0.0f, 0.1f, 0.4f},
			mse.derivative(new float[][]{{0.3f, 0.2f, 0.6f, 0.4f, 0.7f}},
				new float[][]{{0.2f, 0.1f, 0.6f, 0.3f, 0.3f}}, ActivationType.IDENTITY.create())[0], 1e-5f);
	}
}