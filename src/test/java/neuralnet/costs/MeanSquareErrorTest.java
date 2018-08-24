package neuralnet.costs;

import neuralnet.activations.ActivationType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class MeanSquareErrorTest {
	private final Cost MSE = CostType.MEAN_SQUARE_ERROR.create();

	@Test
	void cost() {
		assertEquals(0.095, MSE.cost(new float[]{0.3f, 0.2f, 0.6f, 0.4f, 0.7f}, new float[]{0.2f, 0.1f, 0.6f, 0.3f, 0.3f}), 1e-5f);
	}

	@Test
	void derivative() {
		assertArrayEquals(new float[]{0.1f, 0.1f, 0.0f, 0.1f, 0.4f},
			MSE.derivative(new float[][]{{0.3f, 0.2f, 0.6f, 0.4f, 0.7f}},
				new float[][]{{0.2f, 0.1f, 0.6f, 0.3f, 0.3f}}, ActivationType.IDENTITY.create())[0], 1e-5f);
	}
}