package test.neuralnet.layers;

import main.neuralnet.Model;
import main.neuralnet.costs.CostType;
import main.neuralnet.layers.GRU;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class GRUTest {
	@Test
	void gradientTest() {
		// just a regular test
		Model model = new Model.Builder()
				.add(new GRU.Builder().hiddenSize(5).build()).inputDimensions(2).cost(CostType.MEAN_SQUARE_ERROR).build();
		assertTrue(model.gradientCheck(new double[][]{{0.3, 0.7}, {0.5, 0.2}}, new double[][]{{0.2, 0.3, 0.6, 0.1, 0.8}, {0.8, 0.7, 0.5,
				0.6, 0.2}}));
	}
}