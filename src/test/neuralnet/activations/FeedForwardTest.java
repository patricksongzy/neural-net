package test.neuralnet.activations;

import main.neuralnet.Model;
import main.neuralnet.costs.CostType;
import main.neuralnet.layers.FeedForward;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class FeedForwardTest {
	@Test
	void gradientTest() {
		// just a regular test
		Model model = new Model.Builder().add(new FeedForward.Builder().outputSize(5).build()).inputDimensions(2)
						.cost(CostType.MEAN_SQUARE_ERROR).build();
		assertTrue(model.gradientCheck(new double[][] {{0.2, 1}}, new double[][] {{0.3, 0.6, 0.1, 0.7, 0.9}}));
	}
}