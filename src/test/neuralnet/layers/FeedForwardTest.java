package test.neuralnet.layers;

import main.neuralnet.Model;
import main.neuralnet.activations.ActivationType;
import main.neuralnet.costs.CostType;
import main.neuralnet.layers.FeedForward;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FeedForwardTest {
	@Test
	void forwardTest() {
		FeedForward feedForward = new FeedForward.Builder().outputSize(5).build();
		feedForward.setDimensions(2);
		float[] updated = new float[]
			{1, 2, 1, 0, 1, 1, 0, 0, 1, 2};
		for (int i = 0; i < feedForward.getParameters()[0][0].length; i++)
			feedForward.getParameters()[0][0][i] = updated[i];

		assertArrayEquals(new float[]{5, 1, 3, 0, 5}, feedForward.forward(new float[][]{{1, 2}})[0]);
	}

	@Test
	void gradientTest() {
		// just a regular test
		Model model = new Model.Builder().add(new FeedForward.Builder().outputSize(5).build())
			.add(new FeedForward.Builder().outputSize(5).activationType(ActivationType.SOFTMAX).build()).inputDimensions(2)
			.cost(CostType.CROSS_ENTROPY).build();
		assertTrue(model.gradientCheck(new float[][]{{0.2f, 1}}, new float[][]{{0.1f, 0.05f, 0.55f, 0.2f, 0.1f}}));
	}
}