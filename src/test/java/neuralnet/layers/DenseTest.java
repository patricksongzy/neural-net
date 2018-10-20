package neuralnet.layers;

import neuralnet.Model;
import neuralnet.activations.ActivationType;
import neuralnet.costs.CostType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DenseTest {
	@Test
	void forwardTest() {
		Dense dense = new Dense.Builder().outputSize(5).build();
		dense.setDimensions(2);
		float[] updated = new float[]
			{1, 2, 1, 0, 1, 1, 0, 0, 1, 2};
		float[] biases = new float[]{1, 0, 2, 1, 1};
		for (int i = 0; i < dense.getParameters()[0][0].length; i++)
			dense.getParameters()[0][0][i] = updated[i];
		for (int i = 0; i < dense.getParameters()[1][0].length; i++)
			dense.getParameters()[1][0][i] = biases[i];

		assertArrayEquals(new float[]{6, 1, 5, 1, 6, 6, 1, 5, 1, 6}, dense.forward(new float[]{1, 2, 1, 2}, 2));
	}

	@Test
	void gradientTest() {
		// just a regular test
		Model model = new Model.Builder()
			.add(new Dense.Builder().outputSize(5).activation(ActivationType.SIGMOID).build())
			.add(new Dense.Builder().outputSize(5).activation(ActivationType.SIGMOID).build())
			.inputDimensions(2)
			.cost(CostType.CROSS_ENTROPY).build();
		model.export("src/test/resources/ff-import-test.model");
		model = new Model("src/test/resources/ff-import-test.model");

		assertTrue(model.gradientCheck(new float[]{0.2f, 0.8f, 0.3f, 0.7f}, new float[]{0.3f, 0.1f, 0.3f, 0.2f, 0.1f,
			0.3f, 0.1f, 0.3f, 0.2f, 0.1f}, 2));
	}
}