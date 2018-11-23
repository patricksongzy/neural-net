package neuralnet.layers;

import neuralnet.Model;
import neuralnet.costs.CostType;
import neuralnet.initializers.HeInitialization;
import neuralnet.optimizers.UpdaterType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class GRUTest {
	@Test
	void gradientTest() {
		// just a regular test
		Model model = new Model.Builder()
			.add(new GRU.Builder().hiddenSize(5).initializer(new HeInitialization()).build())
			.add(new GRU.Builder().hiddenSize(5).initializer(new HeInitialization()).build())
			.inputDimensions(2).cost(CostType.MEAN_SQUARE_ERROR).updaterType(UpdaterType.ADAM).build();
		model.export("src/test/resources/gru-import-test.model");
		model = new Model("src/test/resources/gru-import-test.model");

		assertTrue(model.gradientCheck(new float[][]{{0.3f, 0.7f, 0.5f, 0.6f}, {0.3f, 0.7f, 0.5f, 0.6f}}, new float[][]{
			{0.2f, 0.3f, 0.6f, 0.1f, 0.8f, 0.8f, 0.7f, 0.5f, 0.6f, 0.2f}, {0.2f, 0.3f, 0.6f, 0.1f, 0.8f, 0.8f, 0.7f, 0.5f, 0.6f, 0.2f}
		}, 2));
	}
}