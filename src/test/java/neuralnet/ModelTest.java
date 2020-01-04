package neuralnet;

import neuralnet.activations.ActivationType;
import neuralnet.costs.CostType;
import neuralnet.initializers.HeInitialization;
import neuralnet.layers.Convolutional;
import neuralnet.optimizers.UpdaterType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ThreadLocalRandom;

class ModelTest {
	@Test void testModel() {
		Convolutional conv1 =
			new Convolutional.Builder().filterAmount(8).filterSize(2).initializer(new HeInitialization()).pad(1).stride(2)
			.activationType(ActivationType.RELU).build();
		Convolutional conv2 =
			new Convolutional.Builder(conv1).filterAmount(12).filterSize(3).initializer(new HeInitialization()).pad(1).stride(2)
				.activationType(ActivationType.RELU).build();

		Model model =
			new Model.Builder(false).add(conv2).inputDimensions(2, 36, 36).updaterType(UpdaterType.AMSGRAD).cost(CostType.MEAN_SQUARE_ERROR)
				.build();

		// just a regular test
		float[] input = new float[36 * 36 * 2 * 2];
		float[] target = new float[10 * 10 * 12 * 2];

		for (int i = 0; i < input.length; i++) {
			input[i] = ThreadLocalRandom.current().nextFloat();
		}

		for (int i = 0; i < target.length; i++) {
			target[i] = ThreadLocalRandom.current().nextFloat();
		}

		model.forward(input, 32);
		model.backward(target);
	}
}