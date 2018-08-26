package neuralnet.layers;

import neuralnet.Model;
import neuralnet.activations.ActivationType;
import neuralnet.costs.CostType;
import neuralnet.initializers.HeInitialization;
import neuralnet.optimizers.UpdaterType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ThreadLocalRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ConvolutionalTest {
	@Test
	void padTest() {
		Convolutional convolutional = new Convolutional.Builder().filterAmount(1).activationType(ActivationType.RELU).filterSize(3)
			.initializer(new HeInitialization()).pad(3).stride(2).updaterType(UpdaterType.ADAM).build();
		convolutional.setDimensions(2, 2, 2);
		float[] input = new float[]
			{
				1, 2,
				2, 1,

				3, 3,
				1, 3,

				1, 2,
				2, 1,

				3, 3,
				1, 3
			};

		float[] target = new float[]
			{
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 1, 2, 0, 0, 0,
				0, 0, 0, 2, 1, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,

				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 3, 3, 0, 0, 0,
				0, 0, 0, 1, 3, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,

				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 1, 2, 0, 0, 0,
				0, 0, 0, 2, 1, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,

				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 3, 3, 0, 0, 0,
				0, 0, 0, 1, 3, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0,
			};

		assertArrayEquals(target, convolutional.pad(input, 2));
	}

	@Test
	void convolutionTest() {
		Convolutional convolutional = new Convolutional.Builder().filterAmount(1).activationType(ActivationType.RELU).filterSize(3)
				.initializer(new HeInitialization()).pad(2).stride(2).updaterType(UpdaterType.ADAM).build();
		convolutional.setDimensions(3, 3, 1);
		float[][] input = new float[][]{
				{2, 1, 0,
				2, 0, 1,
				1, 2, 0,}};

		float[] updated = new float[]{2, 1, 3, 1, 2, 1, 3, 1, 1};
		for (int i = 0; i < convolutional.getParameters()[0][0].length; i++) {
			convolutional.getParameters()[0][0][i] = updated[i];
		}

		assertArrayEquals(new float[][]{{2, 7, 0, 9, 13, 1, 3, 4, 0}}, convolutional.forward(input, 1));
	}

	@Test
	void gradientTest() {
		Model model = new Model.Builder().add(new Convolutional.Builder().filterAmount(16).filterSize(2)
				.initializer(new HeInitialization()).updaterType(UpdaterType.ADAM).pad(1).stride(2).activationType(ActivationType.RELU).build())
				.cost(CostType.MEAN_SQUARE_ERROR).inputDimensions(32, 36, 1).build();

		// just a regular test
		float[] input = new float[32 * 32 * 3];
		float[] target = new float[17 * 19 * 16];

		for (int i = 0; i < input.length; i++) {
			input[i] = ThreadLocalRandom.current().nextFloat();
			target[i] = ThreadLocalRandom.current().nextFloat();
		}

		assertTrue(model.gradientCheck(new float[][]{input}, new float[][]{target}, 1));
	}
}