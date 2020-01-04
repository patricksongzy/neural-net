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

		assertArrayEquals(target, Convolutional.pad(input, 2, 3, 2, 8, 8, 2, 2));
	}

	@Test
	void convolutionTest() {
		Convolutional convolutional = new Convolutional.Builder().filterAmount(2).activationType(ActivationType.RELU).filterSize(3)
			.initializer(new HeInitialization()).pad(2).stride(2).build();
		convolutional.setDimensions(new int[]{1, 3, 3}, UpdaterType.ADAM);
		float[] input = new float[]{
			2, 1, 0,
			2, 0, 1,
			1, 2, 0,

			2, 1, 0,
			2, 0, 1,
			1, 2, 0};

		float[] updated = new float[]{
			4, 2, 6,
			2, 4, 2,
			6, 2, 2,

			4, 2, 6,
			2, 4, 2,
			6, 2, 2,

			4, 2, 6,
			2, 4, 2,
			6, 2, 2,

			4, 2, 6,
			2, 4, 2,
			6, 2, 2};

		float[] target = new float[]{4, 14, 0, 18, 26, 2, 6, 8, 0, 4, 14, 0, 18, 26, 2, 6, 8, 0,
			4, 14, 0, 18, 26, 2, 6, 8, 0, 4, 14, 0, 18, 26, 2, 6, 8, 0};

		for (int i = 0; i < convolutional.getParameters()[0][0].length; i++) {
			convolutional.getParameters()[0][0][i] = updated[i];
		}

		assertArrayEquals(target, convolutional.forward(input, 2));
	}

	@Test
	void dilationTest() {
		Convolutional convolutional = new Convolutional.Builder().filterAmount(2).activationType(ActivationType.RELU).filterSize(5)
			.initializer(new HeInitialization()).pad(2).stride(2).build();
		convolutional.setDimensions(new int[]{2, 3, 3}, UpdaterType.ADAM);
		float[] input = new float[]{
			2, 1, 0,
			2, 0, 1,
			1, 2, 0,

			2, 1, 0,
			2, 0, 1,
			1, 2, 0,};

		float[] updated = new float[]{
			4, 0, 2, 0, 6,
			0, 0, 0, 0, 0,
			2, 0, 4, 0, 2,
			0, 0, 0, 0, 0,
			6, 0, 2, 0, 2,

			4, 0, 2, 0, 6,
			0, 0, 0, 0, 0,
			2, 0, 4, 0, 2,
			0, 0, 0, 0, 0,
			6, 0, 2, 0, 2,

			4, 0, 2, 0, 6,
			0, 0, 0, 0, 0,
			2, 0, 4, 0, 2,
			0, 0, 0, 0, 0,
			6, 0, 2, 0, 2,

			4, 0, 2, 0, 6,
			0, 0, 0, 0, 0,
			2, 0, 4, 0, 2,
			0, 0, 0, 0, 0,
			6, 0, 2, 0, 2
		};

		for (int i = 0; i < convolutional.getParameters()[0][0].length; i++) {
			convolutional.getParameters()[0][0][i] = updated[i];
		}

		Convolutional dilated = new Convolutional.Builder().filterAmount(2).activationType(ActivationType.RELU).filterSize(3)
			.initializer(new HeInitialization()).pad(2).stride(2).dilation(2).build();
		dilated.setDimensions(new int[]{2, 3, 3}, UpdaterType.ADAM);

		updated = new float[]{
			4, 2, 6,
			2, 4, 2,
			6, 2, 2,

			4, 2, 6,
			2, 4, 2,
			6, 2, 2,

			4, 2, 6,
			2, 4, 2,
			6, 2, 2,

			4, 2, 6,
			2, 4, 2,
			6, 2, 2
		};

		for (int i = 0; i < dilated.getParameters()[0][0].length; i++) {
			dilated.getParameters()[0][0][i] = updated[i];
		}

		assertArrayEquals(convolutional.forward(input, 1), dilated.forward(input, 1));
	}

	@Test
	void gradientTest() {
		Model model = new Model.Builder(true).add(
			new Convolutional.Builder().filterAmount(8).filterSize(2).initializer(new HeInitialization())
				.pad(1).stride(2).activationType(ActivationType.RELU).build()
		).add(
			new Convolutional.Builder().filterAmount(12).filterSize(3).initializer(new HeInitialization())
				.pad(1).stride(2).activationType(ActivationType.RELU).build()
		).cost(CostType.MEAN_SQUARE_ERROR).updaterType(UpdaterType.ADAM).inputDimensions(2, 36, 36).build();

		// just a regular test
		float[] input = new float[36 * 36 * 2 * 2];
		float[] target = new float[10 * 10 * 12 * 2];

		for (int i = 0; i < input.length; i++) {
			input[i] = ThreadLocalRandom.current().nextFloat();
		}

		for (int i = 0; i < target.length; i++) {
			target[i] = ThreadLocalRandom.current().nextFloat();
		}

		assertTrue(model.gradientCheck(input, target, 2));
	}
}
