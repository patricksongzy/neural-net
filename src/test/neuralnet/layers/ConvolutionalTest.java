package test.neuralnet.layers;

import main.neuralnet.Model;
import main.neuralnet.activations.ActivationType;
import main.neuralnet.costs.CostType;
import main.neuralnet.initializers.HeInitialization;
import main.neuralnet.layers.Convolutional;
import main.neuralnet.optimizers.UpdaterType;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

import static org.junit.jupiter.api.Assertions.*;

class ConvolutionalTest {
	@Test
	void padTest() {
		Convolutional convolutional = new Convolutional.Builder().filterAmount(1).activationType(ActivationType.RELU).filterSize(3)
				.initializer(new HeInitialization()).pad(2).stride(2).updaterType(UpdaterType.ADAM).build();
		convolutional.setDimensions(2, 2, 2);
		double[][] input = new double[][] {{
				1, 2,
				2, 1,

				3, 3,
				1, 3}
		};

		double[] target = new double[]{
				0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0,
				0, 0, 1, 2, 0, 0,
				0, 0, 2, 1, 0, 0,
				0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0,

				0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0,
				0, 0, 3, 3, 0, 0,
				0, 0, 1, 3, 0, 0,
				0, 0, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0,};

		assertArrayEquals(target, convolutional.pad(input)[0]);
	}

	@Test
	void convolutionTest() {
		Convolutional convolutional = new Convolutional.Builder().filterAmount(1).activationType(ActivationType.RELU).filterSize(3)
				.initializer(new HeInitialization()).pad(2).stride(2).updaterType(UpdaterType.ADAM).build();
		convolutional.setDimensions(3, 3, 1);
		double[][] input = new double[][] {
				{2, 1, 0,
				2, 0, 1,
				1, 2, 0,}};

		double[] updated = new double[] {2, 1, 3, 1, 2, 1, 3, 1, 1};
		for (int i = 0; i < convolutional.getParameters()[0][0].length; i++) {
			convolutional.getParameters()[0][0][i] = updated[i];
		}

		assertArrayEquals(new double[] {2, 7, 0, 9, 13, 1, 3, 4, 0}, convolutional.forward(input)[0]);
	}

	@Test
	void gradientTest() {
		Model model = new Model.Builder().add(new Convolutional.Builder().filterAmount(16).filterSize(2)
				.initializer(new HeInitialization()).updaterType(UpdaterType.ADAM).pad(1).stride(2).activationType(ActivationType.RELU).build())
				.cost(CostType.MEAN_SQUARE_ERROR).inputDimensions(32, 36, 1).build();

		// just a regular test
		double[][] input = new double[1][32 * 32 * 3];
		double[][] target = new double[1][17 * 19 * 16];

		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[0].length; j++)
				input[i][j] = ThreadLocalRandom.current().nextDouble();
			for (int j = 0; j < target[0].length; j++)
				target[i][j] = ThreadLocalRandom.current().nextDouble();
		}

		assertTrue(model.gradientCheck(input, target));
	}
}