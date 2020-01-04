package neuralnet.layers;

import neuralnet.Model;
import neuralnet.activations.ActivationType;
import neuralnet.costs.CostType;
import neuralnet.initializers.HeInitialization;
import neuralnet.optimizers.UpdaterType;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PoolingTest {
	@Test
	void forward() {
		Pooling pooling = new Pooling.Builder().downsampleSize(2).downsampleStride(2).build();
		pooling.setDimensions(new int[]{1, 4, 4}, null);
		float[] input = new float[]{1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4};
		System.out.println(Arrays.toString(pooling.forward(input, 1)));
		assertArrayEquals(new float[]{6, 8, 3, 4}, pooling.forward(input, 1));
	}

	@Test
	void backward() {
		Pooling pooling = new Pooling.Builder().downsampleSize(2).downsampleStride(2).build();
		pooling.setDimensions(new int[]{1, 4, 4}, null);
		pooling.forward(new float[]{1, 1, 2, 4, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4}, 1);
		assertArrayEquals(new float[]{0, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 0, 0, 0, 0, 4}, pooling.backward(new float[]{1, 2, 3, 4}, true));
	}

	@Test
	void gradientCheck() {
		Model model = new Model.Builder(true).add(
			new Convolutional.Builder().filterAmount(16).filterSize(2).initializer(new HeInitialization())
				.pad(2).stride(2).activationType(ActivationType.RELU).build()
		).add(
			new Pooling.Builder().downsampleSize(2).downsampleStride(2).mode(Pooling.Mode.MAX).build()
		).add(
			new Convolutional.Builder().filterAmount(16).filterSize(2).initializer(new HeInitialization())
				.pad(1).stride(1).activationType(ActivationType.RELU).build()
		).cost(CostType.MEAN_SQUARE_ERROR).updaterType(UpdaterType.ADAM).inputDimensions(3, 36, 36).build();

		float[] input = new float[36 * 36 * 3];
		for (int i = 0; i < input.length; i++) {
			input[i] = ThreadLocalRandom.current().nextFloat();
		}

		float[] target = new float[11 * 11 * 16];
		for (int i = 0; i < target.length; i++) {
			target[i] = ThreadLocalRandom.current().nextFloat();
		}

		assertTrue(model.gradientCheck(input, target, 1));
	}
}
