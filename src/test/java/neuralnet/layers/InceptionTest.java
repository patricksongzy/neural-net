package neuralnet.layers;

import neuralnet.initializers.HeInitialization;
import neuralnet.optimizers.UpdaterType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ThreadLocalRandom;

class InceptionTest {
	@Test
	void gradientCheck() {
		Inception inception = new Inception.Builder().initializer(new HeInitialization()).updaterType(UpdaterType.ADAM)
			.filterAmount(16, 32, 16, 16, 32, 32).build();
		inception.setDimensions(28, 28, 64);

		float[] input = new float[28 * 28 * 64];
		for (int i = 0; i < input.length; i++) {
			input[i] = ThreadLocalRandom.current().nextFloat();
		}

		inception.forward(input, 1);
	}
}