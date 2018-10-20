package neuralnet.layers;

import neuralnet.initializers.HeInitialization;
import neuralnet.optimizers.UpdaterType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ThreadLocalRandom;

class ResidualTest {

	@Test
	void forward() {
		Residual residual =
			new Residual.Builder().filterAmount(64).initializer(new HeInitialization()).updaterType(UpdaterType.ADAM).outputDepth(128).build();
		residual.setDimensions(28, 28, 3);

		float[] input = new float[28 * 28 * 3];
		for (int i = 0; i < input.length; i++)
			input[i] = ThreadLocalRandom.current().nextFloat();

		// doesn't do anything, was just to see if there would be any errors, but ResNet is confirmed working.
		residual.forward(input, 1);
	}
}