package neuralnet.initializers;

import java.util.concurrent.ThreadLocalRandom;

public class HeInitialization implements Initializer {
	public float initialize(int inputSize) {
		if (inputSize < 0)
			throw new IllegalArgumentException("Input size must be > 0.");

		double r = Math.sqrt(2.0 / inputSize);

		return (float) truncatedNormal(r);
	}

	private static double truncatedNormal(double stddev) {
		double random = ThreadLocalRandom.current().nextGaussian();
		while (Math.abs(random) > 2 * stddev)
			random = ThreadLocalRandom.current().nextGaussian();

		return random * stddev;
	}
}