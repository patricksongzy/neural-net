package neuralnet.initializers;

import java.util.concurrent.ThreadLocalRandom;

@SuppressWarnings("unused")
public class HeInitialization implements Initializer {
	public float initialize(int inputSize) {
		if (inputSize < 0)
			throw new IllegalArgumentException("Input size must be > 0.");

		double r = Math.sqrt(2.0 / inputSize);

		return (float) truncatedNormal(r);
	}

	private static double truncatedNormal(double stddev) {
		// weights must be within 2 standard deviations to prevent bad initializations
		double random;
		do {
			random = ThreadLocalRandom.current().nextGaussian() * stddev;
		} while (Math.abs(random) > 2 * stddev);

		return random;
	}
}