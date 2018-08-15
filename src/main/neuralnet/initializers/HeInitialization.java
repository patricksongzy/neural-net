package main.neuralnet.initializers;

import java.util.concurrent.ThreadLocalRandom;

public class HeInitialization implements Initializer {
	public double initialize(int inputSize) {
		double r = Math.sqrt(2.0 / inputSize); // recommended for ReLU

		return truncatedNormal(r);
	}

	private static double truncatedNormal(double stddev) {
		double random = ThreadLocalRandom.current().nextGaussian();
		while (Math.abs(random) > 2 * stddev)
			random = ThreadLocalRandom.current().nextGaussian();

		return random * stddev;
	}
}