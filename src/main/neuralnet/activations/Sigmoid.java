package main.neuralnet.activations;

import java.util.stream.IntStream;

public class Sigmoid implements Activation{
	public ActivationType getType() {
		return ActivationType.SIGMOID;
	}

	public void activation(float[] x) {
		IntStream.range(0, x.length).parallel().forEach(i -> x[i] = 1 / (float) (1 + Math.exp(-x[i])));
	}

	public void activation(float[][] x) {
		IntStream.range(0, x.length).parallel().forEach(b -> {
			for (int i = 0; i < x[0].length; i++)
				x[b][i] = 1 / (float) (1 + Math.exp(-x[b][i]));
		});
	}

	public float[][] derivative(float[][] x) {
		float[][] derivative = new float[x.length][x[0].length];

		IntStream.range(0, x.length).parallel().forEach(b -> {
			// assuming the derivative at 0 is equal to 0
			for (int i = 0; i < x[0].length; i++)
				derivative[b][i] = x[b][i] * (1 - x[b][i]);
		});

		return derivative;
	}
}