package main.neuralnet.activations;

import java.util.stream.IntStream;

public class TanH implements Activation {
	public ActivationType getType() {
		return ActivationType.TANH;
	}

	public void activation(float[] x) {
		IntStream.range(0, x.length).parallel().forEach(i -> x[i] = (float) Math.tanh(x[i]));
	}

	public void activation(float[][] x) {
		IntStream.range(0, x.length).parallel().forEach(b -> {
			for (int i = 0; i < x[0].length; i++) {
				x[b][i] = (float) Math.tanh(x[b][i]);
			}
		});
	}

	public float[][] derivative(float[][] x) {
		float[][] derivative = new float[x.length][x[0].length];

		IntStream.range(0, x.length).parallel().forEach(b -> {
			for (int i = 0; i < x[0].length; i++) {
				derivative[b][i] = 1 - (float) Math.pow(x[b][i], 2);
			}
		});

		return derivative;
	}
}