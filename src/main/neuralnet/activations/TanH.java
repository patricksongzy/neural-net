package main.neuralnet.activations;

import java.util.stream.IntStream;

public class TanH implements Activation {
	public ActivationType getType() {
		return ActivationType.TANH;
	}

	public void activation(double[] x) {
		IntStream.range(0, x.length).parallel().forEach(i -> {
			x[i] = Math.tanh(x[i]);
		});
	}

	public void activation(double[][] x) {
		IntStream.range(0, x.length).parallel().forEach(b -> {
			for (int i = 0; i < x[0].length; i++) {
				x[b][i] = Math.tanh(x[b][i]);
			}
		});
	}

	public double[][] derivative(double[][] x) {
		double[][] derivative = new double[x.length][x[0].length];

		IntStream.range(0, x.length).parallel().forEach(b -> {
			for (int i = 0; i < x[0].length; i++) {
				derivative[b][i] = 1 - Math.pow(x[b][i], 2);
			}
		});

		return derivative;
	}
}