package main.neuralnet.activations;

import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

/**
 * The softmax activation is given by <code>e^x_i/sum(e^x)</code>
 */
public class Softmax implements Activation {
	public ActivationType getType() {
		return ActivationType.SOFTMAX;
	}

	public void activation(double[] x) {
		double top = Float.NEGATIVE_INFINITY;
		AtomicReference<Double> sum = new AtomicReference<>((double) 0);

		for (double value : x) {
			if (value > top) {
				top = value;
			}
		}

		final double max = top;
		IntStream.range(0, x.length).parallel().forEach(i -> {
			double value = Math.exp(x[i] - max);

			sum.updateAndGet(v -> v + value);
			x[i] = value;
		});

		for (int i = 0; i < x.length; i++)
			x[i] /= sum.get();
	}

	public void activation(double[][] x) {
		double[] sum = new double[x.length];
		double[] max = new double[x.length];

		IntStream.range(0, x.length).parallel().forEach(b -> {
			max[b] = Float.NEGATIVE_INFINITY;

			for (int i = 0; i < x[0].length; i++)
				if (x[b][i] > max[b])
					max[b] = x[b][i];

			for (int i = 0; i < x[0].length; i++) {
				double value = Math.exp(x[b][i] - max[b]);

				sum[b] += value;
				x[b][i] = value;
			}

			for (int i = 0; i < x[0].length; i++)
				x[b][i] /= sum[b];
		});
	}

	public double[][] derivative(double[][] x) {
		return x;
	}
}
