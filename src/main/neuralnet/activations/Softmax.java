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

	public void activation(float[] x) {
		float top = Float.NEGATIVE_INFINITY;
		AtomicReference<Float> sum = new AtomicReference<>((float) 0);

		for (float value : x) {
			if (value > top) {
				top = value;
			}
		}

		final float max = top;
		IntStream.range(0, x.length).parallel().forEach(i -> {
			float value = (float) Math.exp(x[i] - max);

			sum.updateAndGet(v -> v + value);
			x[i] = value;
		});

		for (int i = 0; i < x.length; i++)
			x[i] /= sum.get();
	}

	public void activation(float[][] x) {
		float[] sum = new float[x.length];
		float[] max = new float[x.length];

		IntStream.range(0, x.length).parallel().forEach(b -> {
			max[b] = Float.NEGATIVE_INFINITY;

			for (int i = 0; i < x[0].length; i++)
				if (x[b][i] > max[b])
					max[b] = x[b][i];

			for (int i = 0; i < x[0].length; i++) {
				float value = (float) Math.exp(x[b][i] - max[b]);

				sum[b] += value;
				x[b][i] = value;
			}

			for (int i = 0; i < x[0].length; i++)
				x[b][i] /= sum[b];
		});
	}

	public float[][] derivative(float[][] x) {
		return x;
	}
}
