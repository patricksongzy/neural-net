package neuralnet.activations;

import java.util.stream.IntStream;

/**
 * The ReLU activation is given by <code>max(0, x)</code>.
 */
public class ReLU implements Activation {
	public ActivationType getType() {
		return ActivationType.RELU;
	}

	public void activation(float[] x, int batchSize) {
		IntStream.range(0, x.length).parallel().forEach(i -> {
			// same as max of 0 and x
			if (x[i] < 0)
				x[i] = 0;
		});
	}

	public float[] derivative(float[] x) {
		float[] derivative = new float[x.length];

		IntStream.range(0, x.length).parallel().forEach(i -> derivative[i] = x[i] > 0 ? 1 : 0);

		return derivative;
	}
}
