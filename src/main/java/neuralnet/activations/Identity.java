package neuralnet.activations;

import java.util.stream.IntStream;

/**
 * The Identity activation is given by <code>x</code>.
 */
public class Identity implements Activation {
	public ActivationType getType() {
		return ActivationType.IDENTITY;
	}

	public void activation(float[] x, int batchSize) {
	}

	public float[] derivative(float[] x) {
		float[] derivative = new float[x.length];

		IntStream.range(0, x.length).parallel().forEach(b -> derivative[b] = 1);

		return derivative;
	}
}