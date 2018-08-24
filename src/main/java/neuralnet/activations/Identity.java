package neuralnet.activations;

import java.util.stream.IntStream;

/**
 * The Identity activation is given by <code>x</code>.
 */
public class Identity implements Activation {
	public ActivationType getType() {
		return ActivationType.IDENTITY;
	}

	public void activation(float[] x) {
	}

	public void activation(float[][] x) {
	}

	public float[][] derivative(float[][] x) {
		float[][] derivative = new float[x.length][x[0].length];

		IntStream.range(0, x.length).parallel().forEach(b -> {
			for (int i = 0; i < x[0].length; i++) {
				derivative[b][i] = 1;
			}
		});

		return derivative;
	}
}