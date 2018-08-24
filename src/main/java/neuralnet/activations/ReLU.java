package neuralnet.activations;

import java.util.stream.IntStream;

/**
 * The ReLU activation is given by <code>max(0, x)</code>.
 */
public class ReLU implements Activation {
	public ActivationType getType() {
		return ActivationType.RELU;
	}

	public void activation(float[] x) {
		IntStream.range(0, x.length).parallel().forEach(i -> {
			// same as max of 0 and x
			if (x[i] < 0)
				x[i] = 0;
		});
	}

	public void activation(float[][] x) {
		IntStream.range(0, x.length).parallel().forEach(b -> {
			for (int i = 0; i < x[0].length; i++)
				// same as max of 0 and x
				if (x[b][i] < 0)
					x[b][i] = 0;
		});
	}

	public float[][] derivative(float[][] x) {
		float[][] derivative = new float[x.length][x[0].length];

		IntStream.range(0, x.length).parallel().forEach(b -> {
			// assuming the derivative at 0 is equal to 0
			for (int i = 0; i< x[0].length; i++)
				derivative[b][i] = x[b][i] > 0 ? 1 : 0;
		});

		return derivative;
	}
}
