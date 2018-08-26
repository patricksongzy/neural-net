package neuralnet.activations;

import java.util.stream.IntStream;

public class TanH implements Activation {
	public ActivationType getType() {
		return ActivationType.TANH;
	}

	public void activation(float[] x, int batchSize) {
		IntStream.range(0, x.length).parallel().forEach(i -> x[i] = (float) Math.tanh(x[i]));
	}

	public float[] derivative(float[] x) {
		float[] derivative = new float[x.length];

		IntStream.range(0, x.length).parallel().forEach(i -> derivative[i] = 1 - (float) Math.pow(x[i], 2));

		return derivative;
	}
}