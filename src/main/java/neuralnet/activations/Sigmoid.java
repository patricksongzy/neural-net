package neuralnet.activations;

import java.util.stream.IntStream;

public class Sigmoid implements Activation{
	public ActivationType getType() {
		return ActivationType.SIGMOID;
	}

	public void activation(float[] x, int batchSize) {
		IntStream.range(0, x.length).parallel().forEach(i -> x[i] = 1 / (float) (1 + Math.exp(-x[i])));
	}

	public float[] derivative(float[] x) {
		float[] derivative = new float[x.length];

		IntStream.range(0, x.length).parallel().forEach(i -> derivative[i] = x[i] * (1 - x[i]));

		return derivative;
	}
}