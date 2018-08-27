package neuralnet.costs;

import neuralnet.activations.Activation;
import neuralnet.activations.ActivationType;

import java.util.stream.IntStream;

public class MeanSquareError implements Cost {
	public CostType getType() {
		return CostType.MEAN_SQUARE_ERROR;
	}

	public float cost(float[] out, float[] target) {
		float cost = (float) IntStream.range(0, target.length).parallel().mapToDouble(i -> Math.pow(target[i] - out[i], 2)).sum();

		return 0.5f * cost;
	}

	public float[] derivative(float[] output, float[] target, Activation activation, int batchSize) {
		float[] delta = new float[output.length];

		if (activation.getType() == ActivationType.SOFTMAX)
			throw new UnsupportedOperationException();

		float[] derivative = activation.derivative(output);

		int size = output.length / batchSize;
		IntStream.range(0, batchSize).parallel().forEach(b -> {
			for (int i = 0; i < size; i++) {
				int index = i + size * b;

				delta[index] = (output[index] - target[index]) * derivative[index];
			}
		});

		return delta;
	}
}