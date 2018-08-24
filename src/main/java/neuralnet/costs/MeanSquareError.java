package neuralnet.costs;

import neuralnet.activations.Activation;
import neuralnet.activations.ActivationType;

import java.util.stream.IntStream;

public class MeanSquareError implements Cost {
	public CostType getType() {
		return CostType.CROSS_ENTROPY;
	}

	public float cost(float[] out, float[] target) {
		float cost = (float) IntStream.range(0, target.length).parallel().mapToDouble(i -> Math.pow(target[i] - out[i], 2)).sum();

		return 0.5f * cost;
	}

	public float[][] derivative(float[][] output, float[][] target, Activation activation) {
		float[][] delta = new float[output.length][output[0].length];

		if (activation.getType() == ActivationType.SOFTMAX)
			throw new UnsupportedOperationException();

		float[][] derivative = activation.derivative(output);

		IntStream.range(0, output.length).parallel().forEach(b -> {
			for (int i = 0; i < output[0].length; i++) {
				delta[b][i] = (output[b][i] - target[b][i]) * derivative[b][i];
			}
		});

		return delta;
	}
}