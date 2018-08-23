package main.neuralnet.costs;

import main.neuralnet.activations.Activation;
import main.neuralnet.activations.ActivationType;

import java.util.stream.IntStream;

/**
 * The cross entropy loss is given by <code>sum(target * log(output))</code>
 */
public class SparseCrossEntropy implements Cost {
	public CostType getType() {
		return CostType.CROSS_ENTROPY;
	}

	public float cost(float[] out, float[] target) {
		if (target.length > 1 || target[0] > out.length)
			throw new IllegalArgumentException();

		return (float) -Math.log(out[(int) target[0]] + 1e-16);
	}

	public float[][] derivative(float[][] output, float[][] target, Activation activation) {
		float[][] delta = new float[output.length][output[0].length];
		float[][] derivative = activation.derivative(output);

		if (activation.getType() == ActivationType.SOFTMAX) {
			IntStream.range(0, output.length).parallel().forEach(b -> {
				System.arraycopy(output[b], 0, delta[b], 0, output[0].length);
				delta[b][(int) target[b][0]] -= 1;
			});
		} else {
			IntStream.range(0, output.length).parallel().forEach(b -> {
				int i = (int) target[b][0];
				delta[b][i] = (-1 / output[b][i]) * derivative[b][i];
			});
		}

		return delta;
	}
}