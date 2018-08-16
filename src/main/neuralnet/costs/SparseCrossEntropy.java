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

	public double cost(double[] out, double[] target) {
		if (target.length > 1 || target[0] > out.length)
			throw new IllegalArgumentException();

		return -Math.log(out[(int) target[0]] + 1e-16);
	}

	public double[][] derivative(double[][] output, double[][] target, Activation activation) {
		double[][] delta = new double[output.length][output[0].length];
		double[][] derivative = activation.derivative(output);

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