package main.neuralnet.costs;

import main.neuralnet.activations.Activation;
import main.neuralnet.activations.ActivationType;

import java.util.stream.IntStream;

/**
 * The cross entropy loss is given by <code>sum(target * log(output))</code>
 */
public class CrossEntropy implements Cost {
	public CostType getType() {
		return CostType.CROSS_ENTROPY;
	}

	public double cost(double[] out, double[] target) {
		double cost = IntStream.range(0, target.length).parallel().mapToDouble(i -> (target[i] * Math.log(out[i] + 1e-16))).sum();

		return -cost;
	}

	public double[][] derivative(double[][] output, double[][] target, Activation activation) {
		double[][] delta = new double[output.length][output[0].length];

		double[][] derivative = activation.derivative(output);

		IntStream.range(0, output.length).parallel().forEach(b -> {
			for (int i = 0; i < output[0].length; i++) {
				if (activation.getType() == ActivationType.SOFTMAX)
					delta[b][i] = (output[b][i] - target[b][i]); // softmax derivative simplifies to this
				else
					delta[b][i] = (-target[b][i] / output[b][i]) * derivative[b][i];
			}
		});

		return delta;
	}
}