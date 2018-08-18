package main.neuralnet.costs;

import main.neuralnet.activations.Activation;
import main.neuralnet.activations.ActivationType;

import java.util.stream.IntStream;

public class MeanSquareError implements Cost {
	public CostType getType() {
		return CostType.CROSS_ENTROPY;
	}

	public double cost(double[] out, double[] target) {
		double cost = IntStream.range(0, target.length).parallel().mapToDouble(i -> Math.pow(target[i] - out[i], 2)).sum();

		return 0.5 * cost;
	}

	public double[][] derivative(double[][] output, double[][] target, Activation activation) {
		double[][] delta = new double[output.length][output[0].length];

		if (activation.getType() == ActivationType.SOFTMAX)
			throw new UnsupportedOperationException();

		double[][] derivative = activation.derivative(output);

		IntStream.range(0, output.length).parallel().forEach(b -> {
			for (int i = 0; i < output[0].length; i++) {
				delta[b][i] = (output[b][i] - target[b][i]) * derivative[b][i];
			}
		});

		return delta;
	}
}