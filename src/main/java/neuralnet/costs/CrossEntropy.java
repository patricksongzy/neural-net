package neuralnet.costs;

import neuralnet.activations.Activation;
import neuralnet.activations.ActivationType;

import java.util.stream.IntStream;

/**
 * The cross entropy loss is given by <code>sum(target * log(output))</code>
 */
public class CrossEntropy implements Cost {
	public CostType getType() {
		return CostType.CROSS_ENTROPY;
	}

	public float cost(float[] out, float[] target) {
		float cost = (float) IntStream.range(0, target.length).parallel().mapToDouble(i -> (target[i] * Math.log(out[i] + 1e-16))).sum();

		return -cost;
	}

	public float[] derivative(float[][] output, float[][] target, Activation activation) {
		float[] delta = new float[output.length * output[0].length];

		float[][] derivative = activation.derivative(output);

		IntStream.range(0, output.length).parallel().forEach(b -> {
			for (int i = 0; i < output[0].length; i++) {
				if (activation.getType() == ActivationType.SOFTMAX)
					delta[i + output[0].length * b] = (output[b][i] - target[b][i]); // softmax derivative simplifies to this
				else
					delta[i + output[0].length * b] = (-target[b][i] / output[b][i]) * derivative[b][i];
			}
		});

		return delta;
	}
}