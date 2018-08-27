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

	public float[] derivative(float[] output, float[] target, Activation activation, int batchSize) {
		float[] delta = new float[output.length];

		float[] derivative = activation.derivative(output);

		int size = output.length / batchSize;
		IntStream.range(0, batchSize).parallel().forEach(b -> {
			for (int i = 0; i < size; i++) {
				int index = i + size * b;

				if (activation.getType() == ActivationType.SOFTMAX)
					delta[index] = (output[index] - target[index]); // softmax derivative simplifies to this
				else
					delta[i + size * b] = (-target[index] / output[index]) * derivative[index];
			}
		});

		return delta;
	}
}