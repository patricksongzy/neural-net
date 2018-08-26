package neuralnet.costs;

import neuralnet.activations.Activation;
import neuralnet.activations.ActivationType;

import java.util.stream.IntStream;

/**
 * The cross entropy loss is given by <code>sum(target * log(output))</code>
 */
public class SparseCrossEntropy implements Cost {
	public CostType getType() {
		return CostType.SPARSE_CROSS_ENTROPY;
	}

	public float cost(float[] out, float[] target) {
		if (target.length > 1 || target[0] > out.length)
			throw new IllegalArgumentException();

		return (float) -Math.log(out[(int) target[0]] + 1e-16);
	}

	public float[] derivative(float[] output, float[] target, Activation activation, int batchSize) {
		float[] delta = new float[output.length];

		float[] derivative = activation.derivative(output);

		int size = output.length / batchSize;
		IntStream.range(0, batchSize).parallel().forEach(b -> {
			int index = (int) target[b] + size * b;

			if (activation.getType() == ActivationType.SOFTMAX) {
				System.arraycopy(output, size * b, delta, size * b, size);
				delta[index] -= 1;
			} else {
				delta[index] = (-1 / output[index]) * derivative[index];
			}
		});

		return delta;
	}
}