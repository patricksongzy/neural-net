package main.neuralnet.costs;

import com.aparapi.Kernel;
import com.aparapi.Range;
import main.neuralnet.activations.Activation;
import main.neuralnet.activations.ActivationType;

import java.util.stream.IntStream;

/**
 * The cross entropy loss is given by <code>sum(target * log(output))</code>
 */
public class SparseCrossEntropy implements Cost {
	private DeltaKernel deltaKernel;

	SparseCrossEntropy() {
		deltaKernel = new DeltaKernel();
	}

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
			deltaKernel.init(output, target, delta);
			deltaKernel.execute(Range.create2D(output.length, output[0].length));
		} else {
			IntStream.range(0, output.length).parallel().forEach(b -> {
				int i = (int) target[b][0];
				delta[b][i] = (-1 / output[b][i]) * derivative[b][i];
			});
		}

		return delta;
	}

	/**
	 * The DeltaKernel calculates the delta of an output layer.
	 */
	class DeltaKernel extends Kernel {
		private double[][] output, target, delta;

		void init(double[][] output, double[][] target, double[][] delta) {
			this.output = output;
			this.target = target;
			this.delta = delta;
		}

		public void run() {
			int b = getGlobalId(0); // batch
			int i = getGlobalId(1); // index

			if (target[b].length > 1 || target[b][0] > output[b].length)
				throw new IllegalArgumentException();

			delta[b][i] = output[b][i]; // softmax derivative simplifies to this
			if (i == target[b][0]) {
				delta[b][i] -= 1;
			}
		}
	}
}