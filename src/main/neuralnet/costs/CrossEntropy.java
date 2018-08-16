package main.neuralnet.costs;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Range;
import main.neuralnet.activations.Activation;
import main.neuralnet.activations.ActivationType;

/**
 * The cross entropy loss is given by <code>sum(target * log(output))</code>
 */
public class CrossEntropy implements Cost {
	private DeltaKernel deltaKernel;

	CrossEntropy() {
		deltaKernel = new DeltaKernel();
	}

	public CostType getType() {
		return CostType.CROSS_ENTROPY;
	}

	public double cost(double[] out, double[] target) {
		double cost = 0;
		for (int i = 0; i < target.length; i++)
			cost += (target[i] * Math.log(out[i] + 1e-16));

		return -cost;
	}

	public double[][] derivative(double[][] output, double[][] target, Activation activation) {
		double[][] delta = new double[output.length][output[0].length];

		double[][] derivative = activation.derivative(output);
		deltaKernel.init(output, target, derivative, delta, (byte) (activation.getType() == ActivationType.SOFTMAX ? 1 : 0));
		deltaKernel.execute(Range.create2D(output.length, output[0].length));

		return delta;
	}

	/**
	 * The DeltaKernel calculates the delta of an output layer.
	 */
	class DeltaKernel extends Kernel {
		private byte softmax;
		private double[][] output, target, derivative, delta;

		void init(double[][] output, double[][] target, double[][] derivative, double[][] delta, byte softmax) {
			this.output = output;
			this.target = target;
			this.derivative = derivative;
			this.delta = delta;
			this.softmax = softmax;
		}

		public void run() {
			int b = getGlobalId(0); // batch
			int i = getGlobalId(1); // index

			if (softmax == 1)
				delta[b][i] = (output[b][i] - target[b][i]); // softmax derivative simplifies to this
			else
				delta[b][i] = (-target[b][i] / output[b][i]) * derivative[b][i];
		}
	}
}