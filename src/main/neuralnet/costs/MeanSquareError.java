package main.neuralnet.costs;

import com.amd.aparapi.Kernel;
import com.amd.aparapi.Range;
import main.neuralnet.activations.Activation;
import main.neuralnet.activations.ActivationType;

public class MeanSquareError implements Cost {
	private DeltaKernel deltaKernel;

	MeanSquareError() {
		deltaKernel = new DeltaKernel();
	}

	public CostType getType() {
		return CostType.CROSS_ENTROPY;
	}

	public double cost(double[] out, double[] target) {
		double cost = 0;
		for (int i = 0; i < target.length; i++)
			cost += Math.pow(target[i] - out[i], 2);

		return 0.5 * cost;
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
		private double[][] output, target, derivative, delta;

		void init(double[][] output, double[][] target, double[][] derivative, double[][] delta, byte softmax) {
			this.output = output;
			this.target = target;
			this.derivative = derivative;
			this.delta = delta;

			if (softmax == 1)
				throw new UnsupportedOperationException();
		}

		public void run() {
			int b = getGlobalId(0); // batch
			int i = getGlobalId(1); // index

			delta[b][i] = (output[b][i] - target[b][i]) * derivative[b][i];
		}
	}
}