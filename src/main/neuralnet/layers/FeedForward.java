package main.neuralnet.layers;

import com.aparapi.Kernel;
import com.aparapi.Range;
import main.neuralnet.activations.Activation;
import main.neuralnet.activations.ActivationType;
import main.neuralnet.costs.Cost;
import main.neuralnet.initializers.HeInitialization;
import main.neuralnet.initializers.Initializer;
import main.neuralnet.optimizers.Updater;
import main.neuralnet.optimizers.UpdaterType;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * FeedForward layers have a weight for each input. The inputs are multiplied to the weights, summed, then a bias term is added. The
 * output is
 * then run through an activation function.
 */
public class FeedForward implements Layer {
	private Mode mode = Mode.TRAIN;

	private int inputSize, outputSize;
	private double temperature;
	private ForwardKernel forwardKernel;
	private DeltaKernel deltaKernel;
	private Initializer initializer;
	private UpdaterType updaterType;
	private Activation activation;
	private Updater[] weightUpdaters;
	private Updater[] biasUpdaters;
	private double[] weights, biases;
	private double[] gradient, biasGradient;
	private double[][] input, output;

	/**
	 * Initializes a FeedForward layer neural network from a file.
	 *
	 * @param dis the input stream
	 */
	FeedForward(DataInputStream dis) throws IOException {
		inputSize = dis.readInt();
		outputSize = dis.readInt();
		temperature = dis.readDouble();

		activation = ActivationType.fromString(dis).create();
		updaterType = UpdaterType.fromString(dis);

		biases = new double[outputSize];
		biasUpdaters = new Updater[outputSize];
		weights = new double[outputSize * inputSize];
		weightUpdaters = new Updater[outputSize * inputSize];

		for (int i = 0; i < outputSize; i++) {
			biases[i] = dis.readDouble();
			biasUpdaters[i] = updaterType.create(dis);

			for (int j = 0; j < inputSize; j++) {
				int index = j + inputSize * i;

				weights[index] = dis.readDouble();
				weightUpdaters[index] = updaterType.create(dis);
			}
		}

		forwardKernel = new ForwardKernel(inputSize);
		deltaKernel = new DeltaKernel(inputSize, outputSize);
	}

	private FeedForward(int outputSize, double temperature, Initializer initializer, UpdaterType updaterType, ActivationType activationType) {
		this.outputSize = outputSize;

		this.temperature = temperature;

		this.initializer = initializer;
		this.updaterType = updaterType;

		activation = activationType.create();
	}

	public void setMode(Mode mode) {
		this.mode = mode;
	}

	public void setDimensions(int... dimensions) {
		if (dimensions.length != 1) {
			inputSize = 1;

			for (int dimension : dimensions) {
				inputSize *= dimension;
			}
		} else {
			inputSize = dimensions[0];
		}

		if (inputSize <= 0)
			throw new IllegalArgumentException();

		biases = new double[outputSize];
		biasUpdaters = new Updater[outputSize];
		weights = new double[outputSize * inputSize];
		weightUpdaters = new Updater[outputSize * inputSize];

		for (int i = 0; i < outputSize; i++) {
			biasUpdaters[i] = updaterType.create();

			for (int j = 0; j < inputSize; j++) {
				int index = j + inputSize * i;

				weights[index] = initializer.initialize(inputSize);
				weightUpdaters[index] = updaterType.create();
			}
		}

		forwardKernel = new ForwardKernel(inputSize);
		deltaKernel = new DeltaKernel(inputSize, outputSize);
	}

	public double[][][] getParameters() {
		return new double[][][] {{weights, gradient}, {biases, biasGradient}};
	}

	public double[][] forward(double[][] x) {
		input = x;
		output = new double[x.length][outputSize];

		// multiplying against weights
		forwardKernel.init(weights, biases, input, output);
		forwardKernel.execute(Range.create2D(x.length, outputSize));

		if (mode == Mode.EVAL && temperature != 1) {
			IntStream.range(0, x.length).parallel().forEach(b -> {
				for (int i = 0; i < outputSize; i++) {
					output[b][i] /= temperature;
				}
			});
		}

		// activating output
		activation.activation(output);

		return output;
	}

	public double[][] backward(Cost cost, double[][] target) {
		double[][] previousDelta = cost.derivative(output, target, activation);

		biasGradient = new double[outputSize];
		double[][] delta = new double[output.length][inputSize];

		for (int i = 0; i < outputSize; i++) {
			for (double[] d : previousDelta) {
				biasGradient[i] += d[i];
			}
		}

		getGradient(previousDelta);

		// calculating delta
		deltaKernel.init(previousDelta, weights, delta);
		deltaKernel.execute(Range.create2D(output.length, inputSize));

		return delta;
	}

	public double[][] backward(double[][] previousDelta) {
		output = activation.derivative(output);
		biasGradient = new double[outputSize];
		double[][] delta = new double[output.length][inputSize];

		for (int b = 0; b < previousDelta.length; b++) {
			for (int i = 0; i < outputSize; i++) {
				previousDelta[b][i] *= output[b][i];
			}
		}

		System.out.println(Arrays.toString(previousDelta[0]));

		for (int i = 0; i < outputSize; i++) {
			for (double[] d : previousDelta) {
				biasGradient[i] += d[i];
			}
		}

		// calculating gradient
		getGradient(previousDelta);

		// calculating dL/dx
		deltaKernel.init(previousDelta, weights, delta);
		deltaKernel.execute(Range.create2D(output.length, inputSize));

		return delta;
	}

	/**
	 * Calculates the gradient, then stores it, if on gradient check mod. Else, updates the parameters.
	 */
	private void getGradient(double[][] delta) {
		gradient = new double[outputSize * inputSize];

		IntStream.range(0, output.length).parallel().forEach(b -> {
			for (int i = 0; i < outputSize; i++) {
				for (int j = 0; j < inputSize; j++) {
					gradient[j + inputSize * i] += delta[b][i] * input[b][j];
				}
			}
		});

		if (mode == Mode.TRAIN) {
			// updating paramters
			update(gradient);
		}
	}

	/**
	 * Updates parameters given gradients.
	 *
	 * @param gradient the weight gradient
	 */
	private void update(double[] gradient) {
		IntStream.range(0, outputSize).parallel().forEach(i -> {
			biases[i] += biasUpdaters[i].update(biasGradient[i] / output.length);

			for (int j = 0; j < inputSize; j++) {
				int k = j + inputSize * i;

				weights[k] += weightUpdaters[k].update(gradient[k] / output.length);
			}
		});
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(inputSize);
		dos.writeInt(outputSize);
		dos.writeDouble(temperature);

		activation.getType().export(dos);
		updaterType.export(dos);

		for (int i = 0; i < outputSize; i++) {
			dos.writeDouble(biases[i]);
			biasUpdaters[i].export(dos);

			for (int j = 0; j < inputSize; j++) {
				int index = j + inputSize * i;

				dos.writeDouble(weights[index]);
				weightUpdaters[index].export(dos);
			}
		}
	}

	public int[] getOutputDimensions() {
		return new int[]{outputSize};
	}

	public LayerType getType() {
		return LayerType.FEED_FORWARD;
	}

	/**
	 * Builder for FeedForward layers.
	 */
	public static class Builder {
		private int outputSize;
		private double temperature;
		private UpdaterType updaterType;
		private ActivationType activationType;
		private Initializer initializer;

		public Builder() {
			temperature = 1;
			updaterType = UpdaterType.ADAM;
			activationType = ActivationType.RELU;
			initializer = new HeInitialization();
		}

		/**
		 * The output size is the amount of outputs for the layer.
		 *
		 * @param outputSize the output size
		 */
		public Builder outputSize(int outputSize) {
			this.outputSize = outputSize;
			return this;
		}

		/**
		 * The temperature is used on evaluation. The lower the temperature, the more conservative the network will be with it's
		 * predictions. DO NOT set (use default value) if using for classification or similar tasks.
		 *
		 * @param temperature the temperature
		 */
		public void setTemperature(double temperature) {
			this.temperature = temperature;
		}

		/**
		 * The updater updates parameters.
		 *
		 * @param updaterType the updater
		 */
		public Builder updaterType(UpdaterType updaterType) {
			if (updaterType != null)
				this.updaterType = updaterType;
			else
				throw new IllegalArgumentException();

			return this;
		}

		/**
		 * The activation simulates neurons firing.
		 *
		 * @param activationType the activation type
		 */
		public Builder activationType(ActivationType activationType) {
			if (activationType != null)
				this.activationType = activationType;
			else
				throw new IllegalArgumentException();

			return this;
		}

		/**
		 * The initializer initializes weights.
		 *
		 * @param initializer the initializer
		 */
		public Builder initializer(Initializer initializer) {
			if (initializer != null)
				this.initializer = initializer;
			else
				throw new IllegalArgumentException();

			return this;
		}

		public FeedForward build() {
			if (outputSize > 0 && temperature > 0)
				return new FeedForward(outputSize, temperature, initializer, updaterType, activationType);

			throw new IllegalArgumentException();
		}
	}

	/**
	 * Multiplies the inputs against the weights, sums them, then adds a bias term.
	 */
	class ForwardKernel extends Kernel {
		private int inputSize;
		private double[] weights, biases;
		private double[][] output, input;

		ForwardKernel(int inputSize) {
			this.inputSize = inputSize;
		}

		void init(double[] weights, double[] biases, double[][] input, double[][] output) {
			this.weights = weights;
			this.biases = biases;
			this.input = input;
			this.output = output;
		}

		public void run() {
			int b = getGlobalId(0);
			int i = getGlobalId(1);

			for (int j = 0; j < inputSize; j++)
				output[b][i] += weights[j + inputSize * i] * input[b][j];

			output[b][i] += biases[i];
		}
	}

	/**
	 * The DeltaKernel calculates dL/dx
	 */
	class DeltaKernel extends Kernel {
		private int inputSize, outputSize;
		private double[] weights;
		private double[][] previousDelta, delta;

		DeltaKernel(int inputSize, int outputSize) {
			this.inputSize = inputSize;
			this.outputSize = outputSize;
		}

		void init(double[][] previousDelta, double[] weights, double[][] delta) {
			this.previousDelta = previousDelta;
			this.weights = weights;
			this.delta = delta;
		}

		public void run() {
			int b = getGlobalId(0);
			int i = getGlobalId(1);

			// dL/dx for the current layer
			for (int j = 0; j < outputSize; j++)
				delta[b][i] += previousDelta[b][j] * weights[i + inputSize * j];
		}
	}
}
