package main.neuralnet.layers;

import main.GPU;
import main.neuralnet.activations.Activation;
import main.neuralnet.activations.ActivationType;
import main.neuralnet.costs.Cost;
import main.neuralnet.initializers.HeInitialization;
import main.neuralnet.initializers.Initializer;
import main.neuralnet.optimizers.Updater;
import main.neuralnet.optimizers.UpdaterType;
import org.jocl.blast.CLBlastTranspose;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.stream.IntStream;

/**
 * FeedForward layers have a weight for each input. The inputs are multiplied to the weights, summed, then a bias term is added. The
 * output is
 * then run through an activation function.
 */
public class FeedForward implements Layer {
	private Mode mode = Mode.TRAIN;

	private int inputSize, outputSize;
	private float temperature;
	private Initializer initializer;
	private UpdaterType updaterType;
	private Activation activation;
	private Updater[] weightUpdaters;
	private Updater[] biasUpdaters;
	private float[] weights, biases;
	private float[] gradient, biasGradient;
	private float[][] input, output;

	/**
	 * Initializes a FeedForward layer neural network from a file.
	 *
	 * @param dis the input stream
	 */
	FeedForward(DataInputStream dis) throws IOException {
		inputSize = dis.readInt();
		outputSize = dis.readInt();
		temperature = dis.readFloat();

		activation = ActivationType.fromString(dis).create();
		updaterType = UpdaterType.fromString(dis);

		biases = new float[outputSize];
		biasUpdaters = new Updater[outputSize];
		weights = new float[outputSize * inputSize];
		weightUpdaters = new Updater[outputSize * inputSize];

		for (int i = 0; i < outputSize; i++) {
			biases[i] = dis.readFloat();
			biasUpdaters[i] = updaterType.create(dis);

			for (int j = 0; j < inputSize; j++) {
				int index = j + inputSize * i;

				weights[index] = dis.readFloat();
				weightUpdaters[index] = updaterType.create(dis);
			}
		}
	}

	private FeedForward(int outputSize, float temperature, Initializer initializer, UpdaterType updaterType,
						ActivationType activationType) {
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

		biases = new float[outputSize];
		biasUpdaters = new Updater[outputSize];
		weights = new float[outputSize * inputSize];
		weightUpdaters = new Updater[outputSize * inputSize];

		for (int i = 0; i < outputSize; i++) {
			biasUpdaters[i] = updaterType.create();

			for (int j = 0; j < inputSize; j++) {
				int index = j + inputSize * i;

				weights[index] = initializer.initialize(inputSize);
				weightUpdaters[index] = updaterType.create();
			}
		}
	}

	public float[][][] getParameters() {
		return new float[][][]{{weights, gradient}, {biases, biasGradient}};
	}

	public float[][] forward(float[][] x) {
		input = x;
		output = new float[x.length][outputSize];

		for (int b = 0; b < x.length; b++) {
			output[b] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, 1,
				outputSize, inputSize, input[b], inputSize, weights, inputSize, biases, outputSize);
		}

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

	public float[][] backward(Cost cost, float[][] target) {
		float[][] previousDelta = cost.derivative(output, target, activation);

		biasGradient = new float[outputSize];

		for (int i = 0; i < outputSize; i++) {
			for (float[] d : previousDelta) {
				biasGradient[i] += d[i];
			}
		}

		getGradient(previousDelta);

		return getDelta(previousDelta);
	}

	public float[][] backward(float[][] previousDelta) {
		output = activation.derivative(output);
		biasGradient = new float[outputSize];

		for (int i = 0; i < outputSize; i++) {
			for (int b = 0; b < previousDelta.length; b++) {
				previousDelta[b][i] *= output[b][i];
			}

			for (float[] d : previousDelta) {
				biasGradient[i] += d[i];
			}
		}

		// calculating gradient
		getGradient(previousDelta);

		return getDelta(previousDelta);
	}

	private float[][] getDelta(float[][] previousDelta) {
		float[][] delta = new float[output.length][inputSize];

		for (int b = 0; b < output.length; b++) {
			delta[b] = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, 1,
				inputSize, outputSize, previousDelta[b], outputSize, weights, inputSize, new float[outputSize], outputSize);
		}

		return delta;
	}

	/**
	 * Calculates the gradient, then stores it, if on gradient check mode. Else, updates the parameters.
	 */
	private void getGradient(float[][] delta) {
		gradient = new float[outputSize * inputSize];

		for (int b = 0; b < output.length; b++) {
			gradient = GPU.sger(outputSize, inputSize, delta[b], input[b], gradient, inputSize);
		}

		if (mode == Mode.TRAIN) {
			// updating parameters
			update(gradient);
		}
	}

	/**
	 * Updates parameters given gradients.
	 *
	 * @param gradient the weight gradient
	 */
	private void update(float[] gradient) {
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
		dos.writeFloat(temperature);

		activation.getType().export(dos);
		updaterType.export(dos);

		for (int i = 0; i < outputSize; i++) {
			dos.writeFloat(biases[i]);
			biasUpdaters[i].export(dos);

			for (int j = 0; j < inputSize; j++) {
				int index = j + inputSize * i;

				dos.writeFloat(weights[index]);
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
	@SuppressWarnings("unused")
	public static class Builder {
		private int outputSize;
		private float temperature;
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
		public void setTemperature(float temperature) {
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
}
