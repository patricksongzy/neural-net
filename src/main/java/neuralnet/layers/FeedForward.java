package neuralnet.layers;

import neuralnet.GPU;
import neuralnet.activations.Activation;
import neuralnet.activations.ActivationType;
import neuralnet.costs.Cost;
import neuralnet.initializers.HeInitialization;
import neuralnet.initializers.Initializer;
import neuralnet.optimizers.Updater;
import neuralnet.optimizers.UpdaterType;
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
	private float[] input;
	private float[][] output;

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
		input = flatten(x);
		output = new float[x.length][outputSize];

		float[] flattened = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, x.length,
			outputSize, inputSize, input, inputSize, weights, inputSize, biases, outputSize);

		if (mode == Mode.EVAL && temperature != 1)
			IntStream.range(0, flattened.length).parallel().forEach(i -> flattened[i] /= temperature);

		output = unflatten(flattened, x.length);

		// activating output
		activation.activation(output);

		return output;
	}

	private float[] flatten(float[][] x) {
		int size = x[0].length;
		float[] flat = new float[x.length * size];

		IntStream.range(0, x.length).parallel().forEach(b -> System.arraycopy(x[b], 0, flat, b * size, size));

		return flat;
	}

	private float[][] unflatten(float[] x, int batchSize) {
		int size = x.length / batchSize;
		float[][] unflattened = new float[batchSize][size];

		IntStream.range(0, batchSize).parallel().forEach(b -> System.arraycopy(x, b * size, unflattened[b], 0, size));

		return unflattened;
	}

	public float[] backward(Cost cost, float[][] target) {
		float[] previousDelta = cost.derivative(output, target, activation);

		biasGradient = new float[outputSize];

		for (int b = 0; b < output.length; b++) {
			for (int i = 0; i < outputSize; i++) {
				biasGradient[i] += previousDelta[i + outputSize * b];
			}
		}

		getGradient(previousDelta);
		return getDelta(previousDelta);
	}

	public float[] backward(float[] previousDelta) {
		output = activation.derivative(output);
		biasGradient = new float[outputSize];

		for (int b = 0; b < output.length; b++) {
			for (int i = 0; i < outputSize; i++) {
				previousDelta[i + outputSize * b] *= output[b][i];
				biasGradient[i] += previousDelta[i + outputSize * b];
			}
		}

		// calculating gradient
		getGradient(previousDelta);
		return getDelta(previousDelta);
	}

	private float[] getDelta(float[] previousDelta) {
		float[] delta = new float[output.length * inputSize];

		delta = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, output.length,
			inputSize, outputSize, previousDelta, outputSize, weights, inputSize, delta, inputSize);

		return delta;
	}

	/**
	 * Calculates the gradient, then stores it, if on gradient check mode. Else, updates the parameters.
	 */
	private void getGradient(float[] delta) {
		gradient = new float[outputSize * inputSize];

		gradient = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, outputSize,
			inputSize, output.length, delta, output.length, input, inputSize, gradient, inputSize);

		if (mode != Mode.GRADIENT_CHECK) {
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
	@SuppressWarnings({"unused", "WeakerAccess"})
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
