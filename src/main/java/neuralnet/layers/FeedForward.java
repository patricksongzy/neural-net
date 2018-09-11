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

	private int batchSize;
	private int inputSize, outputSize;
	private float temperature;
	private Initializer initializer;
	private UpdaterType updaterType;
	private Activation activation;
	private Updater weightUpdater;
	private Updater biasUpdater;
	private float[] weights, biases;
	private float[] gradient, biasGradient;
	private float[] input, output;

	/**
	 * Initializes a FeedForward layer neural network from a file.
	 *
	 * @param dis the input stream
	 */
	FeedForward(DataInputStream dis) throws IOException {
		inputSize = dis.readInt();
		outputSize = dis.readInt();
		temperature = dis.readFloat();

		activation = Activation.fromString(dis);
		updaterType = UpdaterType.fromString(dis);

		weights = new float[outputSize * inputSize];
		weightUpdater = updaterType.create(dis);

		biases = new float[outputSize];
		biasUpdater = updaterType.create(dis);

		gradient = new float[outputSize * inputSize];
		biasGradient = new float[outputSize];

		for (int i = 0; i < outputSize; i++) {
			biases[i] = dis.readFloat();

			for (int j = 0; j < inputSize; j++) {
				int index = j + inputSize * i;

				weights[index] = dis.readFloat();
			}
		}
	}

	private FeedForward(int outputSize, float temperature, Initializer initializer,
						UpdaterType updaterType, Activation activation) {
		if (outputSize <= 0)
			throw new IllegalArgumentException("Output size must be > 0.");
		if (temperature <= 0)
			throw new IllegalArgumentException("Temperature must be > 0.");
		if (initializer == null || updaterType == null || activation == null)
			throw new IllegalArgumentException("Values cannot be null.");

		this.outputSize = outputSize;
		this.temperature = temperature;
		this.initializer = initializer;
		this.updaterType = updaterType;
		this.activation = activation;
	}

	public void setMode(Mode mode) {
		this.mode = mode;
	}

	public void setDimensions(int... dimensions) {
		inputSize = dimensions[0];
		for (int i = 1; i < dimensions.length; i++)
			inputSize *= dimensions[i];

		if (inputSize <= 0)
			throw new IllegalArgumentException("Invalid input dimensions.");

		weights = new float[outputSize * inputSize];
		weightUpdater = updaterType.create(weights.length);

		biases = new float[outputSize];
		biasUpdater = updaterType.create(biases.length);

		gradient = new float[outputSize * inputSize];
		biasGradient = new float[outputSize];

		for (int i = 0; i < outputSize; i++) {
			for (int j = 0; j < inputSize; j++) {
				int index = j + inputSize * i;

				weights[index] = initializer.initialize(inputSize);
			}
		}
	}

	public float[][][] getParameters() {
		return new float[][][]{{weights, gradient}, {biases, biasGradient}};
	}

	public float[] forward(float[] input, int batchSize) {
		this.batchSize = batchSize;
		this.input = input;

		output = new float[batchSize * outputSize];

		for (int b = 0; b < batchSize; b++)
			System.arraycopy(biases, 0, output, b * outputSize, outputSize);

		output = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, batchSize,
			outputSize, inputSize, input, inputSize, weights, inputSize, output, outputSize);

		if (mode == Mode.EVAL && temperature != 1) {
			for (int i = 0; i < batchSize; i++) {
				output[i] /= temperature;
			}
		}

		// activating output
		activation.activation(output, batchSize);

		return output;
	}

	public float[] backward(Cost cost, float[] target) {
		float[] previousDelta;

		if (activation.getType() == Activation.Type.SOFTMAX)
			previousDelta = cost.derviativeSoftmax(output, target, batchSize);
		else
			previousDelta = cost.derivative(output, target, batchSize);

		return backward(previousDelta);
	}

	public float[] backward(float[] previousDelta) {
		output = activation.derivative(output);

		IntStream.range(0, batchSize).parallel().forEach(b -> {
			for (int i = 0; i < outputSize; i++) {
				int index = i + outputSize * b;

				if (activation.getType() != Activation.Type.SOFTMAX)
					previousDelta[index] *= output[index];

				biasGradient[i] += previousDelta[index];
			}
		});

		gradient = GPU.sgemm(CLBlastTranspose.CLBlastTransposeYes, CLBlastTranspose.CLBlastTransposeNo, outputSize,
			inputSize, batchSize, previousDelta, outputSize, input, inputSize, gradient, inputSize);

		float[] delta = new float[batchSize * inputSize];

		delta = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, batchSize,
			inputSize, outputSize, previousDelta, outputSize, weights, inputSize, delta, inputSize);

		return delta;
	}

	/**
	 * Updates parameters given gradients.
	 */
	public void update(int size) {
		float[] update = weightUpdater.update(gradient);
		float[] biasUpdate = biasUpdater.update(biasGradient);

		weights = GPU.saxpy(weights.length, 1.0f / size, update, weights);
		biases = GPU.saxpy(biases.length, 1.0f / size, biasUpdate, biases);

		gradient = new float[outputSize * inputSize];
		biasGradient = new float[outputSize];
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(inputSize);
		dos.writeInt(outputSize);
		dos.writeFloat(temperature);

		activation.export(dos);
		updaterType.export(dos);

		weightUpdater.export(dos);
		biasUpdater.export(dos);

		for (int i = 0; i < outputSize; i++) {
			dos.writeFloat(biases[i]);

			for (int j = 0; j < inputSize; j++) {
				int index = j + inputSize * i;

				dos.writeFloat(weights[index]);
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
		private Activation activation;
		private Initializer initializer;

		public Builder() {
			temperature = 1;
			updaterType = UpdaterType.ADAM;
			activation = ActivationType.RELU;
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
		public Builder temperature(float temperature) {
			this.temperature = temperature;

			return this;
		}

		/**
		 * The updater updates parameters.
		 *
		 * @param updaterType the updater
		 */
		public Builder updaterType(UpdaterType updaterType) {
			this.updaterType = updaterType;

			return this;
		}

		/**
		 * The activation simulates neurons firing.
		 *
		 * @param activation the activation type
		 */
		public Builder activation(Activation activation) {
			this.activation = activation;

			return this;
		}

		/**
		 * The initializer initializes weights.
		 *
		 * @param initializer the initializer
		 */
		public Builder initializer(Initializer initializer) {
			this.initializer = initializer;

			return this;
		}

		public FeedForward build() {
			return new FeedForward(outputSize, temperature, initializer, updaterType, activation);
		}
	}
}
