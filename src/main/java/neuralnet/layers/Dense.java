package neuralnet.layers;

import neuralnet.GPU;
import neuralnet.activations.Activation;
import neuralnet.costs.Cost;
import neuralnet.initializers.Initializer;
import neuralnet.optimizers.Updater;
import neuralnet.optimizers.UpdaterType;
import org.jocl.CL;
import org.jocl.blast.CLBlastTranspose;
import org.jocl.cl_mem;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Dense layers have a weight for each input. The inputs are multiplied to the weights, summed, then a bias term is added. The
 * output is
 * then run through an activation function.
 */
public class Dense implements Layer {
	private Mode mode;

	private int batchSize;
	private int inputSize, outputSize;
	private float temperature;
	private Initializer initializer;
	private Activation activation;
	private Updater weightUpdater;
	private Updater biasUpdater;
	private float[] weights, biases;
	private float[] gradient, biasGradient;
	private float[] output;
	private cl_mem inputBuffer, weightBuffer;

	/**
	 * Initializes a Dense layer neural network from a file.
	 *
	 * @param dis the input stream
	 */
	Dense(DataInputStream dis, UpdaterType updaterType) throws IOException {
		System.out.println("Type: " + getType());

		inputSize = dis.readInt();
		outputSize = dis.readInt();
		temperature = dis.readFloat();

		System.out.println("Input Size: " + inputSize);
		System.out.println("Output Size: " + outputSize);
		System.out.println("Temperature: " + temperature);

		activation = Activation.fromString(dis);
		System.out.println("Activation: " + activation.getType());

		System.out.println("Importing weights.");
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

		System.out.println("Done importing weights.");
	}

	private Dense(int outputSize, float temperature, Initializer initializer, Activation activation) {
		if (initializer == null || activation == null)
			throw new IllegalArgumentException("Values cannot be null.");

		if (activation.getType() != Activation.Type.SOFTMAX && temperature != 1) {
			System.err.println("WARNING: Temperature is usually used with softmax.");

			for (StackTraceElement element : Thread.currentThread().getStackTrace()) {
				System.err.println(element);
			}
		}

		this.outputSize = outputSize;
		this.temperature = temperature;
		this.initializer = initializer;
		this.activation = activation;
	}

	public void setMode(Mode mode) {
		this.mode = mode;
	}

	public void setDimensions(int[] dimensions, UpdaterType updaterType) {
		System.out.println("Type: " + getType());

		inputSize = dimensions[0];
		for (int i = 1; i < dimensions.length; i++)
			inputSize *= dimensions[i];

		System.out.println("Input Size: " + inputSize);
		System.out.println("Output Size: " + outputSize);
		System.out.println("Temperature: " + temperature);

		if (inputSize <= 0)
			throw new IllegalArgumentException("Invalid input dimensions.");
		if (outputSize <= 0)
			throw new IllegalArgumentException("Output size must be > 0.");
		if (temperature <= 0)
			throw new IllegalArgumentException("Temperature must be > 0.");

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

		output = new float[batchSize * outputSize];

		for (int b = 0; b < batchSize; b++)
			System.arraycopy(biases, 0, output, b * outputSize, outputSize);

		inputBuffer = GPU.gpuAlloc(CL.CL_MEM_READ_ONLY, input.length, input);
		weightBuffer = GPU.gpuAlloc(CL.CL_MEM_READ_ONLY, weights.length, weights);

		output = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeYes, batchSize,
			outputSize, inputSize, inputBuffer, inputSize, weightBuffer, inputSize, output, outputSize);

		if (mode == Mode.EVAL) {
			if (temperature != 1) {
				for (int i = 0; i < batchSize; i++) {
					output[i] /= temperature;
				}
			}

			CL.clReleaseMemObject(inputBuffer);
			CL.clReleaseMemObject(weightBuffer);
		}

		// activating output
		activation.activation(output, batchSize);

		return output;
	}

	public float[] backward(Cost cost, float[] target, boolean calculateDelta) {
		float[] previousDelta;

		if (activation.getType() == Activation.Type.SOFTMAX)
			previousDelta = cost.derviativeSoftmax(output, target, batchSize);
		else
			previousDelta = cost.derivative(output, target, batchSize);

		return backward(previousDelta, calculateDelta);
	}

	public float[] backward(float[] previousDelta, boolean calculateDelta) {
		output = activation.derivative(output);

		for (int b = 0; b < batchSize; b++) {
			for (int i = 0; i < outputSize; i++) {
				int index = i + outputSize * b;

				if (activation.getType() != Activation.Type.SOFTMAX)
					previousDelta[index] *= output[index];

				biasGradient[i] += previousDelta[index];
			}
		}

		cl_mem deltaBuffer = GPU.gpuAlloc(CL.CL_MEM_READ_ONLY, previousDelta.length, previousDelta);

		gradient = GPU.sgemm(CLBlastTranspose.CLBlastTransposeYes, CLBlastTranspose.CLBlastTransposeNo, outputSize,
			inputSize, batchSize, deltaBuffer, outputSize, inputBuffer, inputSize, gradient, inputSize);

		CL.clReleaseMemObject(inputBuffer);

		float[] delta = null;
		if (calculateDelta)
			delta = GPU.sgemm(CLBlastTranspose.CLBlastTransposeNo, CLBlastTranspose.CLBlastTransposeNo, batchSize,
				inputSize, outputSize, deltaBuffer, outputSize, weightBuffer, inputSize, new float[batchSize * inputSize], inputSize);

		CL.clReleaseMemObject(deltaBuffer);
		CL.clReleaseMemObject(weightBuffer);

		return delta;
	}

	public void update(int scale) {
		weightUpdater.update(weights, gradient, scale);
		biasUpdater.update(biases, biasGradient, scale);

		gradient = new float[outputSize * inputSize];
		biasGradient = new float[outputSize];
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(inputSize);
		dos.writeInt(outputSize);
		dos.writeFloat(temperature);

		activation.export(dos);

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
	 * Builder for Dense layers.
	 */
	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private int outputSize;
		private float temperature;
		private Activation activation;
		private Initializer initializer;

		public Builder() {
			temperature = 1;
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

		public Dense build() {
			return new Dense(outputSize, temperature, initializer, activation);
		}
	}
}
