package neuralnet.layers;

import neuralnet.activations.Activation;
import neuralnet.activations.ActivationType;
import neuralnet.costs.Cost;
import neuralnet.initializers.HeInitialization;
import neuralnet.initializers.Initializer;
import neuralnet.optimizers.Updater;
import neuralnet.optimizers.UpdaterType;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.stream.IntStream;

/**
 * The convolutional layer revolves around convolutions in image processing. Using a similar method, filters are convolved around an image
 * (the input) and an output is created. A bias term is added, then the output is activated Deltas are calculated a layer ahead
 * (the next layer that will be updated), so that upsampling layers are compatible
 */
public class Convolutional implements Layer {
	private int batchSize;

	// the filter amount is the output depth
	// the filter size is the size of the filters
	private int filterAmount, filterSize;

	private UpdaterType updaterType;
	private Initializer initializer;
	private Activation activation;
	private Updater filterUpdater;
	private Updater biasUpdater;

	// the pad is the amount an input is padded
	// the pad width and pad height are the heights and widths of the input after it is padded
	private int padHeight, padWidth, pad;

	// the stride is how much a filter skips each time
	// the output width and output height are the heights and widths of the output after convolution
	private int outputHeight, outputWidth, stride;

	// the input width and input height are the heights and widths of the inputs provided to the layer
	private int inputHeight, inputWidth, depth;

	private float[] filters, biases;
	private float[] gradient, biasGradient;
	private float[] input, output;

	private Convolutional(int pad, int stride, int filterAmount, int filterSize, Initializer initializer, UpdaterType updaterType,
						  ActivationType activationType) {
		this.pad = pad;
		this.stride = stride;
		this.filterAmount = filterAmount;
		this.filterSize = filterSize;

		this.updaterType = updaterType;
		this.initializer = initializer;

		activation = activationType;
	}

	/**
	 * Initializes a convolutional layer from a file.
	 *
	 * @param dis the input stream
	 */
	Convolutional(DataInputStream dis) throws IOException {
		inputHeight = dis.readInt();
		inputWidth = dis.readInt();
		depth = dis.readInt();
		padHeight = dis.readInt();
		padWidth = dis.readInt();
		pad = dis.readInt();
		outputHeight = dis.readInt();
		outputWidth = dis.readInt();
		stride = dis.readInt();
		filterAmount = dis.readInt();
		filterSize = dis.readInt();

		activation = Activation.fromString(dis);
		updaterType = UpdaterType.fromString(dis);

		filterUpdater = updaterType.create(dis);
		filters = new float[filterAmount * depth * filterSize * filterSize];

		biasUpdater = updaterType.create(dis);
		biases = new float[filterAmount];

		for (int f = 0; f < filterAmount; f++) {
			biases[f] = dis.readFloat();

			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int index = n + filterSize * (m + filterSize * (k + depth * f));

						filters[index] = dis.readFloat();
					}
				}
			}
		}
	}

	public void setDimensions(int... dimensions) {
		if (dimensions.length != 3)
			throw new IllegalArgumentException();

		this.inputHeight = dimensions[0];
		this.inputWidth = dimensions[1];
		this.depth = dimensions[2];

		if (inputHeight <= 0 || inputWidth <= 0 || depth <= 0)
			throw new IllegalArgumentException("Invalid input dimensions.");

		if (pad < 0)
			throw new IllegalArgumentException("Padding must be > 0.");

		// calculating the post padding dimensions
		padHeight = inputHeight + 2 * pad;
		padWidth = inputWidth + 2 * pad;

		if ((padHeight - filterSize) % stride != 0)
			throw new IllegalArgumentException("Invalid output dimensions.");
		if ((padWidth - filterSize) % stride != 0)
			throw new IllegalArgumentException("Invalid output dimensions.");

		// calculating the post convolution dimensions
		this.outputHeight = (padHeight - filterSize) / stride + 1;
		this.outputWidth = (padWidth - filterSize) / stride + 1;

		if (outputHeight <= 0 || outputWidth <= 0 || filterAmount <= 0)
			throw new IllegalArgumentException("Invalid output dimensions.");

		if (filterSize <= 0)
			throw new IllegalArgumentException("Invalid filter dimensions.");

		filters = new float[filterAmount * depth * filterSize * filterSize];
		filterUpdater = updaterType.create(filters.length);

		biases = new float[filterAmount];
		biasUpdater = updaterType.create(biases.length);

		int inputSize = depth * filterSize * filterSize;

		IntStream.range(0, filterAmount).parallel().forEach(f -> {
			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int index = n + filterSize * (m + filterSize * (k + depth * f));

						filters[index] = initializer.initialize(inputSize);
					}
				}
			}
		});
	}

	public void setMode(Mode mode) {

	}

	/**
	 * Pads the input.
	 *
	 * @param input the input
	 * @return the padded input
	 */
	float[] pad(float[] input, int batchSize) {
		if (batchSize <= 0)
			throw new IllegalArgumentException("Batch size must be > 0.");

		if (pad > 0) {
			// creating an array, with the dimensions of the padded input
			float[] out = new float[batchSize * depth * padHeight * padWidth];

			// padding the array
			int position = 0;
			for (int b = 0; b < batchSize; b++) {
				for (int j = 0; j < depth; j++) {
					position += pad * padWidth;

					for (int k = 0; k < inputWidth * inputHeight; k += inputWidth) {
						System.arraycopy(input, k + (inputWidth * inputHeight) * (j + depth * b), out, pad + position, inputWidth);
						position += padWidth;
					}

					position += pad * padWidth;
				}
			}

			return out;
		}

		return input;
	}

	public float[] forward(float[] x, int batchSize) {
		this.batchSize = batchSize;

		input = pad(x, batchSize);
		output = new float[batchSize * filterAmount * outputHeight * outputWidth];

		IntStream.range(0, batchSize).parallel().forEach(b -> {
			for (int f = 0; f < filterAmount; f++) {
				for (int i = 0; i < outputHeight; i++) {
					for (int j = 0; j < outputWidth; j++) {
						// performing strides
						int h = i * stride;
						int w = j * stride;

						// convoluted value is the sum of the filters multiplied against the inputs at a certain position
						float conv = 0;

						for (int k = 0; k < depth; k++) {
							for (int m = 0; m < filterSize; m++) {
								for (int n = 0; n < filterSize; n++) {
									int filterIndex = n + filterSize * (m + filterSize * (k + depth * f));
									int inputIndex = (w + n) + padWidth * ((h + m) + padHeight * (k + depth * b));

									conv += filters[filterIndex] * input[inputIndex];
								}
							}
						}

						// adding biases to shift the activation function
						int activatedIndex = j + outputWidth * (i + outputHeight * (f + filterAmount * b));
						output[activatedIndex] = (conv + biases[f]);
					}
				}
			}
		});

		// activation
		activation.activation(output, batchSize);

		return output;
	}

	public float[] backward(Cost cost, float[] target) {
		float[] previousDelta = cost.derivative(output, target, batchSize);

		return backward(previousDelta);
	}

	public float[] backward(float[] previousDelta) {
		// back propagation on the Convolutional layers are calculated a layer ahead
		float[] delta = new float[batchSize * depth * padHeight * padWidth];

		gradient = new float[filterAmount * depth * filterSize * filterSize];
		biasGradient = new float[filterAmount];

		// derivative
		output = activation.derivative(output);

		IntStream.range(0, batchSize).parallel().forEach(b -> {
			for (int f = 0; f < filterAmount; f++) {
				for (int i = 0, h = 0; i < outputHeight; i++, h += stride) {
					for (int j = 0, w = 0; j < outputWidth; j++, w += stride) {
						int index = j + outputWidth * (i + outputHeight * (f + filterAmount * b));

						// the bias gradient is the delta, since biases are just added to the output
						float d = previousDelta[index] * output[index];
						biasGradient[f] += d;

						for (int k = 0; k < depth; k++) {
							for (int m = 0; m < filterSize; m++) {
								for (int n = 0; n < filterSize; n++) {
									int gradientIndex = n + filterSize * (m + filterSize * (k + depth * f));
									int inputIndex = (w + n) + padWidth * ((h + m) + padHeight * (k + depth * b));

									gradient[gradientIndex] += d * input[inputIndex];
								}
							}
						}
					}
				}
			}
		});

		// calculating delta
		IntStream.range(0, batchSize).parallel().forEach(b -> {
			for (int k = 0; k < depth; k++) {
				for (int i = 0; i < padHeight; i++) {
					for (int j = 0; j < padWidth; j++) {
						int h = i * stride;
						int w = j * stride;
						int deltaIndex = j + padWidth * (i + padHeight * (k + depth * b));

						for (int f = 0; f < filterAmount; f++) {
							for (int m = 0; m < filterSize; m++) {
								for (int n = 0; n < filterSize; n++) {
									if ((w - n) < outputWidth && (h - m) < outputHeight && (w - n) >= 0 && (h - m) >= 0) {
										int upsampledIndex = (w - n) + outputWidth * ((h - m) + outputHeight * f);
										int filterIndex = n + filterSize * (m + filterSize * (k + depth * (f + filterAmount * b)));

										// same as forward propagation, except the activation derivative is multiplied later
										delta[deltaIndex] += previousDelta[upsampledIndex] * filters[filterIndex];
									}
								}
							}
						}
					}
				}
			}
		});

		return delta;
	}

	/**
	 * Update the parameters given gradients.
	 */
	public void update(int size) {
		float[] biasUpdate = biasUpdater.update(biasGradient);
		float[] filterUpdate = filterUpdater.update(gradient);

		IntStream.range(0, filterAmount).parallel().forEach(f -> {
			biases[f] += biasUpdate[f] / size;

			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int filterIndex = n + filterSize * (m + filterSize * (k + depth * f));

						filters[filterIndex] += filterUpdate[filterIndex] / size;
					}
				}
			}
		});
	}

	public float[][][] getParameters() {
		return new float[][][]{{filters, gradient}, {biases, biasGradient}};
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(inputHeight);
		dos.writeInt(inputWidth);
		dos.writeInt(depth);
		dos.writeInt(padHeight);
		dos.writeInt(padWidth);
		dos.writeInt(pad);
		dos.writeInt(outputHeight);
		dos.writeInt(outputWidth);
		dos.writeInt(stride);
		dos.writeInt(filterAmount);
		dos.writeInt(filterSize);

		activation.export(dos);
		updaterType.export(dos);

		filterUpdater.export(dos);
		biasUpdater.export(dos);

		for (int f = 0; f < filterAmount; f++) {
			dos.writeFloat(biases[f]);

			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int index = n + filterSize * (m + filterSize * (k + depth * f));

						dos.writeFloat(filters[index]);
					}
				}
			}
		}
	}

	public int[] getOutputDimensions() {
		return new int[]{outputHeight, outputWidth, filterAmount};
	}

	public LayerType getType() {
		return LayerType.CONVOLUTIONAL;
	}

	/**
	 * Builder for Convolutional layers.
	 */
	@SuppressWarnings({"unused", "WeakerAccess"})
	public static class Builder {
		private int pad;
		private int stride;
		private int filterAmount, filterSize;
		private Initializer initializer;
		private UpdaterType updaterType;
		private ActivationType activationType;

		public Builder() {
			initializer = new HeInitialization();
			updaterType = UpdaterType.ADAM;
			activationType = ActivationType.RELU;
		}

		/**
		 * The pad is the amount of zeroes that are padded around an input. Padding is used to preserve edge features during convolutions,
		 * by allowing filters to traverse such areas.
		 *
		 * @param pad the pad
		 */
		public Builder pad(int pad) {
			this.pad = pad;

			return this;
		}

		/**
		 * The stride is the amount a filter moves by, each time, when performing convolution. Strides can be used to downsample images,
		 * instead
		 * of using pooling
		 *
		 * @param stride the stride
		 */
		public Builder stride(int stride) {
			this.stride = stride;

			return this;
		}

		/**
		 * The filter amount is the amount of kernels that can be learned and applied during convolution.
		 *
		 * @param filterAmount the filter amount
		 */
		public Builder filterAmount(int filterAmount) {
			this.filterAmount = filterAmount;

			return this;
		}

		/**
		 * The filter size is the size of filters. Larger filter sizes decrease output dimensions faster.
		 *
		 * @param filterSize the filter size
		 */
		public Builder filterSize(int filterSize) {
			this.filterSize = filterSize;

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

		/**
		 * The updater updates parameters.
		 *
		 * @param updaterType the updater type
		 */
		public Builder updaterType(UpdaterType updaterType) {
			this.updaterType = updaterType;

			return this;
		}

		/**
		 * The activation simulates a neuron firing.
		 *
		 * @param activationType the activation type
		 */
		public Builder activationType(ActivationType activationType) {
			this.activationType = activationType;

			return this;
		}

		public Convolutional build() {
			return new Convolutional(pad, stride, filterAmount, filterSize, initializer, updaterType, activationType);
		}
	}
}