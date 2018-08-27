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
	private Mode mode = Mode.TRAIN;

	private int batchSize;

	// the filter amount is the output depth
	// the filter size is the size of the filters
	private int filterAmount, filterSize;

	private UpdaterType updaterType;
	private Initializer initializer;
	private Activation activation;
	private Updater[] filterUpdaters;
	private Updater[] biasUpdaters;

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
	private float[][] input, output;

	private Convolutional(int pad, int stride, int filterAmount, int filterSize, Initializer initializer, UpdaterType updaterType,
						  ActivationType activationType) {
		this.pad = pad;
		this.stride = stride;
		this.filterAmount = filterAmount;
		this.filterSize = filterSize;

		this.updaterType = updaterType;
		this.initializer = initializer;

		activation = activationType.create();
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

		filterUpdaters = new Updater[filterAmount * depth * filterSize * filterSize];
		filters = new float[filterAmount * depth * filterSize * filterSize];

		biasUpdaters = new Updater[filterAmount];
		biases = new float[filterAmount];

		activation = ActivationType.fromString(dis).create();
		updaterType = UpdaterType.fromString(dis);

		for (int f = 0; f < filterAmount; f++) {
			biases[f] = dis.readFloat();
			biasUpdaters[f] = updaterType.create(dis);

			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int index = n + filterSize * (m + filterSize * (k + depth * f));

						filters[index] = dis.readFloat();
						filterUpdaters[index] = updaterType.create(dis);
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
			throw new IllegalArgumentException();

		// calculating the post padding dimensions
		padHeight = inputHeight + 2 * pad;
		padWidth = inputWidth + 2 * pad;

		// calculating the post convolution dimensions
		this.outputHeight = (padHeight - filterSize) / stride + 1;
		this.outputWidth = (padWidth - filterSize) / stride + 1;

		filterUpdaters = new Updater[filterAmount * depth * filterSize * filterSize];
		filters = new float[filterAmount * depth * filterSize * filterSize];

		biasUpdaters = new Updater[filterAmount];
		biases = new float[filterAmount];

		int inputSize = depth * filterSize * filterSize;

		IntStream.range(0, filterAmount).parallel().forEach(f -> {
			biasUpdaters[f] = updaterType.create();
			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int index = n + filterSize * (m + filterSize * (k + depth * f));

						filters[index] = initializer.initialize(inputSize);
						filterUpdaters[index] = updaterType.create();
					}
				}
			}
		});
	}

	public void setMode(Mode mode) {
		this.mode = mode;
	}

	/**
	 * Pads the input.
	 *
	 * @param input the input
	 * @return the padded input
	 */
	float[] pad(float[] input, int batchSize) {
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

	public float[][] forward(float[][] x, int batchSize) {
		this.batchSize = batchSize;
		input = new float[x.length][];

		for (int t = 0; t < x.length; t++) {
			input[t] = pad(x[t], batchSize);

			output = new float[x.length][batchSize * filterAmount * outputHeight * outputWidth];

			int time = t;
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

										conv += filters[filterIndex] * input[time][inputIndex];
									}
								}
							}

							// adding biases to shift the activation function
							int activatedIndex = j + outputWidth * (i + outputHeight * (f + filterAmount * b));
							output[time][activatedIndex] = (conv + biases[f]);
						}
					}
				}
			});

			// activation
			activation.activation(output[t], batchSize);
		}

		return output;
	}

	public float[][] backward(Cost cost, float[][] target) {
		float[][] previousDelta = new float[output.length][];

		for (int t = 0; t < output.length; t++) {
			// back propagation on the Convolutional layers are calculated a layer ahead
			previousDelta[t] = cost.derivative(output[t], target[t], activation, batchSize);
		}

		return backward(previousDelta);
	}

	public float[][] backward(float[][] previousDelta) {
		// back propagation on the Convolutional layers are calculated a layer ahead
		float[][] delta = new float[output.length][batchSize * depth * padHeight * padWidth];

		gradient = new float[filterAmount * depth * filterSize * filterSize];
		biasGradient = new float[filterAmount];

		for (int t = 0; t < output.length; t++) {
			final int time = t;

			// derivative
			output[t] = activation.derivative(output[t]);

			IntStream.range(0, batchSize).parallel().forEach(b -> {
				int size = output.length / batchSize;
				if (activation.getType() != ActivationType.SOFTMAX) {
					for (int i = 0; i < size; i++) {
						previousDelta[time][i + size * b] *= output[time][i + size * b];
					}
				}

				for (int f = 0; f < filterAmount; f++) {
					for (int i = 0, h = 0; i < outputHeight; i++, h += stride) {
						for (int j = 0, w = 0; j < outputWidth; j++, w += stride) {
							int index = j + outputWidth * (i + outputHeight * (f + filterAmount * b));

							// the bias gradient is the delta, since biases are just added to the output
							float d = previousDelta[time][index];
							biasGradient[f] += d;

							for (int k = 0; k < depth; k++) {
								for (int m = 0; m < filterSize; m++) {
									for (int n = 0; n < filterSize; n++) {
										int gradientIndex = n + filterSize * (m + filterSize * (k + depth * f));
										int inputIndex = (w + n) + padWidth * ((h + m) + padHeight * (k + depth * b));

										// the gradient is the delta multiplied against the input, since the filters are multiplied with
										// the
										// input
										gradient[gradientIndex] += d * input[time][inputIndex];
									}
								}
							}
						}
					}
				}
			});

			// calculating gradient
			if (mode != Mode.GRADIENT_CHECK) {
				// updating parameters
				update(biasGradient, gradient);
			}

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
											delta[time][deltaIndex] += previousDelta[time][upsampledIndex] * filters[filterIndex];
										}
									}
								}
							}
						}
					}
				}
			});
		}

		return delta;
	}

	/**
	 * Update the parameters given gradients.
	 *
	 * @param delta    the bias gradient
	 * @param gradient the weight gradient
	 */
	private void update(float[] delta, float[] gradient) {
		IntStream.range(0, filterAmount).parallel().forEach(f -> {
			biases[f] += biasUpdaters[f].update(delta[f] / (batchSize * output.length));

			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int filterIndex = n + filterSize * (m + filterSize * (k + depth * f));

						filters[filterIndex] += filterUpdaters[filterIndex].update(gradient[filterIndex] / (batchSize * output.length));
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

		activation.getType().export(dos);
		updaterType.export(dos);

		for (int f = 0; f < filterAmount; f++) {
			dos.writeFloat(biases[f]);
			biasUpdaters[f].export(dos);

			for (int k = 0; k < depth; k++) {
				for (int m = 0; m < filterSize; m++) {
					for (int n = 0; n < filterSize; n++) {
						int index = n + filterSize * (m + filterSize * (k + depth * f));

						dos.writeFloat(filters[index]);
						filterUpdaters[index].export(dos);
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
			if (initializer != null)
				this.initializer = initializer;
			else
				throw new IllegalArgumentException();

			return this;
		}

		/**
		 * The updater updates parameters.
		 *
		 * @param updaterType the updater type
		 */
		public Builder updaterType(UpdaterType updaterType) {
			if (updaterType != null)
				this.updaterType = updaterType;
			else
				throw new IllegalArgumentException();

			return this;
		}

		/**
		 * The activation simulates a neuron firing.
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

		public Convolutional build() {
			if (pad >= 0 && stride > 0 && filterAmount > 0 && filterSize > 0)
				return new Convolutional(pad, stride, filterAmount, filterSize, initializer, updaterType, activationType);

			throw new IllegalArgumentException();
		}
	}
}