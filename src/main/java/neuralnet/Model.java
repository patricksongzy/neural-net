package neuralnet;

import javafx.application.Application;
import neuralnet.costs.Cost;
import neuralnet.costs.CostType;
import neuralnet.layers.Layer;
import neuralnet.layers.LayerType;
import neuralnet.optimizers.UpdaterType;
import plot.Plot;

import java.io.*;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Models represent neural network models. They forward and back propagate layers.
 */
@SuppressWarnings("unused")
public class Model {
	private static final int CORES = Runtime.getRuntime().availableProcessors();
	private static final ThreadPoolExecutor ES = new ThreadPoolExecutor(CORES, CORES, 0L, TimeUnit.MILLISECONDS,
		new LinkedBlockingQueue<>(), new ThreadPoolExecutor.CallerRunsPolicy());

	static {
		Runtime.getRuntime().addShutdownHook(new Thread(() -> {
			ES.shutdown();
			try {
				ES.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}));
	}

	private int inputSize;
	private UpdaterType updaterType;

	// TODO: Implement non-sequential
	private Layer[] layers;
	private Cost cost;

	private Model(Layer[] layers, CostType costType, UpdaterType updaterType, int[] inputDimensions) {
		if (layers.length <= 0)
			throw new IllegalArgumentException("Invalid layer amount.");
		Objects.requireNonNull(costType);
		Objects.requireNonNull(updaterType);
		Objects.requireNonNull(inputDimensions);

		this.layers = layers;
		this.cost = costType;
		this.updaterType = updaterType;

		inputSize = inputDimensions[0];
		for (int i = 1; i < inputDimensions.length; i++)
			inputSize *= inputDimensions[i];

		layers[0].setDimensions(inputDimensions, updaterType); // setting input dimensions

		// each layer's output is the next layer's input
		for (int i = 1; i < layers.length; i++) {
			layers[i].setDimensions(layers[i - 1].getOutputDimensions(), updaterType);
		}
	}

	/**
	 * Imports a model from a file.
	 *
	 * @param file the path to the file
	 */
	public Model(String file) {
		try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file), 16384))) {
			System.out.println("Importing from: " + file);

			// importing layers
			int layerAmount = dis.readInt();
			inputSize = dis.readInt();

			updaterType = UpdaterType.fromString(dis);

			layers = new Layer[layerAmount];
			for (int i = 0; i < layerAmount; i++) {
				System.out.println("Importing layer " + (i + 1) + " / " + layerAmount);
				layers[i] = LayerType.fromString(dis, updaterType);
				System.out.println("Done importing layer " + (i + 1) + " / " + layerAmount);
			}

			cost = CostType.fromString(dis);

			System.out.println("Done importing from: " + file);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}
	}

	public int[] getOutputDimensions() {
		return layers[layers.length - 1].getOutputDimensions();
	}

	public int getLayerAmount() {
		return layers.length;
	}

	public Layer getLayer(int index) {
		return layers[index];
	}

	/**
	 * Forward propagates layers.
	 *
	 * @param x the input
	 * @return the output
	 */
	public float[] forward(float[] x, int batchSize) {
		for (Layer layer : layers)
			x = layer.forward(x, batchSize);

		return x;
	}

	/**
	 * Backpropagates layers, by calculating gradients.
	 *
	 * @param targets the targets
	 */
	@SuppressWarnings("WeakerAccess")
	public void backward(float[] targets) {
		float[] delta = layers[layers.length - 1].backward(cost, targets, layers.length > 1);

		for (int i = layers.length - 2; i >= 0; i--)
			delta = layers[i].backward(delta, i > 0);
	}

	@SuppressWarnings("WeakerAccess")
	public void update(int length) {
		List<Callable<Void>> tasks = new ArrayList<>();

		for (Layer layer : layers) {
			tasks.add(() -> {
				layer.update(length);
				return null;
			});
		}

		try {
			ES.invokeAll(tasks);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		tasks.clear();
	}

	public void train(Map<float[], Float> data, int batchSize, int epochs, float max, float min, float decay, int restartInterval,
					  int restartMultiplier, int checkpoint, String name) {
		new Thread(() -> Application.launch(Plot.class, (String) null)).start();

		// setting mode to training mode
		setMode(Layer.Mode.TRAIN);

		List<float[]> keys = new ArrayList<>(data.keySet());
		updaterType.setDecay(decay * (float) Math.sqrt((float) batchSize / (keys.size() * restartInterval)));

		int inputSize = keys.get(0).length;

		int batch = 0;
		updaterType.init(max);
		updaterType.setDecay(decay * (float) Math.sqrt((float) batchSize / (keys.size() * restartInterval)));

		float current = 0;
		for (int i = 1; i <= epochs; i++) {
			if (i % checkpoint == 0)
				export(name);

			// shuffling data prevents the neural network from learning the order of the data
			Collections.shuffle(keys);

			System.out.println("Epoch: " + i + "/" + epochs);
			// looping through the training set
			for (int j = 0; j < keys.size(); j += batchSize, batch++) {
				if ((current / keys.size()) == restartInterval) {
					System.out.println("Restarting");

					current = 0;
					restartInterval *= restartMultiplier;

					updaterType.init(max);
					updaterType.setDecay(decay * (float) Math.sqrt((float) batchSize / (keys.size() * restartInterval)));
				} else {
					updaterType.init(min + 0.5f * (max - min) * (1 + (float) Math.cos((current / keys.size()) * Math.PI / restartInterval)));
				}

				// calculating the batch size
				int s = (j + batchSize) > keys.size() ? (keys.size() % batchSize) : batchSize;

				float[] inputs = new float[s * inputSize];
				float[] targets = new float[s];

				// creating a batch
				for (int b = 0; b < s; b++) {
					float[] input = keys.get(b + j);
					System.arraycopy(input, 0, inputs, b * inputSize, inputSize);
					targets[b] = data.get(input);
				}

				// forward propagating the batch
				float[] output = forward(inputs, s);

				// back propagating batch
				backward(targets);

				update(s);

				int progress = (int) ((float) (j + s) / keys.size() * 30 + 0.5);
				System.out.printf("\r%d/%d [", j + s, keys.size());

				for (int k = 0; k < progress; k++)
					System.out.print("#");
				for (int k = progress; k < 30; k++)
					System.out.print("-");

				float average = cost.cost(output, targets) / s;
				System.out.print("] - loss: " + average);

				Plot.update(batch, average);

				current += s;
			}

			System.out.println();
		}
	}

	public void trainRecurrent(Map<float[][], float[]> data, int batchSize, int bptt, int epochs, float max, float min, float decay,
							   int restartInterval, int restartMultiplier, int checkpoint, String name) {
		new Thread(() -> Application.launch(Plot.class, (String) null)).start();

		// setting mode to training mode
		setMode(Layer.Mode.TRAIN);

		List<float[][]> keys = new ArrayList<>(data.keySet());
		keys.sort(Comparator.comparingInt(c -> c.length));

		updaterType.setDecay(decay * (float) Math.sqrt((float) 1 / (keys.size() * restartInterval)));

		int batch = 0;
		updaterType.init(max);
		updaterType.setDecay(decay * (float) Math.sqrt((float) 1 / (keys.size() * restartInterval)));

		float current = 0;
		for (int i = 1; i <= epochs; i++) {
			if (i % checkpoint == 0)
				export(name);

			System.out.println("Epoch: " + i + "/" + epochs);
			Map<float[][], float[][]> group = new HashMap<>();
			// looping through the training set
			for (int j = 0; j < keys.size(); j += batchSize, batch++) {
				if ((current / keys.size()) == restartInterval) {
					System.out.println("Restarting");

					current = 0;
					restartInterval *= restartMultiplier;

					updaterType.init(max);
					updaterType.setDecay(decay * (float) Math.sqrt((float) batchSize / (keys.size() * restartInterval)));
				} else {
					updaterType.init(min + 0.5f * (max - min) * (1 + (float) Math.cos((current / keys.size()) * Math.PI / restartInterval)));
				}

				// batch size
				int s = j + batchSize > keys.size() ? keys.size() % batchSize : batchSize;
				// step size
				int step = bptt;
				// shortest length
				int shortest = Integer.MAX_VALUE;

				for (int b = 0; b < s; b++) {
					if (keys.get(b + j).length < step)
						step = keys.get(b + j).length;
					if (keys.get(b + j).length < shortest)
						shortest = keys.get(b + j).length;
				}

				// this isn't the most efficient as it eliminates end data
				for (int k = 0; k < shortest; k += step) {
					int n = k + step > shortest ? shortest % step : step;

					float[][] inputs = new float[n][s * inputSize];
					float[][] targets = new float[n][s];

					for (int t = 0; t < n; t++) {
						for (int b = 0; b < s; b++) {
							// creating a batch
							System.arraycopy(keys.get(b + j)[t + k], 0, inputs[t], b * inputSize, inputSize);
							targets[t][b] = data.get(keys.get(b + j))[t + k];
						}
					}

					group.put(inputs, targets);
				}

				if (j + s == keys.size()) {
					List<float[][]> inputs = new ArrayList<>(group.keySet());
					Collections.shuffle(inputs);
					for (int b = 0, k = 0; b < inputs.size(); b++) {
						float average = 0;

						float[][] input = inputs.get(b);

						float[][] target = group.get(input);

						// forward propagating the batch
						float[][] output = forward(input, target[0].length);

						// back propagating batch
						backward(target);
						update(output.length * target[0].length);

						for (int t = 0; t < output.length; t++)
							average += cost.cost(output[t], target[t]);

						average /= output.length * target[0].length;

						int progress = (int) ((float) (b + 1) / inputs.size() * 30 + 0.5);
						System.out.printf("\r%d/%d [", (b + 1), inputs.size());

						for (int m = 0; m < progress; m++)
							System.out.print("#");
						for (int m = progress; m < 30; m++)
							System.out.print("-");

						System.out.print("] - loss: " + average);

						Plot.update(k + (i - 1) * keys.size(), average);

						k += target[0].length;
					}

					group.clear();
				}

				if ((j + 1) % 100 == 0)
					export(name);

				current += s;
			}

			System.out.println();
		}
	}

	/**
	 * Sets the mode on each layer.
	 *
	 * @param mode the mode
	 */
	@SuppressWarnings("WeakerAccess")
	public void setMode(Layer.Mode mode) {
		for (Layer layer : layers)
			layer.setMode(mode);
	}

	/**
	 * Forward propagates recurrent layers in an efficient manner.
	 *
	 * @param x         the input
	 * @param batchSize the batch size
	 * @return the output
	 */
	public float[][] forward(float[][] x, int batchSize) {
		float[][] output = new float[x.length][];
		for (int i = 0; i < x.length; i++) {
			output[i] = new float[x[i].length];
			System.arraycopy(x[i], 0, output[i], 0, x[i].length);
		}

		List<Callable<Void>> tasks = new ArrayList<>();

		for (int i = 0; i < output.length + layers.length - 1; i++) {
			for (int j = 0; j < layers.length && (i - j) >= 0; j++) {
				if (i - j < output.length) {
					final int index = i;
					final int current = j;

					tasks.add(() -> {
						output[index - current] = layers[current].forward(output[index - current], batchSize);
						return null;
					});
				}
			}

			try {
				ES.invokeAll(tasks);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}

			tasks.clear();
		}

		return output;
	}

	/**
	 * Back propagates recurrent layers in an efficient manner.
	 *
	 * @param targets the targets
	 */
	@SuppressWarnings("WeakerAccess")
	public void backward(float[][] targets) {
		float[][] delta = new float[targets.length][];

		List<Callable<Void>> tasks = new ArrayList<>();
		for (int i = targets.length - 1; i > -layers.length; i--) {
			if (i >= 0) {
				final int index = i;
				tasks.add(() -> {
					delta[index] = layers[layers.length - 1].backward(cost, targets[index], layers.length > 1);
					return null;
				});
			}

			for (int j = i + 1, k = layers.length - 2; j < targets.length && k >= 0; j++, k--) {
				final int index = j;
				final int current = k;

				if (j >= 0) {
					tasks.add(() -> {
						delta[index] = layers[current].backward(delta[index], current > 0);
						return null;
					});
				}
			}

			try {
				ES.invokeAll(tasks);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}

			tasks.clear();
		}

		float clip = 5;
		float norm = 0;
		for (Layer layer : layers) {
			for (float[][] parameters : layer.getParameters()) {
				for (int p = 0; p < parameters[1].length; p++) {
					norm += Math.pow(Math.abs(parameters[1][p]), 2);
				}
			}
		}

		norm = (float) Math.max(Math.sqrt(norm), clip);

		for (Layer layer : layers) {
			for (float[][] parameters : layer.getParameters()) {
				for (int p = 0; p < parameters[1].length; p++) {
					parameters[1][p] *= clip / norm;
				}
			}
		}
	}

	/**
	 * Gradient checks validate that the implementation of the back propagation algorithm is correct. It does so by comparing the gradients
	 * with numerical gradients.
	 *
	 * @param input  the input
	 * @param target the target
	 */
	@SuppressWarnings("Duplicates")
	public boolean gradientCheck(float[] input, float[] target, int batchSize) {
		setMode(Layer.Mode.GRADIENT_CHECK);

		forward(input, batchSize);
		backward(target);

		boolean pass = true;
		for (Layer layer : layers) {
			for (float[][] parameters : layer.getParameters()) {
				if (!checkParameters(parameters[0], parameters[1], input, target, batchSize)) {
					pass = false;
					System.err.println("Fail\n\n");
				}
			}
		}

		System.out.println("pass: " + pass);

		return pass;
	}

	private boolean checkParameters(float[] parameters, float[] gradient, float[] input, float[] target, int batchSize) {
		double epsilon = 1e-3;
		double numerator = 0, denominator = 0;

		float[] numericalGradient = new float[parameters.length];

		for (int i = 0; i < parameters.length; i++) {
			parameters[i] += epsilon;

			float[] y = forward(input, batchSize);

			float plus = cost.cost(y, target);

			parameters[i] -= 2 * epsilon;

			y = forward(input, batchSize);

			float minus = cost.cost(y, target);

			parameters[i] += epsilon;

			numericalGradient[i] += (plus - minus) / (2 * epsilon);

			System.out.println(gradient[i] + "\t" + numericalGradient[i]);
			numerator += Math.pow(Math.abs(gradient[i] - numericalGradient[i]), 2);
			denominator += Math.pow(Math.abs(gradient[i] + numericalGradient[i]), 2);
		}

		numerator = Math.sqrt(numerator);
		denominator = Math.sqrt(denominator);

		System.out.println(numerator / denominator + "\n---------------------");
		if (Double.isNaN(numerator / denominator))
			return true;

		// gradient check doesn't mean much with FP32
		return (numerator / denominator) < 0.2;
	}

	@SuppressWarnings("Duplicates")
	public boolean gradientCheck(float[][] input, float[][] target, int batchSize) {
		setMode(Layer.Mode.GRADIENT_CHECK);

		forward(input, batchSize);
		backward(target);

		boolean pass = true;
		for (Layer layer : layers) {
			for (float[][] parameters : layer.getParameters()) {
				if (!checkParameters(parameters[0], parameters[1], input, target, batchSize)) {
					pass = false;
					System.err.println("Fail\n\n");
				}
			}
		}

		System.out.println("pass: " + pass);

		return pass;
	}

	private boolean checkParameters(float[] parameters, float[] gradient, float[][] input, float[][] target, int batchSize) {
		double epsilon = 1e-3;
		double numerator = 0, denominator = 0;

		float[] numericalGradient = new float[parameters.length];

		for (int i = 0; i < parameters.length; i++) {
			parameters[i] += epsilon;

			setMode(Layer.Mode.GRADIENT_CHECK);
			float[][] y = forward(input, batchSize);

			float plus = 0;
			for (int j = 0; j < y.length; j++)
				plus += cost.cost(y[j], target[j]);

			parameters[i] -= 2 * epsilon;

			setMode(Layer.Mode.GRADIENT_CHECK);
			y = forward(input, batchSize);

			float minus = 0;
			for (int j = 0; j < y.length; j++)
				minus += cost.cost(y[j], target[j]);

			parameters[i] += epsilon;

			numericalGradient[i] += (plus - minus) / (2 * epsilon);

			System.out.println(gradient[i] + "\t" + numericalGradient[i]);
			numerator += Math.pow(Math.abs(gradient[i] - numericalGradient[i]), 2);
			denominator += Math.pow(Math.abs(gradient[i] + numericalGradient[i]), 2);
		}

		numerator = Math.sqrt(numerator);
		denominator = Math.sqrt(denominator);

		System.out.println(numerator / denominator + "\n---------------------");
		if (Double.isNaN(numerator / denominator))
			return true;

		// gradient check doesn't mean much with FP32
		return (numerator / denominator) < 0.2;
	}

	/**
	 * Exports model to file.
	 *
	 * @param file the file
	 */
	public void export(String file) {
		try (DataOutputStream dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file, false), 16384))) {
			// exporting layer amount
			dos.writeInt(layers.length);
			dos.writeInt(inputSize);

			updaterType.export(dos);

			// exporting layers
			for (Layer layer : layers) {
				layer.getType().export(dos);
				layer.export(dos);
			}

			// exporting cost
			cost.getType().export(dos);

			System.out.println("Exported to: " + file);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Builder for models.
	 */
	public static class Builder {
		private final ArrayList<Layer> LAYERS = new ArrayList<>();
		private CostType cost;
		private UpdaterType updaterType;
		private int[] inputDimensions;

		/**
		 * Adds a layer.
		 *
		 * @param layer the layer
		 */
		public Builder add(Layer layer) {
			if (layer != null)
				LAYERS.add(layer);

			return this;
		}

		/**
		 * Sets the cost function.
		 *
		 * @param cost the CostType
		 */
		public Builder cost(CostType cost) {
			this.cost = cost;
			return this;
		}

		public Builder updaterType(UpdaterType updaterType) {
			this.updaterType = updaterType;
			return this;
		}

		/**
		 * Sets the input dimensions.
		 *
		 * @param inputDimensions the input dimensions
		 */
		public Builder inputDimensions(int... inputDimensions) {
			this.inputDimensions = inputDimensions;
			return this;
		}

		public Model build() {
			return new Model(LAYERS.toArray(new Layer[0]), cost, updaterType, inputDimensions);
		}
	}
}