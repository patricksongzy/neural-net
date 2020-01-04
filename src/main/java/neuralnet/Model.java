package neuralnet;

import javafx.application.Application;
import neuralnet.costs.Cost;
import neuralnet.costs.CostType;
import neuralnet.layers.Layer;
import neuralnet.layers.LayerType;
import neuralnet.layers.graph.Node;
import neuralnet.optimizers.UpdaterType;
import neuralnet.schedules.Schedule;
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
@SuppressWarnings("unused") public class Model {
	private static final int CORES = Runtime.getRuntime().availableProcessors();
	private static final ThreadPoolExecutor ES =
		new ThreadPoolExecutor(CORES, CORES, 0L, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<>(),
			new ThreadPoolExecutor.CallerRunsPolicy());

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

	private Layer[] inputLayers;
	private Layer[] layers;
	private Schedule schedule;
	private Cost cost;

	private Model(Layer[] layers, boolean areLayersSequential, CostType costType, UpdaterType updaterType, int[] inputDimensions) {
		if (layers.length <= 0)
			throw new IllegalArgumentException("Invalid layer amount.");

		Objects.requireNonNull(costType);
		Objects.requireNonNull(updaterType);
		Objects.requireNonNull(inputDimensions);

		this.layers = areLayersSequential ? layers : sortLayers(layers);

		this.schedule = new Schedule() {
			@Override public void endEpoch(int i) {
			}

			@Override public void increment(int s) {
			}

			@Override public void init(UpdaterType updaterType, int batchSize, int keyAmount) {

			}

			@Override public void step() {
			}
		};

		this.cost = costType;
		this.updaterType = updaterType;

		inputSize = inputDimensions[0];
		for (int i = 1; i < inputDimensions.length; i++)
			inputSize *= inputDimensions[i];

		this.layers[0].setDimensions(inputDimensions, updaterType); // setting input dimensions

		if (areLayersSequential) {
			inputLayers = new Layer[] { this.layers[0] };

			// each layer's output is the next layer's input
			for (int i = 1; i < layers.length; i++) {
				this.layers[i].setChildren(new Node[] { this.layers[i - 1] });
				this.layers[i].setDimensions(this.layers[i - 1].getOutputDimensions(), updaterType);
			}
		} else {
			inputLayers = Arrays.stream(this.layers).filter(layer -> layer.getChildren().length == 0).toArray(Layer[]::new);
			propagateDimensionsConsumers(this.layers[0]);
		}
	}

	private void propagateDimensionsConsumers(Layer layer) {
		layer.getConsumers().forEach(consumer -> {
			((Layer) consumer).setDimensions(layer.getOutputDimensions(), updaterType);
			propagateDimensionsConsumers((Layer) consumer);
		});
	}

	/**
	 * Imports a model from a file.
	 *
	 * @param file the path to the file
	 */
	public Model(String file) {
		try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file), 16384))) {
			// importing layers
			int layerAmount = dis.readInt();
			inputSize = dis.readInt();

			updaterType = UpdaterType.fromString(dis);

			layers = new Layer[layerAmount];
			for (int i = 0; i < layerAmount; i++)
				layers[i] = LayerType.fromString(dis, updaterType);

			cost = CostType.fromString(dis);

			System.out.println("Imported from: " + file);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}
	}

	/**
	 * Imports a model from an input stream.
	 *
	 * @param is the input stream
	 */
	public Model(InputStream is) {
		try (DataInputStream dis = new DataInputStream(new BufferedInputStream(is, 16384))) {
			// importing layers
			int layerAmount = dis.readInt();
			inputSize = dis.readInt();

			updaterType = UpdaterType.fromString(dis);

			layers = new Layer[layerAmount];
			for (int i = 0; i < layerAmount; i++) {
				layers[i] = LayerType.fromString(dis, updaterType);
			}

			cost = CostType.fromString(dis);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}
	}

	/**
	 * Backpropagates layers, by calculating gradients.
	 *
	 * @param targets the targets
	 */
	public void backward(float[] targets) {
		// calculating the derivative of cost first
		float[] delta = layers[layers.length - 1].backward(cost, targets, layers.length > 1);

		// looping through layers backwards and feeding outputted delta as inputs.
		for (int i = layers.length - 2; i >= 0; i--)
			delta = layers[i].backward(delta, i > 0);
	}

	/**
	 * Back propagates recurrent layers in an efficient manner.
	 *
	 * @param targets the targets
	 */
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
	}

	@SuppressWarnings("Duplicates")
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
	 * Forward propagates layers.
	 *
	 * @param x         the input
	 * @param batchSize the batch size
	 * @return the output
	 */
	public float[] forward(float[] x, int batchSize) {
		// looping through each layer and feeding the output as inputs
		for (Layer layer : layers)
			x = layer.forward(x, batchSize);

		return x;
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

	public Layer getLayer(int index) {
		return layers[index];
	}

	public int getLayerAmount() {
		return layers.length;
	}

	public int[] getOutputDimensions() {
		return layers[layers.length - 1].getOutputDimensions();
	}

	/**
	 * Gradient checks validate that the implementation of the back propagation algorithm is correct. It does so by comparing the gradients
	 * with numerical gradients.
	 *
	 * @param input     the input
	 * @param target    the target
	 * @param batchSize the batch size
	 * @return whether the gradient check passes
	 */
	@SuppressWarnings("Duplicates") public boolean gradientCheck(float[] input, float[] target, int batchSize) {
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

	/**
	 * Gradient checks validate that the implementation of the back propagation algorithm is correct. It does so by comparing the gradients
	 * with numerical gradients.
	 *
	 * @param input     the input
	 * @param target    the target
	 * @param batchSize the batch size
	 * @return whether the gradient check passes
	 */
	@SuppressWarnings("Duplicates") public boolean gradientCheck(float[][] input, float[][] target, int batchSize) {
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

	private void populateDiscoverable(Layer layer, Set<Layer> discoverable) {
		discoverable.add(layer);
		Arrays.stream(layer.getChildren()).forEach(child -> populateDiscoverable((Layer) child, discoverable));
	}

	/**
	 * Sets the mode on each layer.
	 *
	 * @param mode the mode
	 */
	public void setMode(Layer.Mode mode) {
		for (Layer layer : layers)
			layer.setMode(mode);
	}

	public void setSchedule(Schedule schedule) {
		Objects.requireNonNull(schedule);
		this.schedule = schedule;
	}

	private Layer[] sortLayers(Layer... outputLayers) {
		Deque<Layer> topological = new ArrayDeque<>();

		Set<Layer> discoverable = new HashSet<>();
		Arrays.stream(outputLayers).forEach(outputLayer -> populateDiscoverable(outputLayer, discoverable));

		while (!discoverable.isEmpty()) {
			visit(discoverable.iterator().next(), discoverable, topological);
		}

		Map<Layer, Integer> distances = new HashMap<>();
		topological.forEach(layer -> distances.put(layer,
			layer.getConsumers().stream().mapToInt(consumer -> distances.getOrDefault((Layer) consumer, 1) - 1).min().orElse(0)));

		return distances.entrySet().stream().sorted(Map.Entry.comparingByValue()).map(Map.Entry::getKey).toArray(Layer[]::new);
	}

	/**
	 * Trains the model from data. For sparse data, make each target a float array with a single element.
	 *
	 * @param data       the data
	 * @param batchSize  the batch size
	 * @param epochs     the amount of epochs
	 * @param checkpoint the amount of epochs to export
	 * @param name       the exported model name
	 */
	@SuppressWarnings("Duplicates") public void train(Map<float[], float[]> data, int batchSize, int epochs, int checkpoint, String name) {
		new Thread(() -> Application.launch(Plot.class, (String) null)).start();

		// setting mode to training mode
		setMode(Layer.Mode.TRAIN);

		List<float[]> keys = new ArrayList<>(data.keySet());

		int inputSize = keys.get(0).length;
		int targetSize = data.get(keys.get(0)).length;

		schedule.init(updaterType, batchSize, keys.size());

		int batch = 0;
		for (int i = 1; i <= epochs; i++) {
			if (i % checkpoint == 0)
				export(name);

			// shuffling data prevents the neural network from learning the order of the data
			Collections.shuffle(keys);

			System.out.println("Epoch: " + i + "/" + epochs);
			// looping through the training set
			for (int j = 0; j < keys.size(); j += batchSize, batch++) {
				// calculating the batch size
				int s = (j + batchSize) > keys.size() ? (keys.size() % batchSize) : batchSize;
				schedule.step();

				float[] inputs = new float[s * inputSize];
				float[] targets = new float[s * targetSize];

				// creating a batch
				for (int b = 0; b < s; b++) {
					float[] input = keys.get(b + j);
					float[] target = data.get(input);
					System.arraycopy(input, 0, inputs, b * inputSize, inputSize);
					System.arraycopy(target, 0, targets, b * targetSize, targetSize);
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

				schedule.increment(s);
			}

			System.out.println();

			schedule.endEpoch(i);
		}
	}

	@SuppressWarnings("Duplicates")
	public void trainRecurrent(Map<float[][], float[]> data, int batchSize, int bptt, int epochs, int checkpoint, String name) {
		new Thread(() -> Application.launch(Plot.class, (String) null)).start();

		// setting mode to training mode
		setMode(Layer.Mode.TRAIN);

		List<float[][]> keys = new ArrayList<>(data.keySet());
		schedule.init(updaterType, batchSize, keys.size());

		int batch = 0;
		for (int i = 1; i <= epochs; i++) {
			if (i % checkpoint == 0)
				export(name);

			// shuffling data prevents the neural network from learning the order of the data
			Collections.shuffle(keys);

			System.out.println("Epoch: " + i + "/" + epochs);
			// looping through the training set
			for (int j = 0; j < keys.size(); j += batchSize, batch++) {
				// calculating the batch size
				int s = (j + batchSize) > keys.size() ? (keys.size() % batchSize) : batchSize;
				schedule.step();

				// going to loop through timesteps to longest input in time
				int longest = 0;

				for (int b = 0; b < s; b++) {
					if (keys.get(b + j).length > longest)
						longest = keys.get(b + j).length;
				}

				// this is the amount of separate bptt groups the longest input has
				int sections = (int) ((float) longest / bptt + 0.5);

				// the length of this list is equal to the amount of bptt groups
				Map<float[][], float[][]> groups = new HashMap<>();

				// creating a batch
				for (int k = 0; k < longest; k += bptt) {
					int n = (k + bptt) > longest ? (longest % bptt) : bptt;

					float[][] inputs = new float[n][s * inputSize];
					float[][] targets = new float[n][s];

					for (int t = 0; t < n; t++) {
						for (int b = 0; b < s; b++) {
							float[][] key = keys.get(b + j);
							if (t + k < key.length) {
								targets[t][b] = data.get(key)[t + k];

								System.arraycopy(key[t + k], 0, inputs[t], b * inputSize, inputSize);
							} else {
								targets[t][b] = -1;
							}
						}
					}

					groups.put(inputs, targets);
				}

				List<float[][]> inputs = new ArrayList<>(groups.keySet());
				for (float[][] input : inputs) {
					float[][] output = forward(input, s);
					float[][] target = groups.get(input);

					backward(target);

					update(s * output.length);

					int progress = (int) ((float) (j + s) / keys.size() * 30 + 0.5);
					System.out.printf("\r%d/%d [", j + s, keys.size());

					for (int k = 0; k < progress; k++)
						System.out.print("#");
					for (int k = progress; k < 30; k++)
						System.out.print("-");

					float average = 0;

					for (int k = 0; k < output.length; k++) {
						average += cost.cost(output[k], target[k]);
					}

					average /= (output.length * s);

					System.out.print("] - loss: " + average);

					Plot.update(batch, average);

					schedule.increment(s);
				}
			}

			System.out.println();

			schedule.endEpoch(i);
		}
	}

	/**
	 * Updates the parameters of all layers after backpropagation.
	 *
	 * @param length the length of the parameters
	 */
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

	private void visit(Layer layer, Set<Layer> discoverable, Deque<Layer> sorted) {
		if (sorted.contains(layer))
			return;

		if (!discoverable.contains(layer))
			throw new IllegalStateException("Unable to sort a non-directed graph.");

		discoverable.remove(layer);
		Arrays.stream(layer.getChildren()).forEach(child -> visit((Layer) child, discoverable, sorted));

		sorted.push(layer);
	}

	/**
	 * Builder for models.
	 */
	public static class Builder {
		private boolean areLayersSequential;
		private final ArrayList<Layer> LAYERS = new ArrayList<>();
		private Schedule schedule;
		private CostType cost;
		private UpdaterType updaterType;
		private int[] inputDimensions;

		public Builder(boolean areLayersSequential) {
			this.areLayersSequential = areLayersSequential;
		}

		/**
		 * Adds a layer.
		 *
		 * @param layer the layer
		 * @return the builder
		 */
		public Builder add(Layer layer) {
			if (layer != null)
				LAYERS.add(layer);

			return this;
		}

		/**
		 * Builds the model.
		 *
		 * @return the model
		 */
		public Model build() {
			return new Model(LAYERS.toArray(new Layer[0]), areLayersSequential, cost, updaterType, inputDimensions);
		}

		/**
		 * Sets the cost function.
		 *
		 * @param cost the CostType
		 * @return the builder
		 */
		public Builder cost(CostType cost) {
			this.cost = cost;
			return this;
		}

		/**
		 * Sets the input dimensions with the following order: depth, width, height. Any matrices are in row major format.
		 *
		 * @param inputDimensions the input dimensions
		 * @return the builder
		 */
		public Builder inputDimensions(int... inputDimensions) {
			this.inputDimensions = inputDimensions;
			return this;
		}

		/**
		 * Sets the updater type.
		 *
		 * @param updaterType the UpdaterType
		 * @return the builder
		 */
		public Builder updaterType(UpdaterType updaterType) {
			this.updaterType = updaterType;
			return this;
		}
	}
}
