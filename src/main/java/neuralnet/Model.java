package neuralnet;

import neuralnet.costs.Cost;
import neuralnet.costs.CostType;
import neuralnet.layers.Layer;
import neuralnet.layers.LayerType;

import javax.swing.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Models represent neural network models. They forward and back propagate layers.
 */
@SuppressWarnings("unused")
public class Model {
	private Layer[] layers;
	private Cost cost;

	private Model(Layer[] layers, CostType costType, int[] inputDimensions) {
		this.layers = layers;
		this.cost = costType.create();

		layers[0].setDimensions(inputDimensions); // setting input dimensions

		// each layer's output is the next layer's input
		for (int i = 1; i < layers.length; i++)
			layers[i].setDimensions(layers[i - 1].getOutputDimensions());
	}

	/**
	 * Imports a model from a file.
	 *
	 * @param file the path to the file
	 */
	public Model(String file) {
		DataInputStream dis = null;

		try {
			dis = new DataInputStream(new BufferedInputStream(new FileInputStream(file)));
			System.out.println("Importing from: " + file);

			// importing layers
			int layerAmount = dis.readInt();
			layers = new Layer[layerAmount];
			for (int i = 0; i < layerAmount; i++) {
				System.out.println("Importing layer " + (i + 1) + " / " + layerAmount);
				layers[i] = LayerType.fromString(dis);
				System.out.println("Imported layer " + (i + 1) + " / " + layerAmount + " of type " + layers[i].getType());
			}

			cost = CostType.fromString(dis).create();

			System.out.println("Imported from: " + file);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		} finally {
			// closing the streams
			try {
				if (dis != null) {
					dis.close();
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * Backpropagates layers, by calculating gradients.
	 *
	 * @param target the target
	 */
	private void backward(float[][] target) {
		float[][] delta = layers[layers.length - 1].backward(cost, target);

		for (int i = layers.length - 2; i >= 0; i--)
			delta = layers[i].backward(delta);
	}

	/**
	 * Forward propagates layers.
	 *
	 * @param x the input
	 * @return the output
	 */
	public float[][] forward(float[][] x) {
		for (Layer layer : layers)
			x = layer.forward(x);

		return x;
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
	 * Trains a neural network given a map of <code>float[]</code> and <code>float[]</code>. The keys represent the training data, while
	 * the values represent the targets. This works for smaller datasets, but takes lots of memory.
	 * The batch size dictates the amount of training data the network learns, before updating parameters.
	 *
	 * @param data      the map of inputs and targets
	 * @param batchSize the batch size
	 * @param epochs    the amount of complete iterations of the training set
	 * @param interval  the interval to export
	 * @param name      the model name
	 */
	public void train(Map<float[], float[]> data, int batchSize, int epochs, int interval, String name) {
		Plot plot = new Plot();
		JFrame frame = new JFrame("Network Cost");
		frame.setSize(1600, 800);
		frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		frame.add(plot);
		frame.setVisible(true);

		// setting mode to training mode
		setMode(Layer.Mode.TRAIN);

		List<float[]> keys = new ArrayList<>(data.keySet());

		// noinspection IntegerDivisionInFloatingPointContext
		plot.init(epochs, keys.size() / batchSize + ((keys.size() % batchSize) > 0 ? 1 : 0));

		int x = 0;
		for (int i = 1; i <= epochs; i++) {
			// shuffling data prevents the neural network from learning the order of the data
			Collections.shuffle(keys);

			// looping through the training set
			for (int j = 0, batch = 1; j < keys.size(); j += batchSize, batch++) {
				float error = 0;

				// calculating the batch size
				int s = (j + batchSize) > keys.size() ? (keys.size() % batchSize) : batchSize;

				float[][] inputs = new float[s][];
				float[][] targets = new float[s][];

				// creating a batch
				for (int k = j, b = 0; k < j + s; k++, b++) {
					float[] input = keys.get(k);
					inputs[b] = input;
					targets[b] = data.get(input);
				}

				// forward propagating the batch
				float[][] out = forward(inputs);

				// calculating cost
				for (int b = 0; b < s; b++) {
					error += cost.cost(out[b], targets[b]);
				}

				// back propagating batch
				backward(targets);

				plot.update(x++, error / s, i, batch, j);
				if (batch % interval == 0)
					export(name);
			}
		}
	}

	/**
	 * Trains a neural network given a map of <code>float[]</code> and <code>float[]</code>. The keys represent the training data, while
	 * the values represent the targets. This works for smaller datasets, but takes lots of memory.
	 * The batch size dictates the amount of training data the network learns, before updating parameters.
	 * Targets must be encoded sparsely. This method does not support one-hot encoding, for performance reasons.
	 *
	 * @param data      a map of the inputs and the <b>sparse-encoded</b> targets
	 * @param batchSize the batch size
	 * @param epochs    the epochs
	 * @param interval  the interval to export
	 * @param name      the name to export to
	 */
	public void trainRecurrent(Map<float[][], int[]> data, int batchSize, int epochs, int interval, String name) {
		Plot plot = new Plot();
		JFrame frame = new JFrame("Network Cost");
		frame.setSize(1600, 800);
		frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		frame.add(plot);
		frame.setVisible(true);

		// setting mode to training mode
		setMode(Layer.Mode.TRAIN);

		List<float[][]> keys = new ArrayList<>(data.keySet());

		// noinspection IntegerDivisionInFloatingPointContext
		plot.init(epochs, keys.size() / batchSize + ((keys.size() % batchSize) > 0 ? 1 : 0));

		int x = 0;
		for (int i = 1; i <= epochs; i++) {
			// shuffling data prevents the neural network from learning the order of the data
			Collections.shuffle(keys);

			// looping through the training set
			for (int j = 0; j < keys.size(); j++) {
				float[][] key = keys.get(j);
				int[] value = data.get(key);

				for (int k = 0, batch = 1; k < key.length; k += batchSize, batch++) {
					float error = 0;

					int s = (k + batchSize) > key.length ? (key.length % batchSize) : batchSize;

					float[][] inputs = new float[s][];
					float[][] targets = new float[s][];

					for (int t = k, b = 0; t < k + s; t++, b++) {
						inputs[b] = key[t];

						targets[b] = new float[]{value[t]};
					}

					// forward propagating the batch
					float[][] out = forward(inputs);

					// calculating cost
					for (int b = 0; b < s; b++) {
						error += cost.cost(out[b], targets[b]);
					}

					// back propagating batch
					backward(targets);

					plot.update(x++, error / s, i, 1, j);
				}
			}

			if (i % interval == 0)
				export(name);
		}
	}

	/**
	 * Gradient checks validate that the implementation of the back propagation algorithm is correct. It does so by comparing the gradients
	 * with numerical gradients.
	 *
	 * @param input  the input
	 * @param target the target
	 */
	public boolean gradientCheck(float[][] input, float[][] target) {
		setMode(Layer.Mode.GRADIENT_CHECK);

		forward(input);
		backward(target);

		boolean pass = true;
		for (Layer layer : layers) {
			for (float[][] parameters : layer.getParameters()) {
				if (!checkParameters(parameters[0], parameters[1], input, target)) {
					pass = false;
					System.err.println("Fail\n\n");
				}
			}
		}

		System.out.println("pass: " + pass);

		return pass;
	}

	private boolean checkParameters(float[] parameters, float[] gradient, float[][] x, float[][] target) {
		double epsilon = 1e-2;
		double numerator = 0, denominator = 0;

		float[] numericalGradient = new float[parameters.length];

		for (int i = 0; i < parameters.length; i++) {
			parameters[i] += epsilon;

			float[][] y = forward(x);

			float plus = 0;
			for (int t = 0; t < x.length; t++)
				plus += cost.cost(y[t], target[t]);

			parameters[i] -= 2 * epsilon;

			y = forward(x);

			float minus = 0;
			for (int t = 0; t < x.length; t++)
				minus += cost.cost(y[t], target[t]);

			parameters[i] += epsilon;

			numericalGradient[i] += (plus - minus) / (2 * epsilon);

			System.out.println(gradient[i] + "\t" + numericalGradient[i]);
			numerator += Math.pow(Math.abs(gradient[i] - numericalGradient[i]), 2);
			denominator += Math.pow(Math.abs(gradient[i] + numericalGradient[i]), 2);
		}

		numerator = Math.sqrt(numerator);
		denominator = Math.sqrt(denominator);

		System.out.println(numerator / denominator + "\n---------------------");

		// gradient check doesn't mean much with FP32
		return (numerator / denominator) < 0.08;
	}

	/**
	 * Exports model to file.
	 *
	 * @param file the file
	 */
	public void export(String file) {
		DataOutputStream dos = null;

		try {
			dos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file, false)));
			System.out.println("Exporting to: " + file);

			// exporting layer amount
			dos.writeInt(layers.length);

			// exporting layers
			for (int i = 0; i < layers.length; i++) {
				System.out.println("Exporting layer " + (i + 1) + " / " + layers.length);
				layers[i].getType().export(dos);
				layers[i].export(dos);
				System.out.println("Exported layer " + (i + 1) + " / " + layers.length + " of type " + layers[i].getType());
			}

			// exporting cost
			cost.getType().export(dos);
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			// closing streams
			try {
				if (dos != null) {
					dos.flush();
					dos.close();
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		System.out.println("Exported to: " + file);
	}

	/**
	 * Builder for models.
	 */
	public static class Builder {
		private final ArrayList<Layer> LAYERS = new ArrayList<>();
		private CostType cost;
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
			if (LAYERS.size() > 0 && cost != null && inputDimensions != null) {
				return new Model(LAYERS.toArray(new Layer[0]), cost, inputDimensions);
			}

			throw new IllegalArgumentException();
		}
	}
}