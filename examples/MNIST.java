import neuralnet.Model;
import neuralnet.activations.ActivationType;
import neuralnet.activations.OutputActivationType;
import neuralnet.costs.CostType;
import neuralnet.initializers.HeInitialization;
import neuralnet.layers.Convolutional;
import neuralnet.layers.Dense;
import neuralnet.layers.Dropout;
import neuralnet.layers.Pooling;
import neuralnet.optimizers.UpdaterType;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MNIST {
	private static List<int[][]> readImages(String path) {
		throw new NotImplementedException();
	}

	private static int[] readLabels(String path) {
		throw new NotImplementedException();
	}

	public static void train(int batchSize, int epochs) {
		Model model = new Model.Builder()
			.add(
				new Convolutional.Builder()
					.activationType(ActivationType.RELU)
					.filterAmount(32)
					.filterSize(5)
					.initializer(new HeInitialization())
					.pad(2)
					.stride(1)
					.updaterType(UpdaterType.ADAM)
					.build()
			)
			.add(
				new Pooling.Builder().downsampleSize(2).downsampleStride(2).pad(0).mode(Pooling.Mode.MAX).build()
			)
			.add(
				new Convolutional.Builder()
					.activationType(ActivationType.RELU)
					.filterAmount(64)
					.filterSize(5)
					.initializer(new HeInitialization())
					.pad(2)
					.stride(1)
					.updaterType(UpdaterType.ADAM)
					.build()
			)
			.add(
				new Pooling.Builder().downsampleSize(2).downsampleStride(2).pad(0).mode(Pooling.Mode.MAX).build()
			)
			.add(
				new Dense.Builder()
					.outputSize(1024)
					.activation(ActivationType.RELU)
					.initializer(new HeInitialization())
					.updaterType(UpdaterType.ADAM)
					.build()
			)
			.add(
				new Dropout.Builder().dropout(0.5f).build()
			)
			.add(
				new Dense.Builder()
					.outputSize(10)
					.activation(OutputActivationType.SOFTMAX)
					.initializer(new HeInitialization())
					.updaterType(UpdaterType.ADAM)
					.build()
			)
			.cost(CostType.SPARSE_CROSS_ENTROPY)
			.inputDimensions(28, 28, 1)
			.build();

		// load MNIST data (not defined in library)
		List<int[][]> trainImages = readImages("train-images.idx3-ubyte");
		int[] trainLabels = readLabels("train-labels.idx1-ubyte");

		Map<float[], Float> trainData = new HashMap<>();
		// create inputs
		float[][] trainInputs = new float[trainImages.size()][784];

		for (int i = 0; i < trainImages.size(); i++) {
			int pos = 0;
			for (int j = 0; j < trainImages.get(i).length; j++) {
				for (int k = 0; k < trainImages.get(i)[j].length; k++) {
					// normalize
					trainInputs[i][pos++] = trainImages.get(i)[j][k] / 255.0f;
				}
			}

			// use indices for sparse cross-entropy instead of one-hot
			trainData.put(trainInputs[i], (float) trainLabels[i]);
		}

		// train model
		model.train(trainData, batchSize, epochs);
	}
}