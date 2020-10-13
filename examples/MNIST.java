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
import neuralnet.schedules.CosineRestart;
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
					.build()
			)
			.cost(CostType.SPARSE_CROSS_ENTROPY)
			.updaterType(UpdaterType.AMSGRAD)
			.inputDimensions(28, 28, 1)
			.build();

		// load MNIST data (not defined in library)
		List<int[][]> trainImages = readImages("train-images.idx3-ubyte");
		int[] trainLabels = readLabels("train-labels.idx1-ubyte");

		Map<float[], float[]> trainData = new HashMap<>();
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
			trainData.put(trainInputs[i], new float[] {trainLabels[i]});
		}

		// train model
		UpdaterType.AMSGRAD.init(0.9f, 0.999f, 0.1f, 0.125f);
		model.setSchedule(new CosineRestart(0.1f, 0.001f, 0.05f, 1, 2, 1));
		model.train(trainData, batchSize, epochs, 1, "mnist.model");
	}
}
