import neuralnet.Model;
import neuralnet.activations.ActivationType;
import neuralnet.activations.OutputActivationType;
import neuralnet.costs.CostType;
import neuralnet.initializers.Constant;
import neuralnet.layers.*;
import neuralnet.optimizers.UpdaterType;

/**
 * Example implementation of GoogleNet, as described by the paper: Going Deeper with Convolutions
 * https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
 */
public class GoogLeNet {
@SuppressWarnings("UnnecessaryLocalVariable")
	public static Model create() {
		Model model = new Model.Builder().add(
			new Convolutional.Builder().filterSize(7).filterAmount(64).pad(3).activationType(ActivationType.RELU).initializer(
				new Constant(0)).stride(2).updaterType(UpdaterType.ADAM).build()
		).add(
			new Pooling.Builder().downsampleSize(3).downsampleStride(2).mode(Pooling.Mode.MAX).build()
		).add(
			new LRN.Builder().n(5).k(1).alpha(0.0001f).beta(0.75f).build()
		).add(
			new Convolutional.Builder().filterSize(1).filterAmount(64).pad(0).activationType(ActivationType.RELU).initializer(
				new Constant(0)).stride(1).updaterType(UpdaterType.ADAM).build()
		).add(
			new Convolutional.Builder().filterSize(3).filterAmount(192).pad(1).activationType(ActivationType.RELU).initializer(
				new Constant(0)).stride(1).updaterType(UpdaterType.ADAM).build()
		).add(
			new LRN.Builder().n(5).k(1).alpha(0.0001f).beta(0.75f).build()
		).add(
			new Pooling.Builder().downsampleSize(3).downsampleStride(2).mode(Pooling.Mode.MAX).pad(1).build()
		).add(
			new Inception.Builder().filterAmount(64, 96, 16, 128, 32, 32).initializer(new Constant(0)).updaterType(UpdaterType.ADAM)
				.build()
		).add(
			new Inception.Builder().filterAmount(128, 128, 32, 192, 96, 64).initializer(new Constant(0)).updaterType(UpdaterType.ADAM)
				.build()
		).add(
			new Pooling.Builder().downsampleSize(3).downsampleStride(2).mode(Pooling.Mode.MAX).build()
		).add(
			new Inception.Builder().filterAmount(192, 96, 16, 208, 48, 64).initializer(new Constant(0)).updaterType(UpdaterType.ADAM)
				.build()
		).add(
			new Inception.Builder().filterAmount(160, 112, 24, 224, 64, 64).initializer(new Constant(0)).updaterType(UpdaterType.ADAM)
				.build()
		).add(
			new Inception.Builder().filterAmount(128, 128, 24, 256, 64, 64).initializer(new Constant(0)).updaterType(UpdaterType.ADAM)
				.build()
		).add(
			new Inception.Builder().filterAmount(112, 144, 32, 288, 64, 64).initializer(new Constant(0)).updaterType(UpdaterType.ADAM)
				.build()
		).add(
			new Inception.Builder().filterAmount(256, 160, 32, 320, 128, 128).initializer(new Constant(0)).updaterType(UpdaterType
				.ADAM).build()
		).add(
			new Pooling.Builder().downsampleSize(3).downsampleStride(2).mode(Pooling.Mode.MAX).build()
		).add(
			new Inception.Builder().filterAmount(256, 160, 32, 320, 128, 128).initializer(new Constant(0)).updaterType(UpdaterType
				.ADAM).build()
		).add(
			new Inception.Builder().filterAmount(384, 192, 48, 384, 128, 128).initializer(new Constant(0)).updaterType(UpdaterType
				.ADAM).build()
		).add(
			new Pooling.Builder().downsampleSize(7).downsampleStride(1).mode(Pooling.Mode.AVERAGE).build()
		).add(
			new Dropout.Builder().dropout(0.4f).build()
		).add(
			new Dense.Builder().outputSize(1000).activation(OutputActivationType.SOFTMAX).initializer(new Constant(0))
				.updaterType(UpdaterType.ADAM).temperature(1).build()
		).cost(CostType.SPARSE_CROSS_ENTROPY).inputDimensions(224, 224, 3).build();

		return model;
	}
}