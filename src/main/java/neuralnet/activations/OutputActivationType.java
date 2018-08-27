package neuralnet.activations;

import java.util.stream.IntStream;

public enum OutputActivationType implements Activation {
	SOFTMAX {
		public Type getType() {
			return Type.SOFTMAX;
		}

		public void activation(float[] x, int batchSize) {
			float[] sum = new float[batchSize];
			float[] max = new float[batchSize];

			int size = x.length / batchSize;
			IntStream.range(0, batchSize).parallel().forEach(b -> {
				max[b] = Float.NEGATIVE_INFINITY;

				for (int i = 0; i < size; i++) {
					int index = i + size * b;

					if (x[index] > max[b])
						max[b] = x[index];
				}

				for (int i = 0; i < size; i++) {
					int index = i + size * b;

					float value = (float) Math.exp(x[index] - max[b]);

					sum[b] += value;
					x[index] = value;
				}

				for (int i = 0; i < size; i++) {
					x[i + size * b] /= sum[b];
				}
			});
		}

		public float[] derivative(float[] x) {
			return x;
		}
	}
}