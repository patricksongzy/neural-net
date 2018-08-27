package neuralnet.activations;

import java.util.stream.IntStream;

public enum ActivationType implements Activation {
	RELU {
		public Type getType() {
			return Type.RELU;
		}

		public void activation(float[] x, int batchSize) {
			if (batchSize <= 0)
				throw new IllegalArgumentException("Batch size must be > 0");

			IntStream.range(0, x.length).parallel().forEach(i -> {
				// same as max of 0 and x
				if (x[i] < 0)
					x[i] = 0;
			});
		}

		public float[] derivative(float[] x) {
			float[] derivative = new float[x.length];

			IntStream.range(0, x.length).parallel().forEach(i -> derivative[i] = x[i] > 0 ? 1 : 0);

			return derivative;
		}
	}, IDENTITY {
		public Type getType() {
			return Type.IDENTITY;
		}

		public void activation(float[] x, int batchSize) {
		}

		public float[] derivative(float[] x) {
			float[] derivative = new float[x.length];

			IntStream.range(0, x.length).parallel().forEach(b -> derivative[b] = 1);

			return derivative;
		}
	}, TANH {
		public Type getType() {
			return Type.TANH;
		}

		public void activation(float[] x, int batchSize) {
			if (batchSize <= 0)
				throw new IllegalArgumentException("Batch size must be > 0");

			IntStream.range(0, x.length).parallel().forEach(i -> x[i] = (float) Math.tanh(x[i]));
		}

		public float[] derivative(float[] x) {
			float[] derivative = new float[x.length];

			IntStream.range(0, x.length).parallel().forEach(i -> derivative[i] = 1 - (float) Math.pow(x[i], 2));

			return derivative;
		}
	}, SIGMOID {
		public Type getType() {
			return Type.SIGMOID;
		}

		public void activation(float[] x, int batchSize) {
			if (batchSize <= 0)
				throw new IllegalArgumentException("Batch size must be > 0");

			IntStream.range(0, x.length).parallel().forEach(i -> x[i] = 1 / (float) (1 + Math.exp(-x[i])));
		}

		public float[] derivative(float[] x) {
			float[] derivative = new float[x.length];

			IntStream.range(0, x.length).parallel().forEach(i -> derivative[i] = x[i] * (1 - x[i]));

			return derivative;
		}
	}
}