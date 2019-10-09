package neuralnet.costs;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.stream.IntStream;

/**
 * The CostType is used for exporting and importing neural networks, and for repeatedly creating instances of a cost.
 */
public enum CostType implements Cost {
	CROSS_ENTROPY {
		public CostType getType() {
			return CostType.CROSS_ENTROPY;
		}

		public float cost(float[] out, float[] targets) {
			return (float) -IntStream.range(0, targets.length).parallel().mapToDouble(i -> (targets[i] * Math.log(out[i] + 1e-16))).sum();
		}

		public float[] derivative(float[] output, float[] targets, int batchSize) {
			int size = output.length / batchSize;

			if (batchSize <= 0)
				throw new IllegalArgumentException("Batch size must be > 0.");
			if (output.length < (size - 1) + size * (batchSize - 1) || output.length != targets.length)
				throw new IllegalArgumentException("Invalid array lengths.");

			float[] delta = new float[output.length];

			IntStream.range(0, batchSize).parallel().forEach(b -> {
				for (int i = 0; i < size; i++) {
					int index = i + size * b;

					delta[i + size * b] = (-targets[index] / output[index]);
				}
			});

			return delta;
		}

		public float[] derivativeSoftmax(float[] output, float[] targets, int batchSize) {
			int size = output.length / batchSize;

			if (batchSize <= 0)
				throw new IllegalArgumentException("Batch size must be > 0.");
			if (output.length < (size - 1) + size * (batchSize - 1) || output.length != targets.length)
				throw new IllegalArgumentException("Invalid array lengths.");

			float[] delta = new float[output.length];

			IntStream.range(0, batchSize).parallel().forEach(b -> {
				for (int i = 0; i < size; i++) {
					int index = i + size * b;

					delta[index] = (output[index] - targets[index]);
				}
			});

			return delta;
		}
	}, MEAN_SQUARE_ERROR {
		public CostType getType() {
			return CostType.MEAN_SQUARE_ERROR;
		}

		public float cost(float[] output, float[] targets) {
			if (output.length != targets.length)
				throw new IllegalArgumentException("Invalid array lengths.");

			float cost = (float) IntStream.range(0, targets.length).parallel().mapToDouble(i -> Math.pow(targets[i] - output[i], 2)).sum();

			return 0.5f * cost;
		}

		public float[] derivative(float[] output, float[] targets, int batchSize) {
			int size = output.length / batchSize;

			if (batchSize <= 0)
				throw new IllegalArgumentException("Batch size must be > 0.");
			if (output.length < (size - 1) + size * (batchSize - 1) || targets.length < output.length)
				throw new IllegalArgumentException("Invalid array lengths.");

			float[] delta = new float[output.length];

			IntStream.range(0, batchSize).parallel().forEach(b -> {
				for (int i = 0; i < size; i++) {
					int index = i + size * b;

					delta[index] = output[index] - targets[index];
				}
			});

			return delta;
		}

		public float[] derivativeSoftmax(float[] output, float[] targets, int batchSize) {
			throw new UnsupportedOperationException();
		}
	}, SPARSE_CROSS_ENTROPY {
		public CostType getType() {
			return CostType.SPARSE_CROSS_ENTROPY;
		}

		public float cost(float[] output, float[] targets) {
			int size = output.length / targets.length;

			if (output.length < (size - 1) + size * (targets.length - 1))
				throw new IllegalArgumentException("Invalid array lengths.");

			float cost = 0;
			for (int b = 0; b < targets.length; b++) {
				if (targets[b] > size) {
					throw new IllegalArgumentException("Invalid target.");
				} else if (targets[b] != -1) {
					cost += Math.log(output[(int) targets[b] + size * b] + 1e-16);
				}
			}

			return -cost;
		}

		public float[] derivative(float[] output, float[] targets, int batchSize) {
			int size = output.length / batchSize;

			if (batchSize <= 0)
				throw new IllegalArgumentException("Batch size must be > 0.");
			if (output.length < (size - 1) + size * (batchSize - 1) || output.length != targets.length)
				throw new IllegalArgumentException("Invalid array lengths.");

			float[] delta = new float[output.length];

			IntStream.range(0, batchSize).parallel().forEach(b -> {
				int index = (int) targets[b] + size * b;

				if (targets[b] > size) {
					throw new IllegalArgumentException("Invalid targets.");
				} else if (targets[b] != -1) {
					delta[index] = -1 / output[index];
				}
			});

			return delta;
		}

		public float[] derivativeSoftmax(float[] output, float[] targets, int batchSize) {
			int size = output.length / batchSize;

			if (batchSize <= 0)
				throw new IllegalArgumentException("Batch size must be > 0.");
			if (output.length < (size - 1) + size * (batchSize - 1))
				throw new IllegalArgumentException("Invalid array lengths.");

			float[] delta = new float[output.length];

			IntStream.range(0, batchSize).parallel().forEach(b -> {
				int index = (int) targets[b] + size * b;

				if (targets[b] > size) {
					throw new IllegalArgumentException("Invalid targets.");
				} else if (targets[b] != -1) {
					System.arraycopy(output, size * b, delta, size * b, size);
					delta[index] -= 1;
				}
			});

			return delta;
		}
	};

	/**
	 * Creates a CostType, given an input stream.
	 *
	 * @param dis the input stream
	 * @return the CostType
	 * @throws IOException if there is an error reading the file
	 */
	public static CostType fromString(DataInputStream dis) throws IOException {
		return valueOf(dis.readUTF());
	}

	/**
	 * Exports the CostType.
	 *
	 * @param dos the output stream
	 * @throws IOException if there is an error writing to the file
	 */
	public void export(DataOutputStream dos) throws IOException {
		dos.writeUTF(toString());
	}
}
