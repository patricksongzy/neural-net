package main.neuralnet.costs;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * The CostType is used for exporting and importing neural networks, and for repeatedly creating instances of a cost.
 */
public enum CostType {
	CROSS_ENTROPY, MEAN_SQUARE_ERROR, SPARSE_CROSS_ENTROPY;

	/**
	 * Creates a CostType, given an input stream.
	 *
	 * @param dis the input stream
	 * @return the CostType
	 */
	public static CostType fromString(DataInputStream dis) throws IOException {
		return valueOf(dis.readUTF());
	}

	/**
	 * Creates an instance, given the current CostType.
	 *
	 * @return an instance of the current CostType
	 */
	public Cost create() {
		switch (this) {
			case MEAN_SQUARE_ERROR:
				return new MeanSquareError();
			case SPARSE_CROSS_ENTROPY:
				return new SparseCrossEntropy();
			case CROSS_ENTROPY:
			default:
				return new CrossEntropy();
		}
	}

	/**
	 * Exports the CostType.
	 *
	 * @param dos the output stream
	 */
	public void export(DataOutputStream dos) throws IOException {
		dos.writeUTF(toString());
	}
}