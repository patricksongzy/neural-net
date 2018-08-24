package neuralnet.activations;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * The ActivationType is used for exporting and importing neural networks, and for repeatedly creating instances of an activation.
 */
public enum ActivationType {
	RELU, SOFTMAX, IDENTITY, TANH, SIGMOID;

	/**
	 * Creates an ActivationType, given a String.
	 *
	 * @param dis the input stream
	 * @return the ActivationType
	 */
	public static ActivationType fromString(DataInputStream dis) throws IOException {
		return valueOf(dis.readUTF());
	}

	/**
	 * Creates an instance, given the current ActivationType.
	 *
	 * @return an instance of the current ActivationType.
	 */
	public Activation create() {
		switch (this) {
			case SOFTMAX:
				return new Softmax();
			case IDENTITY:
				return new Identity();
			case TANH:
				return new TanH();
			case SIGMOID:
				return new Sigmoid();
			case RELU:
			default:
				return new ReLU();
		}
	}

	/**
	 * Exports the ActivationType.
	 *
	 * @param dos the output stream
	 */
	public void export(DataOutputStream dos) throws IOException {
		dos.writeUTF(toString());
	}
}