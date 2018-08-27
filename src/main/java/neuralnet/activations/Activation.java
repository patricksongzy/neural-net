package neuralnet.activations;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Activations simulate the firing of neurons in a neural network. They must be differentiable, in order to back-propagate.
 */
public interface Activation {
	/**
	 * Creates a Type, given a String.
	 *
	 * @param dis the input stream
	 * @return the ActivationType
	 */
	static Activation fromString(DataInputStream dis) throws IOException {
		return Type.valueOf(dis.readUTF()).create();
	}

	/**
	 * This method simulates the activation of neurons. It directly modifies the input.
	 *
	 * @param x the pre-activated input
	 */
	void activation(float[] x, int batchSize);

	/**
	 * This method calculates the derivative of the activation.
	 *
	 * @param x the activated output
	 */
	float[] derivative(float[] x);

	/**
	 * This method returns the activation type for use when exporting neural networks.
	 *
	 * @return the activation type
	 */
	Type getType();

	/**
	 * Exports the ActivationType.
	 *
	 * @param dos the output stream
	 */
	default void export(DataOutputStream dos) throws IOException {
		dos.writeUTF(getType().toString());
	}

	enum Type implements ActivationFactory {
		RELU {
			public Activation create() {
				return ActivationType.RELU;
			}
		}, IDENTITY {
			public Activation create() {
				return ActivationType.IDENTITY;
			}
		}, TANH {
			public Activation create() {
				return ActivationType.TANH;
			}
		}, SIGMOID {
			public Activation create() {
				return ActivationType.SIGMOID;
			}
		}, SOFTMAX {
			public Activation create() {
				return OutputActivationType.SOFTMAX;
			}
		}
	}

	interface ActivationFactory {
		Activation create();
	}
}