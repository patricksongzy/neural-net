package neuralnet.optimizers;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * The UpdaterType is used for exporting and importing neural networks, and for repeatedly creating instances of an updater.
 */
@SuppressWarnings("Duplicates")
public enum UpdaterType {
	ADAM, AMSGRAD;

	/**
	 * Creates an UpdaterType, given an input stream.
	 *
	 * @param dis the input stream
	 * @return the UpdaterType
	 * @throws IOException if there is an error reading from the file
	 */
	public static UpdaterType fromString(DataInputStream dis) throws IOException {
		switch (valueOf(dis.readUTF())) {
			case ADAM:
				Adam.importParameters(dis);
				return ADAM;
			case AMSGRAD:
				AMSGrad.importParameters(dis);
				return AMSGRAD;
			default:
				return null;
		}
	}

	public void init(float learningRate) {
		switch (this) {
			case ADAM:
				Adam.setLearningRate(learningRate);
				break;
			case AMSGRAD:
				AMSGrad.setLearningRate(learningRate);
				break;
			default:
		}
	}

	public void setDecay(float decay) {
		switch (this) {
			case ADAM:
				break;
			case AMSGRAD:
				AMSGrad.setLambda(decay);
				break;
			default:
		}
	}

	public void init(float... parameters) {
		switch (this) {
			case ADAM:
				Adam.init(parameters);
				break;
			case AMSGRAD:
				AMSGrad.init(parameters);
				break;
			default:
		}
	}

	/**
	 * Creates an instance, given the current UpdaterType.
	 *
	 * @param size the size of the parameters
	 * @param decay the decay
	 * @return an instance of the current UpdaterType
	 */
	public Updater create(int size, boolean decay) {
		switch (this) {
			case ADAM:
				return new Adam(size);
			case AMSGRAD:
				return new AMSGrad(size, decay);
			default:
				return null;
		}
	}

	/**
	 * Creates an Updater, given the current type, then imports it's parameters given an input stream.
	 *
	 * @param dis the input stream
	 * @return the updater
	 * @throws IOException if there is an error reading from the file
	 */
	public Updater create(DataInputStream dis) throws IOException {
		switch (this) {
			case ADAM:
				return new Adam(dis);
			case AMSGRAD:
				return new AMSGrad(dis);
			default:
				return null;
		}
	}

	/**
	 * Exports the UpdaterType.
	 *
	 * @param dos the output stream
	 * @throws IOException if there is an error writing to the file
	 */
	public void export(DataOutputStream dos) throws IOException {
		dos.writeUTF(toString());

		switch (this) {
			case ADAM:
				Adam.exportParameters(dos);
				break;
			case AMSGRAD:
				AMSGrad.exportParameters(dos);
				break;
			default:
		}
	}
}
