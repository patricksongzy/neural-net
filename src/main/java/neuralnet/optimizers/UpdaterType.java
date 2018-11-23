package neuralnet.optimizers;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * The UpdaterType is used for exporting and importing neural networks, and for repeatedly creating instances of an updater.
 */
public enum UpdaterType {
	ADAM, AMSGRAD;

	/**
	 * Creates an UpdaterType, given an input stream.
	 *
	 * @param dis the input stream
	 * @return the UpdaterType
	 */
	public static UpdaterType fromString(DataInputStream dis) throws IOException {
		switch (valueOf(dis.readUTF())) {
			case ADAM:
				Adam.importParameters(dis);
				return ADAM;
			case AMSGRAD:
			default:
				AMSGrad.importParameters(dis);
				return AMSGRAD;
		}
	}

	/**
	 * Creates an instance, given the current UpdaterType.
	 *
	 * @return an instance of the current UpdaterType
	 */
	public Updater create(int size) {
		switch (this) {
			case ADAM:
				return new Adam(size);
			case AMSGRAD:
			default:
				return new AMSGrad(size);
		}
	}

	/**
	 * Creates an Updater, given the current type, then imports it's parameters given an input stream.
	 *
	 * @param dis the input stream
	 * @return the updater
	 */
	public Updater create(DataInputStream dis) throws IOException {
		switch (this) {
			case ADAM:
				return new Adam(dis);
			case AMSGRAD:
			default:
				return new AMSGrad(dis);
		}
	}

	/**
	 * Exports the UpdaterType.
	 *
	 * @param dos the output stream
	 */
	public void export(DataOutputStream dos) throws IOException {
		dos.writeUTF(toString());

		switch (this) {
			case ADAM:
				Adam.exportParameters(dos);
				break;
			case AMSGRAD:
			default:
				AMSGrad.exportParameters(dos);
		}
	}
}