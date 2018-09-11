package neuralnet.optimizers;

import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Updaters calculate deltas to update parameters.
 */
public interface Updater {
	/**
	 * Calculates the delta for the parameters.
	 *
	 * @param gradient the parameter gradient
	 * @return the delta
	 */
	float[] update(float[] gradient);

	/**
	 * Export the parameter specific parameters to an output stream.
	 *
	 * @param dos the output stream
	 */
	void export(DataOutputStream dos) throws IOException;
}