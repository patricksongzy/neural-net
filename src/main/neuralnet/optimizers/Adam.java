package main.neuralnet.optimizers;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * The ADAM updater stores <code>m</code> and <code>v</code> as parameters for each layer. Although other updaters may outperform ADAM,
 * ADAM is commonly used, due to the fact that it works well without adjusting any hyper-parameters.
 */
public class Adam implements Updater {
	private static double beta1 = 0.9f;
	private static double beta2 = 0.999f;
	private static double epsilon = 1e-8f;
	private static double learningRate = 0.0001f;

	private int t = 0;
	private double m = 0, v = 0;

	public Adam() {

	}

	public Adam(DataInputStream dis) throws IOException {
		t = dis.readInt();
		m = dis.readDouble();
		v = dis.readDouble();
	}

	/**
	 * The learning rate reduces updates overshooting.
	 *
	 * @param learningRate the learning rate
	 */
	@SuppressWarnings("unused")
	public static void init(double learningRate) {
		Adam.learningRate = learningRate;
	}

	@SuppressWarnings("unused")
	public static void init(double beta1, double beta2, double epsilon, double learningRate) {
		Adam.beta1 = beta1;
		Adam.beta2 = beta2;
		Adam.epsilon = epsilon;
		Adam.learningRate = learningRate;
	}

	public double update(double gradient) {
		t++;

		m = beta1 * m + (1 - beta1) * gradient;
		double mt = m / (1 - Math.pow(beta1, t));
		v = beta2 * v + (1 - beta2) * Math.pow(gradient, 2);
		double vt = v / (1 - Math.pow(beta2, t));
		return -learningRate * (mt / (Math.sqrt(vt) + epsilon));
	}

	/**
	 * Imports parameters from an input stream.
	 *
	 * @param dis the input stream
	 */
	static void importParameters(DataInputStream dis) throws IOException {
		beta1 = dis.readDouble();
		beta2 = dis.readDouble();
		epsilon = dis.readDouble();
		learningRate = dis.readDouble();
	}

	/**
	 * Exports parameters to an output stream.
	 *
	 * @param dos the output stream
	 */
	static void exportParameters(DataOutputStream dos) throws IOException {
		dos.writeDouble(beta1);
		dos.writeDouble(beta2);
		dos.writeDouble(epsilon);
		dos.writeDouble(learningRate);
	}

	public void export(DataOutputStream dos) throws IOException {
		dos.writeInt(t);
		dos.writeDouble(m);
		dos.writeDouble(v);
	}
}