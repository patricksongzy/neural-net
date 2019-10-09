package neuralnet.initializers;

@SuppressWarnings("unused")
public class Constant implements Initializer {
	private float constant;

	public Constant(float constant) {
		this.constant = constant;
	}

	public float initialize(int inputSize) {
		return constant;
	}
}
