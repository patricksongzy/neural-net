package neuralnet.costs;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class SparseCrossEntropyTest {
	private final Cost SPARSE_CE = CostType.SPARSE_CROSS_ENTROPY;

	@Test
	void cost() {
		float cost = SPARSE_CE.cost(new float[]{0.2f, 0.6f, 0.1f, 0.0f, 0.1f}, new float[]{2});

		assertEquals(2.30258509299f, cost, 1e-8f);
	}

	@Test
	void derivative() {
		// this is just a general test
		float[] delta = SPARSE_CE.derviativeSoftmax(new float[]{0.2f, 0.6f, 0.1f, 0.0f, 0.1f}, new float[]{2}, 1);

		assertArrayEquals(new float[]{0.2f, 0.6f, -0.9f, 0.0f, 0.1f}, delta);
	}
}