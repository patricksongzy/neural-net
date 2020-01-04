package neuralnet.layers.graph;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class Node {
	// the amount of nodes which have been created
	private static long nodeCount = 0;

	private final long nodeID = nodeCount++;
	// the children of the node are used as inputs
	protected Node[] children;
	// the consumers of the node use its output
	private List<Node> consumers = new ArrayList<>();

	protected Node(Node... children) {
		this.children = children;
	}

	public Node[] getChildren() {
		return children;
	}

	public void setChildren(Node[] children) {
		this.children = children;

		Arrays.stream(children).forEach(child -> child.getConsumers().add(this));
	}

	public List<Node> getConsumers() {
		return consumers;
	}

	public void setConsumers(List<Node> consumers) {
		this.consumers = consumers;
	}
}
