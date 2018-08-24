package neuralnet;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Line2D;
import java.util.ArrayList;
import java.util.List;

class Plot extends JPanel {
	private static final Font FONT = new Font("Ubuntu Mono", Font.PLAIN, 16);
	private StringBuilder progress;
	private final List<double[]> POSITIONS = new ArrayList<>();
	private double[] last;
	private double batchAmount;
	private int iteration, epochs, batch, batchPosition;
	private double top, xScale, yScale;

	void init(int epochs, double batchAmount) {
		this.epochs = epochs;
		this.batchAmount = batchAmount;
	}

	void update(double x, double y, int iteration, int batch, int batchPosition) {
		xScale = getWidth() / x;

		if (y > top)
			top = y;

		yScale = getHeight() / top;

		this.iteration = iteration;
		this.batch = batch;
		this.batchPosition = batchPosition;
		double percent = (batchAmount * (iteration - 1) + batch) / (batchAmount * epochs) * 100;
		progress = new StringBuilder("[");
		for (int i = 0; i <= 50; i++) {
			if (i <= percent / 2)
				progress.append("#");
			else
				progress.append("-");
		}

		progress.append("]");
		progress.append(String.format(" %.02f%%", percent));

		last = new double[]{x, y};
		POSITIONS.add(last);
		repaint();
	}

	public void paintComponent(Graphics g) {
		super.paintComponent(g);

		Graphics2D g2 = (Graphics2D) g;
		g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
		g2.setFont(FONT);
		g2.setColor(Color.BLUE);

		List<double[]> average = new ArrayList<>();
		if (POSITIONS.size() > 1) {
			g2.drawString("Epoch: " + iteration + " / " + epochs, getWidth() - 500, 40);
			g2.drawString("Batch: " + batch + " / " + batchAmount + " [" + batchAmount * epochs + "] : " + batchPosition, getWidth() -
					500, 60);
			g2.drawString("Cost: " + last[1], getWidth() - 500, 80);
			g2.drawString(progress.toString(), getWidth() - 500, 100);

			double sum = 0;
			int amount = 0, total = (int) (yScale / xScale);

			for (int i = 1; i < POSITIONS.size(); i++) {
				double[] from = POSITIONS.get(i - 1);
				double[] to = POSITIONS.get(i);
				g2.draw(new Line2D.Double(from[0] * xScale, getHeight() - from[1] * yScale, to[0] * xScale, getHeight() - to[1] * yScale));

				sum += to[1];
				amount++;

				if (total != 0) {
					if ((i - 1) % total == 0) {
						average.add(new double[]{((i - amount) + (i - 1)) / 2.0, sum / amount});
						sum = 0;
						amount = 0;
					}
				}
			}

			if (amount != 0) {
				average.add(new double[]{((POSITIONS.size() - 1 - amount) + (POSITIONS.size() - 2)) / 2.0, sum / amount});
			}
		}

		g2.setColor(Color.RED);
		for (int i = 1; i < average.size(); i++) {
			double[] from = average.get(i - 1);
			double[] to = average.get(i);
			g2.draw(new Line2D.Double(from[0] * xScale, getHeight() - from[1] * yScale, to[0] * xScale, getHeight() - to[1] * yScale));
		}
	}
}