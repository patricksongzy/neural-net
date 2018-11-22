package plot;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;

public class Controller {
	@FXML
	private LineChart<Number, Number> plot;
	private XYChart.Series<Number, Number> series;

	public void initialize() {
		series = new XYChart.Series<>();
		plot.setLegendVisible(false);
		plot.setAnimated(false);
		plot.setCreateSymbols(false);
		plot.getData().add(series);
	}

	public void update(float x, float y) {
		Platform.runLater(() -> series.getData().add(new XYChart.Data<>(x, y)));
	}
}