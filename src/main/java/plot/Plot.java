package plot;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;

public class Plot extends Application {
	private static Controller controller;

	public static void update(float x, float y) {
		controller.update(x, y);
	}

	public void start(Stage stage) throws IOException {
		FXMLLoader loader = new FXMLLoader();
		loader.setLocation(getClass().getResource("/plot.fxml"));
		Parent root = loader.load();
		stage.setScene(new Scene(root, 600, 400));
		stage.show();

		controller = loader.getController();
	}
}
