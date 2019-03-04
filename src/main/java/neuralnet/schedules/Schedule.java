package neuralnet.schedules;

import neuralnet.optimizers.UpdaterType;

public interface Schedule {
	void init(UpdaterType updaterType, int batchSize, int keyAmount);

	void step();

	void increment(int s);

	void endEpoch(int i);
}