package neuralnet.schedules;

import neuralnet.optimizers.UpdaterType;

public class CosineRestart implements Schedule {
	private UpdaterType updaterType;

	private float max, min, decay;
	private int restartInterval, restartMultiplier, warmup, keyAmount;
	private int current;

	@SuppressWarnings("unused")
	public CosineRestart(float max, float min, float decay, int restartInterval, int restartMultiplier, int warmup) {
		this.max = max;
		this.min = min;
		this.decay = decay;
		this.restartInterval = restartInterval;
		this.restartMultiplier = restartMultiplier;
		this.warmup = warmup;
	}

	public void init(UpdaterType updaterType, int batchSize, int keyAmount) {
		this.updaterType = updaterType;
		this.keyAmount = keyAmount;

		updaterType.setDecay(decay * (float) Math.sqrt((float) batchSize / (keyAmount * restartInterval)));
		updaterType.init(max);
	}

	public void increment(int s) {
		current += s;
	}

	public void endEpoch(int i) {
		if (i == warmup)
			current = 0;
	}

	public void step() {
		if (current < warmup) {
			updaterType.init((max / warmup) * ((float) current / keyAmount));
		} else if ((current / keyAmount) == restartInterval) {
			System.out.println("Restarting");

			current = 0;
			restartInterval *= restartMultiplier;

			updaterType.init(max);
		} else {
			updaterType.init(min + 0.5f * (max - min) * (1 + (float) Math.cos(((float) current / keyAmount) * Math.PI / restartInterval)));
		}
	}
}
