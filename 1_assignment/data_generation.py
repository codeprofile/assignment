import numpy as np
import pandas as pd


if __name__ == "__main__":
    # Ensure both 0 and 1 labels exist
    n_samples = 100
    downtime_labels = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

    data = {
        "Machine_ID": np.arange(1, n_samples + 1),
        "Temperature": np.random.randint(50, 120, n_samples),
        "Run_Time": np.random.randint(500, 2000, n_samples),
        "Downtime_Flag": np.random.permutation(downtime_labels)  # Shuffle to mix 0s and 1s
    }

    df = pd.DataFrame(data)
    df.to_csv("data/sample_data.csv", index=False)
