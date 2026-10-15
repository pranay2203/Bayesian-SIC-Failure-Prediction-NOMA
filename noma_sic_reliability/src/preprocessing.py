"""Scale input features and create reproducible train/test splits."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(frame, features, test_size, seed):
    x_train, x_test, y_train, y_test = train_test_split(
        frame[features].values, frame["sic_failure"].values, test_size=test_size,
        stratify=frame["sic_failure"].values, random_state=seed)
    scaler = StandardScaler()
    return (scaler.fit_transform(x_train).astype(np.float32), scaler.transform(x_test).astype(np.float32),
            y_train.astype(np.float32), y_test.astype(np.float32), scaler)
