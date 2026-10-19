"""Latih ulang model final tim (hasil Random Search Keras Tuner, Trial #17).

Mereproduksi bagian 8 notebook "Hyperparameter Tuning CNN pada CIFAR-10":
  - Conv(32,3x3) -> Pool -> Conv(64,3x3) -> Pool -> Conv(128,3x3)
    -> Flatten -> Dense 128 -> Dropout 0.7 -> Dense 10 softmax (356.810 parameter)
  - SGD (momentum 0.9, Nesterov), learning rate 0.0037566671460548032
  - batch 32, maks 40 epoch, EarlyStopping val_accuracy patience 5
  - tanpa augmentasi, normalisasi piksel / 255
  - split train/val 90/10 stratified, seed 42

Jika file cnn_best_tuned.keras dari Colab tersedia, cukup salin ke
model/model.keras; skrip ini tidak perlu dijalankan.

Jalankan:  python train_final.py
"""
import json

import keras
import numpy as np
import tensorflow as tf

from train_temp import ROOT, load_split

SEED = 42
NUM_CLASSES = 10
BEST_HP = {
    "learning_rate": 0.0037566671460548032,
    "optimizer": "sgd",
    "kernel_size": 3,
    "base_filters": 32,
    "dropout": 0.7,
    "augment": False,
    "batch_size": 32,
    "epochs": 40,
}


def stratified_split(x, y, val_frac, seed):
    """Pengganti sklearn train_test_split(stratify=y): val_frac dari tiap kelas."""
    rng = np.random.default_rng(seed)
    val_idx = []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        val_idx.extend(idx[: int(round(len(idx) * val_frac))])
    mask = np.zeros(len(y), dtype=bool)
    mask[val_idx] = True
    return x[~mask], x[mask], y[~mask], y[mask]


def build_model(hp):
    k = (hp["kernel_size"], hp["kernel_size"])
    base = hp["base_filters"]
    model = keras.Sequential(name="cnn_tuned")
    model.add(keras.Input(shape=(32, 32, 3)))
    model.add(keras.layers.Conv2D(base, k, activation="relu"))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(base * 2, k, activation="relu"))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(base * 4, k, activation="relu"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(hp["dropout"]))
    model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=hp["learning_rate"], momentum=0.9, nesterov=True),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    keras.utils.set_random_seed(SEED)

    x_all, y_all = load_split("train")
    x_test, y_test = load_split("test")
    x_train, x_val, y_train, y_val = stratified_split(x_all, y_all, 0.10, SEED)
    # Normalisasi HARUS sama dengan inference.py (÷255).
    x_train, x_val, x_test = (a.astype("float32") / 255.0 for a in (x_train, x_val, x_test))
    to_oh = lambda y: keras.utils.to_categorical(y, NUM_CLASSES)

    model = build_model(BEST_HP)
    model.summary()
    history = model.fit(
        x_train,
        to_oh(y_train),
        validation_data=(x_val, to_oh(y_val)),
        batch_size=BEST_HP["batch_size"],
        epochs=BEST_HP["epochs"],
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True
            )
        ],
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(x_val, to_oh(y_val), verbose=0)
    test_loss, test_acc = model.evaluate(x_test, to_oh(y_test), verbose=0)
    print(f"Val accuracy: {val_acc:.4f} | Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    out = ROOT / "model" / "model.keras"
    out.parent.mkdir(exist_ok=True)
    model.save(out)
    print(f"Model disimpan ke {out}")

    metrics = {
        "model": "cnn_tuned (Random Search Trial #17)",
        "hyperparameters": BEST_HP,
        "val_accuracy": round(float(val_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
        "epochs_run": len(history.history["loss"]),
        "params": model.count_params(),
        "tensorflow_version": tf.__version__,
        "colab_reference": {"val_accuracy": 0.7372, "test_accuracy": 0.7259, "macro_f1": 0.7260},
    }
    (ROOT / "model" / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
