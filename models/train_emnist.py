"""Train the EMNIST-based air-writing classifier for KINESYS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import tensorflow as tf
import tensorflow_datasets as tfds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CHAR_SIZE,
    EMNIST_BATCH_SIZE,
    EMNIST_CLASS_COUNT,
    EMNIST_CONV1_FILTERS,
    EMNIST_CONV2_FILTERS,
    EMNIST_DATASET_NAME,
    EMNIST_DENSE_UNITS,
    EMNIST_DROPOUT,
    EMNIST_EPOCHS,
    EMNIST_INPUT_CHANNELS,
    EMNIST_KERNEL_SIZE,
    EMNIST_MODEL_FILE,
    EMNIST_SHUFFLE_BUFFER,
    EMNIST_TEST_SPLIT,
    EMNIST_TRAIN_SPLIT,
    EMNIST_VALIDATION_SPLIT,
)


AUTOTUNE = tf.data.AUTOTUNE
LETTER_LABEL_OFFSET = 1
MODEL_PARENT_DEPTH = 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training."""

    parser = argparse.ArgumentParser(description="Train the KINESYS EMNIST CNN model.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=EMNIST_EPOCHS,
        help="Number of epochs to train.",
    )
    return parser.parse_args()


def preprocess_sample(sample: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
    """Normalize one EMNIST sample and align the label space to A-Z."""

    image = tf.cast(sample["image"], tf.float32) / 255.0
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    label = tf.cast(sample["label"], tf.int32) - LETTER_LABEL_OFFSET
    label = tf.one_hot(label, depth=EMNIST_CLASS_COUNT)
    return image, label


def build_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load and preprocess the EMNIST train, validation, and test datasets."""

    train_dataset = tfds.load(EMNIST_DATASET_NAME, split=EMNIST_TRAIN_SPLIT, as_supervised=False)
    validation_dataset = tfds.load(EMNIST_DATASET_NAME, split=EMNIST_VALIDATION_SPLIT, as_supervised=False)
    test_dataset = tfds.load(EMNIST_DATASET_NAME, split=EMNIST_TEST_SPLIT, as_supervised=False)

    train_dataset = (
        train_dataset
        .map(preprocess_sample, num_parallel_calls=AUTOTUNE)
        .shuffle(EMNIST_SHUFFLE_BUFFER)
        .batch(EMNIST_BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    validation_dataset = (
        validation_dataset
        .map(preprocess_sample, num_parallel_calls=AUTOTUNE)
        .batch(EMNIST_BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    test_dataset = (
        test_dataset
        .map(preprocess_sample, num_parallel_calls=AUTOTUNE)
        .batch(EMNIST_BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    return train_dataset, validation_dataset, test_dataset


def build_model() -> tf.keras.Model:
    """Create the convolutional network used for letter classification."""

    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(CHAR_SIZE, CHAR_SIZE, EMNIST_INPUT_CHANNELS)),
            tf.keras.layers.Conv2D(
                filters=EMNIST_CONV1_FILTERS,
                kernel_size=EMNIST_KERNEL_SIZE,
                activation="relu",
                padding="same",
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(
                filters=EMNIST_CONV2_FILTERS,
                kernel_size=EMNIST_KERNEL_SIZE,
                activation="relu",
                padding="same",
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(EMNIST_DENSE_UNITS, activation="relu"),
            tf.keras.layers.Dropout(EMNIST_DROPOUT),
            tf.keras.layers.Dense(EMNIST_CLASS_COUNT, activation="softmax"),
        ]
    )


def main() -> int:
    """Train the EMNIST model and save it into the configured models directory."""

    args = parse_args()
    train_dataset, validation_dataset, test_dataset = build_datasets()
    model = build_model()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    model_path = Path(EMNIST_MODEL_FILE)
    model_path.parents[MODEL_PARENT_DEPTH - 1].mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Saved model to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
