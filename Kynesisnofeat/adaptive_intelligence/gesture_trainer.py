import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'emnist_model.h5')

def train_and_save_model():
    print("Downloading EMNIST Letters dataset and training CNN (~2 mins)...")
    import tensorflow_datasets as tfds

    (ds_train, ds_test), ds_info = tfds.load(
        'emnist/letters',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        # Transpose image to fix EMNIST orientation
        image = tf.transpose(image, [1, 0, 2])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label - 1

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache().shuffle(ds_info.splits['train'].num_examples).batch(128).prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128).cache().prefetch(tf.data.AUTOTUNE)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(26, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(ds_train, epochs=2, validation_data=ds_test)

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def load_or_train_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    return tf.keras.models.load_model(MODEL_PATH)

def predict_character(model, image_28x28):
    """
    Predict a character from a 28x28 numpy array (canvas drawing).
    Ensure the image is a 28x28 numpy array (values 0-255).
    """
    img = image_28x28.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    # the canvas needs to match the orientation of transposed EMNIST
    pred_idx = np.argmax(model.predict(img, verbose=0))
    return chr(pred_idx + ord('A'))

if __name__ == "__main__":
    _ = load_or_train_model()
