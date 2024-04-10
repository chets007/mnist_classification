import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import mlflow
import pickle
import joblib

def train_model():
    # Start MLflow run
    mlflow.set_experiment("MNIST")
    with mlflow.start_run():
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
        train_images, test_images = train_images / 255.0, test_images / 255.0
        
        model = models.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10)
        ])

        # Compile the model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # Log parameters
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("dropout_rate", 0.2)

        # Training the model
        history = model.fit(train_images, train_labels, epochs=5)

        # Evaluate the model
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        print('\nTest accuracy:', test_acc)
        
        # Log metrics
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        # Log the model
        mlflow.tensorflow.log_model(model, "mnist_model")

        model.save('mnist_model.keras')

if __name__ == "__main__":
    train_model()