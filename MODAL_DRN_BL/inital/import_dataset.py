import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import threading


def read_label(callback):
    """
    Read the label data from the meta.txt file and pass it to the callback function.
    """
    meta = np.loadtxt("../dataset/meta.txt", delimiter=" ", dtype=str)
    labels = meta.astype(np.float32)
    callback(labels)


def read_data(callback):
    """
    Read the input data from multiple files and pass the concatenated result to the callback function.
    """
    # Read the first two files separately
    first = np.loadtxt("../dataset/dy_sample/x_vector1.txt", delimiter=' ', dtype=str).astype(np.float32)
    first = np.expand_dims(first, axis=0)
    second = np.loadtxt("../dataset/dy_sample/x_vector2.txt", delimiter=' ', dtype=str).astype(np.float32)
    second = np.expand_dims(second, axis=0)

    # Concatenate the first two arrays
    X = np.concatenate((first, second), axis=0)

    # Read and concatenate the remaining files
    for sample in range(3, 103059):
        x_vector = np.loadtxt(f"../dataset/dy_sample/x_vector{sample}.txt", delimiter=' ', dtype=str).astype(np.float32)
        x_vector = np.expand_dims(x_vector, axis=0)
        X = np.concatenate((X, x_vector), axis=0)

        # Print progress every 10000 samples
        if sample % 10000 == 0:
            print(f"This is the data import of one iteration {sample}")

    callback(X)


def X_callback(X):
    global x_data
    x_data = X


def labels_callback(labels):
    global x_labels
    x_labels = labels


def import_dataset():
    """
    Import the dataset using multithreading for reading data and labels, and split into training, validation, and test sets.
    """
    # Create thread objects
    thread_one = threading.Thread(target=read_label, args=(labels_callback,))
    thread_two = threading.Thread(target=read_data, args=(X_callback,))

    # Start the threads
    thread_one.start()
    thread_two.start()

    # Wait for both threads to finish
    thread_one.join()
    thread_two.join()

    # Split the data into training, validation, and test sets
    x_train, x_test, train_labels, test_labels = train_test_split(
        x_data, x_labels, test_size=0.2, random_state=42)
    x_train, x_val, train_labels, val_labels = train_test_split(
        x_train, train_labels, test_size=0.25, random_state=42)

    # Print the shapes of the datasets
    print("Training set shape:", x_train.shape, np.array(train_labels).shape)
    print("Validation set shape:", x_val.shape, np.array(val_labels).shape)
    print("Test set shape:", x_test.shape, np.array(test_labels).shape)

    return x_train, x_val, x_test, train_labels, val_labels, test_labels


def data_main():
    """
    Main function to import the dataset and return the splits.
    """
    x_train, x_val, x_test, train_labels, val_labels, test_labels = import_dataset()
    return x_train, x_val, x_test, train_labels, val_labels, test_labels


if __name__ == '__main__':
    data_main()
