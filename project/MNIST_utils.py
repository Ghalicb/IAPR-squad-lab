import numpy as np

import gzip

def extract_data(filename, image_shape, image_number):
    """Extract MNIST images from the corresponding compressed file
    Parameters
    ----------
    filename : str
        Filepath of file containing MNIST data
    image_shape : (int, int)
        Tuple describing the shape of the image
    image_number : int
        Number of images in the dataset
    Returns
    -------
    np.ndarray
        Numpy array containing the MNIST images in the file
    """
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(np.prod(image_shape) * image_number)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(image_number, image_shape[0], image_shape[1])
    return data


def extract_labels(filename, image_number):
    """Extract MNIST labels from the corresponding compressed file
    Parameters
    ----------
    filename : str
        Filepath of file containing MNIST labels
    image_number : int
        Number of images in the dataset
    Returns
    -------
    np.ndarray
        Numpy array containing the MNIST labels in the file
    """
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * image_number)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels
