import numpy as np

def load_mnist(images_path, labels_path):
    with open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        images = np.frombuffer(imgpath.read(), dtype=np.uint8)
        images = images.reshape(len(labels), 784)

    return images / 255.0, labels