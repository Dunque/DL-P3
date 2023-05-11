import matplotlib.pyplot as plt


def sample_batch(dataset):
    batch = dataset.take(1).get_single_element()
    if isinstance(batch, tuple):
        batch = batch[0]
    return batch.numpy()

def sample_batch_from_dataset(dataset, n=1):
    batch = dataset.take(n)
    if isinstance(batch, tuple):
        batch = batch[0]
    return batch.numpy()


def display(images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None):
    """
    Displays n random images from each one of the supplied arrays.
    """
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    plt.figure(figsize=size)
    for i in range(n):
        _ = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].astype(as_type), cmap=cmap)
        plt.axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()

def display_one(image):
    if image.max() > 1.0:
        image = image / 255.0
    elif image.min() < 0.0:
        image = (image + 1.0) / 2.0
        
    plt.imshow(image.astype("float32"), cmap="gray_r")
    plt.axis("off")
    plt.show()
    
def display_list(images_list):
    """
    Displays n random images from each one of the supplied arrays.
    """
    for images in images_list:
        if images.max() > 1.0:
            images = images / 255.0
        elif images.min() < 0.0:
            images = (images + 1.0) / 2.0
    
    plt.figure(figsize=(20, 3))
    for i in range(len(images_list)):
        _ = plt.subplot(1, 10, i + 1)
        plt.imshow(images_list[i].astype('uint8'), cmap="gray_r")
        plt.axis("off")

    plt.show()