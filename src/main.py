from constants import *
import timeit

## IMPORT CLASS MODELS ##
import vgg16 as vgg
import alexnet as an
import convnet as cn

def process_image(dataset: list, image_list: list) -> np.ndarray:
    """
    Pre-processing each image in the dataset1 (positive/negative)
    :param dataset: the directory that holds the list of images
    :param image_list: an array in which the preprocessed images will be stored
    :return: np.array of image data
    """
    for image in dataset:
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image_list.append(image)

    return np.array(image_list)


def process_labels(dataset: list, label_list: list) -> np.ndarray:
    """
    Abstracts the labels from the images filenames
    :param dataset: the directory that holds the list of images
    :param label_list: an array in which the preprocessed labels wil be stored
    :return:  np.array of label data
    """
    for label in dataset:
        label = label.split(os.path.sep)[-2]
        label_list.append(label)

    return np.array(label_list)


def one_hot_encoding(labels: np.ndarray) -> np.ndarray:
    """
    Takes in the labels list of strings and performs one-hot encoding
    that to return a vectorized binary output
    :param labels: numpy.ndarray list of strings
    :return: np.ndarray for np.float vector encodings
    """
    counts = Counter(labels)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 0)}
    int_encoded = [vocab_to_int[ii] for ii in labels]
    one_hot = list()
    for value in int_encoded:
        word = [0 for _ in range(len(vocab))]
        word[value] = 1
        one_hot.append(word)

    return np.array(one_hot, dtype=np.int)


def main():
    dataset = list(paths.list_images('../dataset0'))
    inputs = []
    labels = []
    inputs = process_image(dataset, inputs) / 255.0  # scaling between [0, 1]
    labels = process_labels(dataset, labels)

    one_hot = one_hot_encoding(labels)

    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    trainX, testX, trainY, testY = train_test_split(inputs, one_hot, test_size=0.20, shuffle=True)

    trainAug = ImageDataGenerator(
        rotation_range=15,
        fill_mode="nearest")

    # load the VGG16 network, ensuring the head FC layer sets are left
    # off
    vgg16_init = VGG16(weights="imagenet", include_top=False,
                      input_tensor=Input(shape=(224, 224, 3)))

    # Initialize models
    vgg_model = vgg.VGG16(vgg16_init, trainAug, trainX, testX, trainY, testY)
    alex_model = an.AlexNet(trainAug, trainX, testX, trainY, testY)
    conv_model = cn.ConvNet(trainAug, trainX, testX, trainY, testY)

    # Run models
    conv_H = conv_model.run_model()
    alex_H = alex_model.run_model()
    vgg_H = vgg_model.run_model()

    # Plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.title("Training Loss on COVID-19 Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")

    plt.plot(np.arange(0, epochs), conv_H.history["loss"], label="ConvNet")
    plt.plot(np.arange(0, epochs), alex_H.history["loss"], label="AlexNet")
    plt.plot(np.arange(0, epochs), vgg_H.history["loss"], label="VGG16")

    plt.legend(loc="upper right")
    plt.savefig('../output/train_loss.png')


if __name__ == '__main__':
    main()
