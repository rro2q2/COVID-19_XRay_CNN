from constants import *


class ConvNet:
    def __init__(self, train_aug, train_X, test_X, train_y, test_y):
        self.train_aug = train_aug
        self.train_X = train_X
        self.test_X = test_X
        self.train_y = train_y
        self.test_y = test_y

    def run_model(self):
        # (3) Create a sequential model
        model = Sequential()

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(224, 224, 3),))
        model.add(AveragePooling2D(pool_size=(4, 4)))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        model.add(AveragePooling2D(pool_size=(4, 4)))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        model.add(AveragePooling2D(pool_size=(4, 4)))

        # Passing it to a dense layer
        model.add(Flatten())
        # 1st Dense Layer
        model.add(Dense(64, input_shape=(224 * 224 * 3,)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.5))

        # Output Layer
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        # compile our model
        print("[INFO] compiling model...")

        opt = Adam(lr=learning_rate, decay=learning_rate / epochs)
        model.compile(loss="binary_crossentropy", optimizer=opt,
                      metrics=["accuracy"])

        # Record time to train model
        start = timeit.default_timer()
        H = model.fit_generator(
            self.train_aug.flow(self.train_X, self.train_y, batch_size=batch_size),
            steps_per_epoch=len(self.train_X) // batch_size,
            validation_data=(self.test_X, self.test_y),
            validation_steps=len(self.test_X) // batch_size,
            epochs=epochs)
        stop = timeit.default_timer()
        print("Overall training time for ConvNet:", stop - start)

        preds = model.evaluate(self.test_X, self.test_y)
        print("Loss = " + str(preds[0]))
        print("Test Accuracy = " + str(preds[1]))

        # make predictions on the testing set
        print("[INFO] evaluating network...")
        predIdxs = model.predict(self.test_X, batch_size=batch_size)

        # for each image in the testing set we need to find the index of the
        # label with corresponding largest predicted probability
        predIdxs = np.argmax(predIdxs, axis=1)

        # show a nicely formatted classification report
        print(classification_report(self.test_y.argmax(axis=1), predIdxs,
                                    target_names=['negative', 'positive']))

        # compute the confusion matrix and and use it to derive the raw
        # accuracy, sensitivity, and specificity
        cm = confusion_matrix(self.test_y.argmax(axis=1), predIdxs)
        total = sum(sum(cm))
        acc = (cm[0, 0] + cm[1, 1]) / total
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        # show the confusion matrix, accuracy, sensitivity, and specificity
        y_true = pd.Series(self.test_y.argmax(axis=1), name='Actual')
        y_pred = pd.Series(predIdxs, name='Predicted')
        conf_matrix = pd.crosstab(y_true, y_pred)
        print("Confusion Matrix:\n", conf_matrix)

        print("acc: {:.4f}".format(acc))
        print("sensitivity: {:.4f}".format(sensitivity))
        print("specificity: {:.4f}".format(specificity))

        self.plot_confusion_matrix(conf_matrix)

        return H

    def plot_confusion_matrix(self, df_confusion, cmap=plt.cm.Blues):
        plt.matshow(df_confusion, cmap=cmap)  # imshow
        plt.title("ConvNet Confusion Matrix")
        plt.colorbar()
        plt.ylabel(df_confusion.index.name)
        plt.xlabel(df_confusion.columns.name)
        plt.savefig('../output/convnet_cm.png')
