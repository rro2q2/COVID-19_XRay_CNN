from constants import *


class VGG16:
    def __init__(self, base_model, train_aug, train_X, test_X, train_y, test_y):
        self.base_model = base_model
        self.train_aug = train_aug
        self.train_X = train_X
        self.test_X = test_X
        self.train_y = train_y
        self.test_y = test_y

    def run_model(self):
        # construct the head of the model that will be placed on top of the
        # the base model
        head_model = self.base_model.output
        head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
        head_model = Flatten(name="flatten")(head_model)
        head_model = Dense(64, activation="relu")(head_model)
        head_model = Dropout(0.5)(head_model)
        head_model = Dense(num_classes, activation="softmax")(head_model)

        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        model = Model(inputs=self.base_model.input, outputs=head_model)

        # loop over all layers in the base model and freeze them so they will
        # *not* be updated during the first training process
        for layer in self.base_model.layers:
            layer.trainable = False

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
        print("Overall training time for VGG16:", stop - start)

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
        # show the confusion matrix, accuracy, sensitivity, and specificity
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
        plt.title("VGG16 Confusion Matrix")
        plt.colorbar()
        plt.ylabel(df_confusion.index.name)
        plt.xlabel(df_confusion.columns.name)
        plt.savefig('../output/vgg16_cm.png')
