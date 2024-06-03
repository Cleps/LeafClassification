import numpy as np
from sklearn import metrics
import itertools
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score


class Evaluation:
    def __init__(self):
        pass

        
# Calculando a acurácia para cada classe

    def evaluate(self, y_val, preds):
        print('--------------- ACCURACY ----------------')
        class_accuracy = {}
        for i, class_name in enumerate(['Healthy', 'Rust', 'Miner', 'Cercospora', 'Phoma']):
            class_labels = y_val[:, i]
            class_preds = preds[:, i]
            class_accuracy[class_name] = accuracy_score(class_labels, class_preds)
            print(f"The accuracy for {class_name}: {class_accuracy[class_name]:.4f}")


        score = metrics.accuracy_score(y_val, preds)
        print("accuracy:   %0.4f" % score)

        cm = metrics.confusion_matrix(y_val.argmax(axis=1), preds.argmax(axis=1))
        self.plot_confusion_matrix(cm, classes=['Healthy', 'Rust', 'Miner', 'Cercospora', 'Phoma'])

# Calculando a precisão para cada classe

        print('--------------- PRECISION ----------------')
        class_precision = {}
        for i, class_name in enumerate(['Healthy', 'Rust', 'Miner', 'Cercospora', 'Phoma']):
            class_labels = y_val[:, i]
            class_preds = preds[:, i]
            class_precision[class_name] = precision_score(class_labels, class_preds)
            print(f"The precision for {class_name}: {class_precision[class_name]:.4f}")

# Calculando a precisão média

        mean_precision = sum(class_precision.values()) / len(class_precision)
        print(f"The mean precision across all classes: {mean_precision:.4f}")


    def plot_confusion_matrix(self, cm, classes,
                                normalize=False,
                                title='Confusion matrix',
                                cmap=plt.cm.Blues):

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, cm[i, j],
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig('./output/confusionMatrix')
            print('confusion matrix saved in ./output/')
