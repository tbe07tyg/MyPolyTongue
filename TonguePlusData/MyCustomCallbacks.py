import matplotlib
# Specifying the backend to be used before importing pyplot
# to avoid "RuntimeError: Invalid DISPLAY variable"
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

class TrainingPlotCallback(keras.callbacks.Callback):
    """
      A Logger that log average performance per `display` steps.
      """

    def __init__(self, save_path):
        # self.step = 0
        self.save_path = save_path
        # self.metric_cache = {}


    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        print("end epoch logs:", logs)
        print("save path:", self.save_path)
        # # Append the logs, losses and accuracies to the lists
        # self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        # # self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        # # self.val_acc.append(logs.get('val_acc'))
        #
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            #plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, '-b', label = "train_loss")
            # plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, '-r', label = "val_loss")
            # plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig(self.save_path + 'output_losses_Epoch-{}.png'.format(epoch))

            print("loss figure saved!")
            plt.close()
            with open(self.save_path + 'train_losses_Epoch-{}.txt'.format(epoch), 'w') as f:
                for item in self.losses:
                    f.write("%s\n" % item)
            with open(self.save_path + 'val_losses_Epoch-{}.txt'.format(epoch), 'w') as f:
                for item in self.val_losses:
                    f.write("%s\n" % item)