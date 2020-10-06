import matplotlib
# Specifying the backend to be used before importing pyplot
# to avoid "RuntimeError: Invalid DISPLAY variable"
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import os
from pathlib import Path
from glob import glob

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
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)

        self.losses.append(logs.get('loss'))
        # # self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        # self.val_acc.append(logs.get('val_acc'))
        #
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            #plt.style.use("seaborn")
            if len(self.losses) >40:
                # Plot train loss, train acc, val loss and val acc against epochs passed
                plt.figure()
                plt.subplot(211)
                plt.plot(N, self.losses, '-b', label = "train_loss")
                # plt.plot(N, self.acc, label = "train_acc")
                plt.plot(N, self.val_losses, '-r', label = "val_loss")
                # plt.plot(N, self.val_acc, label = "val_acc")
                plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
                plt.xlabel("Epoch #")
                plt.ylabel("Loss/Accuracy")
                plt.legend()
                plt.subplot(212)
                plt.plot(N[30:], self.losses[30:], '-b', label="train_loss")
                # plt.plot(N, self.acc, label = "train_acc")
                plt.plot(N[30:], self.val_losses[30:], '-r', label="val_loss")
                # plt.plot(N, self.val_acc, label = "val_acc")
                # plt.title("Training Loss and Accuracy [Epoch {}] and last 30 epochs".format(epoch))
                plt.xlabel("Last 30 Epochs #")
                plt.ylabel("Loss/Accuracy")
                plt.legend()
                # Make sure there exists a folder called output in the current directory
                # or replace 'output' with whatever direcory you want to put in the plots
                plt.savefig(self.save_path + 'output_losses_Epoch-{}.png'.format(epoch))

                print("loss figure saved!")
            else:
                plt.figure()
                plt.plot(N, self.losses, '-b', label="train_loss")
                # plt.plot(N, self.acc, label = "train_acc")
                plt.plot(N, self.val_losses, '-r', label="val_loss")
                # plt.plot(N, self.val_acc, label = "val_acc")
                plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
                plt.xlabel("Epoch #")
                plt.ylabel("Loss/Accuracy")
                plt.legend()
                plt.savefig(self.save_path + 'output_losses_Epoch-{}.png'.format(epoch))

                print("loss figure saved!")
            plt.close()
            with open(self.save_path + 'train_losses_Epoch-{}.txt'.format(epoch), 'w') as f:
                for item in self.losses:
                    f.write("%s\n" % item)
            with open(self.save_path + 'val_losses_Epoch-{}.txt'.format(epoch), 'w') as f:
                for item in self.val_losses:
                    f.write("%s\n" % item)

class DeleteEarlySavedH5models(keras.callbacks.Callback):
    """
      A Logger that log average performance per `display` steps.
      """

    def __init__(self, modelSavedPath):
        # self.step = 0
        self.modelSavedPath = modelSavedPath
        # self.metric_cache = {}


    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        print("end epoch logs:", logs)
        print("save path:", self.modelSavedPath)
        saved_model_files =  glob(self.modelSavedPath + '/*.h5')
        if len(saved_model_files) >2:
            saved_model_files.sort(key=os.path.getctime)
            print("all the files:", saved_model_files)
            old_files =  saved_model_files[0:len(saved_model_files)-1]
            print("files to be deleted:", old_files)

            for file in old_files:
                os.remove(file)


