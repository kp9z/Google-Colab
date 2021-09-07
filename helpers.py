import pandas as pd
import matplotlib.pyplot as plt

def plot_loss_curve(model_history):
    """
    Plot loss curve of function & metrics on a seperate plot
    :param model_history: a
    :return: Plots of accuracy & metrics
    """
    df = pd.DataFrame(model_history.history)
    epochs = range(1,len(df)+1)

    plt.figure(figsize=(10,5))

    # Setup loss curve
    plt.subplot(121)
    plt.plot(epochs, df['loss'],label = 'Train loss')
    plt.plot(epochs, df['val_loss'], label = 'Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    #Setup accuracy_curve
    plt.subplot(122)
    plt.plot(epochs, df['accuracy'],label = 'Train accuracy')
    plt.plot(epochs, df['val_accuracy'],label = 'Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Accuracy')

    plt.legend()
    plt.show()


