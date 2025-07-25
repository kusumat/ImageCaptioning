import matplotlib.pyplot as plt

def plot_metrics(history, save_path_prefix):
    epochs = list(range(1, len(history['loss']) + 1))

    plt.figure()
    plt.plot(epochs, history['loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(f"{save_path_prefix}_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history['bleu'], label='BLEU')
    plt.plot(epochs, history['rouge'], label='ROUGE')
    plt.plot(epochs, history['meteor'], label='METEOR')
    plt.plot(epochs, history['cosine'], label='Cosine Sim')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Captioning Metrics')
    plt.legend()
    plt.savefig(f"{save_path_prefix}_metrics.png")
    plt.close()
