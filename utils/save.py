# utils/save.py

import os
import torch

def save_best_model(model, optimizer, scheduler, epoch, score, best_score, path="checkpoints"):
    """
    Save the model if it has the best score so far.
    Removes previous best model to save disk space.
    """

    if score > best_score:
        # Create checkpoints directory if not exists
        os.makedirs(path, exist_ok=True)

        # Remove old best model if exists
        for filename in os.listdir(path):
            if filename.startswith("best_model"):
                os.remove(os.path.join(path, filename))

        # Save new best model
        save_path = os.path.join(path, f"best_model.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'score': score
        }, save_path)

        print(f"[INFO] ✅ Best model saved at epoch {epoch} with score {score:.4f}")
        return score  # update best_score

    return best_score
