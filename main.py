

import time
import traceback
import torch

import config
import data_model
import utils

from utils import save_results
from data_model.epochs import train_one_epoch, validate_one_epoch


# Training settings
MAX_EPOCHS = 60
TEST_INTERVAL = 30


def main():
    print("Starting Animal Detection training...")

    # # Create a new training folder
    # if not utils.continue_training(enabled=False):
    #     config.CURRENT_PATH = utils.create_current_folder()

    # utils.setup_env()
    
    # Load the previously saved model weights.
    checkpoint_path = (
        r".\!Results\11-10_16-46-08\model.pth"
    )

    print("Loading previous model weights...")
    utils.load_model(checkpoint_path)

    # Create a NEW results folder so the old checkpoint is preserved.
    config.CURRENT_PATH = utils.create_current_folder()

    utils.setup_env()

    print("Loaded previous model weights successfully.")
    print("New training results will be saved to:", config.CURRENT_PATH)


    print("\n========== CONFIGURATION ==========")
    print("Device:", config.DEVICE)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Maximum epochs:", MAX_EPOCHS)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU. Training may be slower.")

    # Verify dataloaders
    print("\n========== DATASET ==========")
    print("Training batches:", len(data_model.train_loader))
    print("Validation batches:", len(data_model.valid_loader))

    if len(data_model.train_loader) == 0:
        raise RuntimeError(
            "Training dataloader is empty. Check your dataset paths."
        )

    if len(data_model.valid_loader) == 0:
        raise RuntimeError(
            "Validation dataloader is empty. Check your dataset paths."
        )

    max_score = float("-inf")
    start_time = time.time()
    interrupted = False

    try:
        while config.EPOCH_COUNT < MAX_EPOCHS:
            epoch = config.EPOCH_COUNT

            print(
                f"\n========== Epoch {epoch + 1}/{MAX_EPOCHS} =========="
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Training
            print("\nTraining started...")
            train_logs = train_one_epoch(data_model.train_loader)
            print("Training results:", train_logs)

            # Validation
            print("\nValidation started...")
            valid_logs = validate_one_epoch(data_model.valid_loader)
            print("Validation results:", valid_logs)

            if not isinstance(valid_logs, dict):
                raise RuntimeError(
                    "Validation must return a dictionary."
                )

            if "iou_score" not in valid_logs:
                raise KeyError(
                    "The validation results do not contain 'iou_score'. "
                    "Check data_model/epochs.py."
                )

            score = float(valid_logs["iou_score"])

            if not torch.isfinite(torch.tensor(score)):
                raise RuntimeError(
                    f"Validation IoU is invalid: {score}"
                )

            
            

            # Save best model
            if score > max_score:
                max_score = score
                utils.save_model()
                print(f"Best model saved. Validation IoU: {score:.4f}")

            # Adjust learning rate
            if epoch + 1 == 30:
                config.optimizer.param_groups[0]["lr"] = 1e-5
                print("Learning rate reduced to 1e-5.")

            elif epoch + 1 == 200:
                config.optimizer.param_groups[0]["lr"] = 1e-6
                print("Learning rate reduced to 1e-6.")

            config.EPOCH_COUNT += 1

    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user.")

    except Exception:
        print("\nTraining failed:")
        traceback.print_exc()
        raise

    finally:
        elapsed = time.time() - start_time
        config.ELAPSED_TIME = elapsed

        print(f"\nTraining time: {elapsed / 60:.2f} minutes")
        print("Epochs completed:", config.EPOCH_COUNT)
        print("Best validation IoU:", max_score)

    # Final evaluation after completion or manual interruption
    if interrupted or config.EPOCH_COUNT >= MAX_EPOCHS:
        print("\nRunning final test evaluation...")
        test_log = data_model.evaluate_test_data()
        print("Final test results:", test_log)

        config.save_stats(test_log)

        if (
            hasattr(data_model, "test_visualize")
            and hasattr(data_model, "test_dataset")
        ):
            save_results(
                data_model.test_visualize,
                data_model.test_dataset,
                count=-1
            )

        print("\nTraining finished.")


if __name__ == "__main__":
    main()
