import torch
import yaml


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, best_valid_loss=float("inf"), output="outputs/"):
        self.best_valid_loss = best_valid_loss
        self.output = output

    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                f"{self.output}/best_model.pt",
            )


def get_config(args):
    """Load default configs followed by user configs and populate dataset tags.
    Args:
        cfg_path(str or Path): Path to user config yaml
        model(optional): model parameters (overwrites loaded config)
        dataset(optional): Information on dataset (overwrites loaded config)
        output( optional): Output information for training and processing (overwrites loaded config)
        train/val(optional): Training and validation options(overwrites loaded config)
    Returns:
        dict: The loaded training configuration dictionary
    """

    # Read config file
    with open(args.config) as cfg_path:
        config = yaml.safe_load(cfg_path)
    print(config)

    # Overwrite training config if specified in command line
    # input dir
    if args.input_dir is not None:
        config["dataset"]["input_dir"] = args.input_dir
        config["dataset"]["val_input_dir"] = args.input_dir
    # run name
    if args.run is not None:
        config["output"]["run_name"] = args.run
    # run name
    if args.output_dir is not None:
        config["output"]["output_dir"] = args.output_dir
    # run name
    if args.logging is not None:
        config["output"]["logging"] = args.logging

    # number of epochs
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    # batch size
    if args.batch is not None:
        config["train"]["batch_size"] = int(args.batch)
        config["val"]["batch_size"] = int(args.batch)
    # early stopping
    if args.early_stopping is not None:
        config["train"]["early_stopping"] = int(args.early_stopping)
    # pretrained model
    if args.pretrained is not None:
        config["train"]["pretrained"] = args.pretrained
    if args.pretrained_model is not None:
        config["train"]["pretrained_model"] = args.pretrained_model

    return config
