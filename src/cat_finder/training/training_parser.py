from argparse import ArgumentParser


def training_parser():
    ap = ArgumentParser()
    ap.add_argument("--config", default="config.yaml", help="train config to load")
    ap.add_argument("--run", default=None, help="Run name for training")
    ap.add_argument(
        "--input_dir",
        default=None,
        type=str,
        help="directory where the input for training and validation samples are",
    )
    ap.add_argument("--batch", default=None, type=int, help="batch size")
    ap.add_argument(
        "--early_stopping", default=None, type=int, help="if early stopping"
    )
    ap.add_argument("--pretrained", default=None, type=str, help="use pretrained model")
    ap.add_argument("--pretrained_model", default=None, type=str, help="path to pretrained model")
    ap.add_argument(
        "--epochs", default=None, type=int, help="number of epochs to train on"
    )
    ap.add_argument("--output_dir", default=None, type=str, help="path to output")
    ap.add_argument("--logging", default=None, type=str, help="path to logging")
    return ap
