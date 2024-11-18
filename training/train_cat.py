import os
import torch
from cat_finder.training import gnn_model, datatools, utils
import wandb
from torch_geometric.loader import DataLoader
import yaml
from cat_finder.training.training_parser import training_parser
from cat_finder.training.training_functions import train


if __name__ == "__main__":
    ap = training_parser()
    args = ap.parse_args()

    config = utils.get_config(args)
    run = config["output"]["run_name"]

    outputdir = config["output"]["output_dir"] + f"/{run}/"
    os.makedirs(outputdir, exist_ok=True)

    dataset_params = {
        "features": config["dataset"]["input_features"],
        "truth": config["dataset"]["truth"],
        "scaling": config["dataset"]["scaling"],
        "clipping": config["dataset"]["clipping"],
    }

    gd = datatools.CDCDataset(
        n_events=config["dataset"]["samples_per_file"],
        evtname=config["dataset"]["evt_type"],
        sampledir=config["dataset"]["input_dir"],
        torch_dir=config["dataset"]["input_dir"],
        **dataset_params,
    )
    print(gd)

    gd_val = datatools.CDCDataset(
        n_events=config["dataset"]["val_samples_per_file"],
        evtname=config["dataset"]["val_type"],
        torch_dir=config["dataset"]["val_input_dir"],
        sampledir=config["dataset"]["val_input_dir"],
        **dataset_params,
    )
    print(gd_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set input dim to feature size
    net = gnn_model.CDCNet(
        input_dim=len(config["dataset"]["input_features"]),
        k=config["model"]["k"],
        nblocks=config["model"]["blocks"],
        coord_dim=config["model"]["coord_dim"],
        space_dimensions=config["model"]["space_dimensions"],
        dim1=config["model"]["dim1"],
        dim2=config["model"]["dim2"],
        momentum=config["model"]["momentum"],
    ).float()
    print(net)

    if config["train"]["pretrained"]:
        loaded_model = torch.load(
            config["train"]["pretrained_model"],
            map_location=torch.device(device),
        )
        net.load_state_dict(loaded_model["model_state_dict"])

    # get Dataloader for training and validation
    train_data = DataLoader(
        gd,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        pin_memory=True,
    )

    val_data = DataLoader(
        gd_val,
        batch_size=config["val"]["batch_size"],
        shuffle=True,
        num_workers=config["val"]["num_workers"],
        pin_memory=True,
    )

    # save model config in output
    with open(outputdir + "/config.yaml", "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    # set up wandb logging for the training
    wandb.init(project="CAT_Finder", config=config)

    # data parallel
    net = torch.nn.DataParallel(net)
    net = net.to(device)
    # set optimizer and scheduling
    learning_rate = config["train"]["learning_rate"]
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        verbose=True,
        patience=30,
        factor=0.5,
        cooldown=10,
    )

    trained_net, final_epoch = train(
        device=device,
        net=net,
        epochs=config["train"]["epochs"],
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=train_data,
        valloader=val_data,
        wandb=wandb,
        output_dir=outputdir,
        coord_dim=config["model"]["coord_dim"],
    )

    torch.save(trained_net.state_dict(), outputdir + f"model_{final_epoch}_epoch.pt")

    print(f"Stopped training after {final_epoch} epochs")
