import torch
import tqdm
from cat_finder.training import losses, utils


def evaluate(
    device,
    net,
    valloader,
    wandb,
    coord_dim=2,
):
    net.eval().to(device)
    with torch.no_grad():
        val_total_loss = 0
        batch_number = 0
        loss_list = torch.zeros(8).to(device)  #
        for batch in valloader:
            batch = batch.to(device, non_blocking=True)
            pred = net(batch.x, batch.batch)

            (
                reploss,
                attloss,
                betaloss,
                suppressloss,
                ploss,
                vloss,
                closs,
                _,
                _,
                _,
                _,
            ) = losses.average_gnn_loss(
                batch,
                pred,
                coord_dim=coord_dim,
                device=device,
            )
            loss = reploss + attloss + betaloss + suppressloss + ploss + vloss + closs

            val_total_loss += loss
            loss_list += torch.tensor(
                [loss, reploss, attloss, betaloss, suppressloss, ploss, vloss, closs],
                device=device,
            )
            batch_number += 1
    return_loss = val_total_loss / batch_number

    log_loss = loss_list.cpu().numpy() / batch_number
    wandb.log(
        {
            "validation full loss": log_loss[0],
            "validation repulsion loss": log_loss[1],
            "validation attraction loss": log_loss[2],
            "validation beta loss": log_loss[3],
            "validation suppress noise loss": log_loss[4],
            "validation momentum loss": log_loss[5],
            "validation vertex loss": log_loss[6],
            "validation charge loss": log_loss[7],
        }
    )

    return return_loss


def train(
    device,
    net,
    epochs,
    optimizer,
    scheduler,
    dataloader,
    valloader,
    wandb,
    output_dir,
    last_trained_epoch=0,
    coord_dim=2,
):
    # Early stopping parameters
    patience = 10  # Number of epochs to wait for improvement before stopping
    best_val_loss = float("inf")  # Initialize best validation loss as infinity
    epochs_since_improvement = 0  # Counter for epochs since last improvement

    # initialize SaveBestModel class
    save_best_model = utils.SaveBestModel(output=output_dir)

    # with torch.autograd.detect_anomaly():
    for t in tqdm.tqdm(range(epochs)):
        loss_list = torch.zeros(8).to(device)  #
        nbtach = 0
        for i, batch in enumerate(dataloader):
            # print(batch.x)
            optimizer.zero_grad()

            net.train()
            batch = batch.to(device, non_blocking=True)

            pred = net(batch.x, batch.batch)

            (
                reploss,
                attloss,
                betaloss,
                suppressloss,
                ploss,
                vloss,
                closs,
                _,
                _,
                _,
                _,
            ) = losses.average_gnn_loss(
                batch,
                pred,
                coord_dim=coord_dim,
                device=device,
            )
            loss = reploss + attloss + betaloss + suppressloss + ploss + vloss + closs

            loss_list += torch.tensor(
                [loss, reploss, attloss, betaloss, suppressloss, ploss, vloss, closs],
                device=device,
            )
            nbtach += 1

            loss.backward()
            optimizer.step()

        log_loss = loss_list.cpu().numpy() / nbtach
        wandb.log(
            {
                "full loss": log_loss[0],
                "repulsion loss": log_loss[1],
                "attraction loss": log_loss[2],
                "beta loss": log_loss[3],
                "suppress noise loss": log_loss[4],
                "momentum loss": log_loss[5],
                "vertex loss": log_loss[6],
                "charge loss": log_loss[7],
            }
        )

        valloss = evaluate(
            device,
            net,
            valloader,
            wandb,
            coord_dim=coord_dim,
        )

        scheduler.step(valloss)
        wandb.log(
            {
                "Learning Rate": scheduler._last_lr[0],
            }
        )
        tmplr = scheduler._last_lr[0]

        # Early stopping logic
        if valloss < best_val_loss:
            best_val_loss = valloss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience and tmplr < 1e-6:
            torch.save(
                net.state_dict(),
                output_dir + f"model_early_stopping.pt",
            )
            print(
                f"Stopping early at epoch {t} due to no improvement in validation loss."
            )
            return net, t

        save_best_model(valloss, t, net, optimizer, "binary_cross_entropy")
        wandb.save("model_checkpoint_epoch_final.pth")
        # checkpointer.step((net, scheduler))

        if t % 100 == 0:
            torch.save(
                net.state_dict(),
                output_dir + f"model_{t+last_trained_epoch}.pt",
            )

    return net, t
