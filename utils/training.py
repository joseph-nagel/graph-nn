'''Training loop.'''

import torch


@torch.no_grad()
def accuracy(preds, targets):
    '''Calculate accuracy.'''
    return (preds == targets).sum() / len(preds)


def train(data,
          model,
          criterion,
          optimizer,
          num_epochs,
          log_every=1):
    '''Train model.'''

    train_losses = torch.zeros(num_epochs + 1, dtype=data.x.dtype)
    train_accs = torch.zeros(num_epochs + 1, dtype=data.x.dtype)

    val_losses = torch.zeros(num_epochs + 1, dtype=data.x.dtype)
    val_accs = torch.zeros(num_epochs + 1, dtype=data.x.dtype)

    # compute losses and accuracies
    model.eval()

    with torch.no_grad():
        y_pred = model(data.x, data.edge_index)

        loss = criterion(
            y_pred[data.train_mask],
            data.y[data.train_mask]
        )

        train_acc = accuracy(
            y_pred[data.train_mask].argmax(dim=1),
            data.y[data.train_mask]
        )

        val_loss = criterion(
            y_pred[data.val_mask],
            data.y[data.val_mask]
        )

        val_acc = accuracy(
            y_pred[data.val_mask].argmax(dim=1),
            data.y[data.val_mask]
        )

    train_losses[0] = loss
    train_accs[0] = train_acc

    val_losses[0] = val_loss
    val_accs[0] = val_acc

    print(
        'Initially, train loss: {:.2e}, train acc.: {:.2f}, val. loss: {:.2e}, val. acc.: {:.2f}'.format(
            loss.item(), train_acc.item(), val_loss.item(), val_acc.item()
        )
    )

    # loop over epochs
    for epoch_idx in range(num_epochs):

        # perform train step
        model.train()

        optimizer.zero_grad()

        y_pred = model(data.x, data.edge_index)

        loss = criterion(
            y_pred[data.train_mask],
            data.y[data.train_mask]
        )

        loss.backward()
        optimizer.step()

        # monitor performance
        if (epoch_idx + 1) % log_every == 0:

            model.eval()

            with torch.no_grad():
                train_acc = accuracy(
                    y_pred[data.train_mask].argmax(dim=1),
                    data.y[data.train_mask]
                )

                y_pred = model(data.x, data.edge_index) # preds may differ in eval mode

                val_loss = criterion(
                    y_pred[data.val_mask],
                    data.y[data.val_mask]
                )

                val_acc = accuracy(
                    y_pred[data.val_mask].argmax(dim=1),
                    data.y[data.val_mask]
                )

                train_losses[epoch_idx + 1] = loss.detach()
                train_accs[epoch_idx + 1] = train_acc

                val_losses[epoch_idx + 1] = val_loss
                val_accs[epoch_idx + 1] = val_acc

            # print summary
            print(
                'Epoch: {:d}, train loss: {:.2e}, train acc.: {:.2f}, val. loss: {:.2e}, val. acc.: {:.2f}'.format(
                    epoch_idx + 1, loss.detach().item(), train_acc.item(), val_loss.item(), val_acc.item()
                )
            )

    # return losses and accuracies
    history = {
        'num_epochs': num_epochs,
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs
    }

    return history

