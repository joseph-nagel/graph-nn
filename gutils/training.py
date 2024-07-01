'''Training loop.'''

import torch


@torch.no_grad()
def accuracy(preds, targets):
    '''Calculate accuracy.'''
    return (preds == targets).sum() / len(preds)


def train(model,
          criterion,
          optimizer,
          num_epochs,
          data,
          log_every=1):
    '''Train model.'''

    # compute val. loss/acc.
    model.eval()

    with torch.no_grad():
        y_pred = model(data.x, data.edge_index)

        val_loss = criterion(
            y_pred[data.val_mask],
            data.y[data.val_mask]
        )

        val_acc = accuracy(
            y_pred[data.val_mask].argmax(dim=1),
            data.y[data.val_mask]
        )

    print(f'Before training, val. loss: {val_loss:.2e}, val. acc.: {val_acc:.2f}')

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

            # print summary
            print(
                'Epoch: {:d}, loss: {:.2e}, acc.: {:.2f}, val. loss: {:.2e}, val. acc.: {:.2f}'.format(
                    epoch_idx + 1, loss.detach().item(), train_acc.detach().item(),
                    val_loss.detach().item(), val_acc.detach().item()
                )
            )

