'''Training loop.'''

import torch


@torch.no_grad()
def test_loss(data, model, criterion):
    '''Compute loss.'''

    model.eval()

    y_pred = model(data.x, data.edge_index)

    loss = criterion(
        y_pred[data.test_mask],
        data.y[data.test_mask]
    )

    return loss


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

    # val_loss = test_loss(data, model, criterion)
    # print('Before training, val. loss: {:.2e}'.format(val_loss))

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

            # compute val. error
            # val_loss = test_loss(data, model, criterion)

            # calculate accuracy
            acc = accuracy(
                preds=y_pred[data.train_mask].argmax(dim=1),
                targets=data.y[data.train_mask]
            )

            # print summary
            print('Epoch: {:d}, loss: {:.2e}, acc.: {:.2f}'. \
                  format(epoch_idx + 1, loss.detach().item(), acc.detach().item()))

