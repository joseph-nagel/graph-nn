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


def train_step(data,
               model,
               criterion,
               optimizer):
    '''Perform a training step.'''

    model.train()

    optimizer.zero_grad()

    y_pred = model(data.x, data.edge_index)

    loss = criterion(
        y_pred[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()
    optimizer.step()

    return loss


def train(data,
          model,
          criterion,
          optimizer,
          num_epochs,
          log_every=1):
    '''Train model.'''

    # val_loss = test_loss(model, criterion, val_loader)
    # print('Before training, val. loss: {:.2e}'.format(val_loss))

    for epoch_idx in range(num_epochs):

        # perform train step
        loss = train_step(
            data=data,
            model=model,
            criterion=criterion,
            optimizer=optimizer
        )

        # compute val. error
        if (epoch_idx + 1) % log_every == 0:
            # val_loss = test_loss(model, criterion, val_loader)
            # print('Epoch: {:d}, batch loss: {:.2e}, val. loss: {:.2e}'. \
            #       format(epoch_idx + 1, loss.detach().item(), val_loss))
            print('Epoch: {:d}, batch loss: {:.2e}'. \
                  format(epoch_idx + 1, loss.detach().item()))

