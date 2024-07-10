'''Training loop.'''

import torch


def _device(device=None):
    '''Determine device.'''
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = device
    return device


@torch.no_grad()
def _accuracy(preds, targets):
    '''Calculate accuracy.'''
    return (preds == targets).sum() / len(preds)


def train_node_level(data,
                     model,
                     criterion,
                     optimizer,
                     num_epochs,
                     log_every=1,
                     device=None):
    '''Train node-level prediction model.'''

    device = _device(device)
    data = data.to(device)

    train_losses = torch.zeros(num_epochs + 1, dtype=data.x.dtype)
    train_accs = torch.zeros(num_epochs + 1, dtype=data.x.dtype)

    val_losses = torch.zeros(num_epochs + 1, dtype=data.x.dtype)
    val_accs = torch.zeros(num_epochs + 1, dtype=data.x.dtype)

    # compute initial losses and accuracies
    model.eval()

    with torch.no_grad():
        y_pred = model(data.x, data.edge_index)

        loss = criterion(
            y_pred[data.train_mask],
            data.y[data.train_mask]
        )

        train_acc = _accuracy(
            y_pred[data.train_mask].argmax(dim=1),
            data.y[data.train_mask]
        )

        val_loss = criterion(
            y_pred[data.val_mask],
            data.y[data.val_mask]
        )

        val_acc = _accuracy(
            y_pred[data.val_mask].argmax(dim=1),
            data.y[data.val_mask]
        )

    train_losses[0] = loss.cpu()
    train_accs[0] = train_acc.cpu()

    val_losses[0] = val_loss.cpu()
    val_accs[0] = val_acc.cpu()

    print(
        'Initially, train loss: {:.2e}, train acc.: {:.2f}, val. loss: {:.2e}, val. acc.: {:.2f}'.format(
            loss.cpu().item(), train_acc.cpu().item(), val_loss.cpu().item(), val_acc.cpu().item()
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
                train_acc = _accuracy(
                    y_pred[data.train_mask].argmax(dim=1),
                    data.y[data.train_mask]
                )

                y_pred = model(data.x, data.edge_index) # preds may differ in eval mode

                val_loss = criterion(
                    y_pred[data.val_mask],
                    data.y[data.val_mask]
                )

                val_acc = _accuracy(
                    y_pred[data.val_mask].argmax(dim=1),
                    data.y[data.val_mask]
                )

                train_losses[epoch_idx + 1] = loss.detach().cpu()
                train_accs[epoch_idx + 1] = train_acc.cpu()

                val_losses[epoch_idx + 1] = val_loss.cpu()
                val_accs[epoch_idx + 1] = val_acc.cpu()

            print(
                'Epoch: {:d}, train loss: {:.2e}, train acc.: {:.2f}, val. loss: {:.2e}, val. acc.: {:.2f}'.format(
                    epoch_idx + 1, loss.detach().cpu().item(), train_acc.cpu().item(),
                    val_loss.cpu().item(), val_acc.cpu().item()
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


@torch.no_grad()
def _test_loss(model,
               criterion,
               data_loader,
               device=None):
    '''Compute the loss over a dataloader.'''

    model.eval()

    # compute batch losses
    batch_losses = torch.zeros(len(data_loader))

    for idx, batch in enumerate(data_loader):
        batch = batch.to(device)

        y_pred = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        batch_loss = criterion(y_pred, batch.y)
        batch_losses[idx] = batch_loss.cpu()

    # reduce batch losses
    if criterion.reduction == 'mean':
        test_loss = batch_losses.mean()
    elif criterion.reduction == 'sum':
        test_loss = batch_losses.mean()
    else:
        test_loss = batch_losses

    return test_loss


def train_graph_level(model,
                      criterion,
                      optimizer,
                      num_epochs,
                      train_loader,
                      val_loader=None,
                      log_every=1,
                      device=None):
    '''Train graph-level prediction model.'''

    device = _device(device)

    train_losses = torch.zeros(num_epochs + 1)

    if val_loader is not None:
        val_losses = torch.zeros(num_epochs + 1)

    # compute initial losses
    loss = _test_loss(model, criterion, train_loader, device=device)
    train_losses[0] = loss

    if val_loader is not None:
        val_loss = _test_loss(model, criterion, val_loader, device=device)
        val_losses[0] = val_loss

    print(
        'Initially, train loss: {:.2e}, val. loss: {:.2e}'.format(
            loss.item(), val_loss.item()
        )
    )

    # loop over epochs
    for epoch_idx in range(num_epochs):

        model.train()

        # loop over batches
        batch_losses = torch.zeros(len(train_loader))

        for batch_idx, batch in enumerate(train_loader):

            # perform train step
            batch = batch.to(device)

            optimizer.zero_grad()

            y_pred = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            loss = criterion(y_pred, batch.y)

            loss.backward()
            optimizer.step()

            # calculate running loss
            batch_losses[batch_idx] = loss.detach().cpu()

            if (batch_idx + 1) < 3:
                running_loss = batch_losses[batch_idx]
            else:
                running_loss = sum(batch_losses[-3:]) / 3

        # monitor performance
        if (epoch_idx + 1) % log_every == 0:
            val_loss = _test_loss(model, criterion, val_loader, device=device)
            val_losses[epoch_idx + 1] = val_loss

            train_losses[epoch_idx + 1] = running_loss

            print(
                'Epoch: {:d}, running loss: {:.2e}, val. loss: {:.2e}'.format(
                    epoch_idx + 1, running_loss.item(), val_loss.item()
                )
            )

    # return losses
    history = {
        'num_epochs': num_epochs,
        'train_loss': train_losses,
        'val_loss': val_losses
    }

    return history

