import os
import copy
from tqdm import tqdm
import torch

def train(dataloaders, model, criterion, optimizer, scheduler, device, optim_model_wts_dir, n_epochs=30,):
    loss_hist = {'train':[], 'val':[]}
    acc_hist = {'train':[], 'val':[]}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(n_epochs):
        current_lr = get_learning_rate(optimizer)
        print('Epoch {}/{}; Current learning rate {}'.format(epoch+1, n_epochs, current_lr))

        # train phase
        model.train()
        train_loss, train_accuracy = get_epoch_loss(model, criterion, dataloaders['train'], device, optimizer)
        loss_hist['train'].append(train_loss)
        acc_hist['train'].append(train_accuracy)

        # validation phase
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = get_epoch_loss(model, criterion, dataloaders['val'], device,)
        if val_accuracy>best_val_acc:
            best_val_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            best_model_name = 'best_model_wts.pt'.format(epoch+1)
            best_model_path = os.path.join(optim_model_wts_dir, best_model_name)
            torch.save(best_model_wts, best_model_path)
            print('Best model weights are updated at epoch {}!'.format(epoch+1))
        loss_hist['val'].append(val_loss)
        acc_hist['val'].append(val_accuracy)

        scheduler.step(val_loss)
        if current_lr!=get_learning_rate(optimizer):
            print('Loading best model weights!')
            model.load_state_dict(best_model_wts)

        print("train loss: {:.6f}, val loss: {:.6f}, accuracy: {:.2f}".format(train_loss, val_loss, 100*val_accuracy))
        print("-"*60)
        print()

    model.load_state_dict(best_model_wts)

    return model, loss_hist, acc_hist

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# given mini-batch data
# return number of correct predictions
def batch_correct_preds(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    correct_preds = pred.eq(target.view_as(pred)).sum().item()
    return correct_preds

# compute mini-batch loss and perform backpropagation
def get_batch_loss(criterion, output, target, optimizer=None):
    loss = criterion(output, target)
    with torch.no_grad():
        n_batch_correct_preds = batch_correct_preds(output, target)
    # if optimizer is provided, backpropagate with it
    if optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), n_batch_correct_preds

# compute epoch loss and prediction accuracy
def get_epoch_loss(model, criterion, dataloader, device, optimizer=None):
    running_loss, running_total_correct_preds = 0.0, 0.0                 # running_metric measures total classification accurary
    len_dataset = len(dataloader.dataset)

    for x_batch, y_batch in tqdm(dataloader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch)
        batch_loss, n_batch_correct_preds = get_batch_loss(criterion, output, y_batch, optimizer)

        running_loss+=batch_loss
        if n_batch_correct_preds:
            running_total_correct_preds += n_batch_correct_preds

    loss = running_loss/float(len_dataset)
    accuracy = running_total_correct_preds/float(len_dataset)
    return loss, accuracy
