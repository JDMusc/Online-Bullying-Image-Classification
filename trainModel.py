import copy
from toolz import pipe as p

import torch
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt


def train_model(
    model, criterion, optimizer, scheduler, dataloaders, 
    dataset_sizes, device, num_epochs=25):

    writer = SummaryWriter()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_acc, model_wts = _run_epoch(
            model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,
            device, num_epochs, epoch, writer)
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model_wts

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def _run_epoch(model, criterion, optimizer, scheduler, dataloaders, 
               dataset_sizes, device, num_epochs, epoch, writer):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        is_train = phase == 'train'
        is_val = not is_train

        if is_train:
            scheduler.step()
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            preds, loss =  _take_step(
                model, criterion, optimizer, inputs, labels, is_train)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        _log_epoch_stats(writer, epoch, phase, epoch_loss, epoch_acc)
        _log_model_params(writer, model, epoch, phase)

        # deep copy the model
        model_wts = copy.deepcopy(model.state_dict())
            
        return epoch_acc, model_wts



def _take_step(model, criterion, optimizer, inputs, labels, is_train):
    
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward
    # track history if only in train
    with torch.set_grad_enabled(is_train):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        if is_train:
            loss.backward()
            optimizer.step()
    
    return preds, loss


def _add_scope(scope, k):
    return scope + '/' + k
    
def _add_scope_gen(scope):
    return lambda k: _add_scope(scope, k)


def _log_model_params(writer, model, run_num, scope):
    with torch.no_grad():
        for (name, param) in model.named_parameters():
            p(name, 
              _add_scope_gen(scope),
              lambda _: writer.add_scalar(_, param.abs().mean(), run_num)
             )


def _log_epoch_stats(writer, epoch, scope, epoch_loss, epoch_acc):  
    log_measure = lambda k, v: p(k,
                                 _add_scope_gen(scope),
                                 lambda _ : writer.add_scalar(_, v, epoch)
                                )
    
    log_measure('loss', epoch_loss)
    log_measure('accuracy', epoch_acc)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        scope, epoch_loss, epoch_acc))
    
    
def visualize_model(model, dataloaders, device, num_images=6, k='val'):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[k]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)