import torch
import numpy as np
from torch import optim
from collections import defaultdict
from itertools import chain
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, confusion_matrix


BEST_SCORE = 0.0
BEST_F1 = 0.0
BEST_AUROC = 0.0
BEST_EPOCH = 0


def disable_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm3d):
            module.eval()


def enable_bn(model):
    model.train()


def train_epoch(net, train_loader, optimizer, warmer, criterion, ep, scheduler=None, MODE='max', use_clinical=False):
    net.train()

    res = defaultdict(list)

    running_loss = 0.0
    running_acc = 0.0

    y_pred_ = []
    y_true_ = []

    device = torch.device("cuda:0")
    for batch_ix, batch in enumerate(train_loader):
        if MODE == 'min':
            # slice based-on min_depth
            img = img[:, :, :train_loader.dataset.min_depth]
        batch = [item.to(device)
                 if isinstance(item, torch.Tensor) else item
                 for item in batch]
        if use_clinical:
            img, clinical_factor, y_true, image_id = batch
        else:
            img, y_true, image_id = batch

        if use_clinical:
            outputs = net(img, clinical_factor=clinical_factor)
        else:
            outputs = net(img)

        loss = criterion(outputs, torch.max(y_true, 1)[1])

        _, y_pred = torch.max(outputs, 1)
        _, y_true = torch.max(y_true, 1)
        y_score = torch.softmax(outputs, 1)[:, 1]

        y_pred_.append(y_pred.detach().cpu().numpy())
        y_true_.append(y_true.cpu().numpy())

        res['y_test'] += y_true.cpu().numpy().tolist()
        res['y_score'] += y_score.detach().cpu().numpy().tolist()
        res['y_hat'] += y_pred.detach().cpu().numpy().tolist()
        res['CaseIdx'] += list(image_id)

        loss.backward()

        step_interval = 1  # default
        if img.size(0) == 1:
            step_interval = 16

        if batch_ix % step_interval == 0:
            optimizer.first_step(zero_grad=True)
            if use_clinical:
                outputs = net(img, clinical_factor=clinical_factor)
            else:
                outputs = net(img)

            criterion(outputs, y_true).backward()
            optimizer.second_step(zero_grad=True)

        if warmer is not None:
            warmer.step(epoch=ep)

        acc = accuracy_score(y_true.cpu(), y_pred.detach().cpu())

        running_loss += loss.item()
        running_acc += acc.item()

    for k, v in res.items():
        res[k] = np.vstack(v)

    running_loss /= len(train_loader)
    running_acc /= len(train_loader)

    # f1-score
    f1 = f1_score(list(chain(*y_true_)), list(chain(*y_pred_)))

    TN, FP, FN, TP = confusion_matrix(res['y_test'], res['y_hat']).ravel()
    SENS = TP / (TP + FN)
    SPC = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)

    # auroc
    fpr, tpr, thresh = roc_curve(res['y_test'], res['y_score'][:, 0])
    roc_auc = auc(fpr, tpr)

    print(f'[Train] ep : {ep} loss : {running_loss:.4f} \
          acc : {running_acc:.4f} / SENS : {SENS:.4f} / SPC : {SPC:.4f} / PPV : {PPV:.4f} / NPV : {NPV:.4f} / f1 : {f1:.4f} / AUROC : {roc_auc:.4f}')

    if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()

    return running_loss, roc_auc


def evaluate(net, test_loader, criterion, ep, MODE='max', use_clinical=False):
    net.eval()

    res = defaultdict(list)

    running_loss = 0.0
    running_acc = 0.0

    y_pred_ = []
    y_true_ = []

    device = torch.device("cuda:0")
    for batch in test_loader:
        if MODE == 'min':
            # slice based-on min_depth
            img = img[:, :, :test_loader.dataset.min_depth]

        batch = [item.to(device)
                 if isinstance(item, torch.Tensor) else item
                 for item in batch]
        if use_clinical:
            img, clinical_factor, y_true, image_id = batch
        else:
            img, y_true, image_id = batch

        with torch.no_grad():
            if use_clinical:
                outputs = net(img, clinical_factor=clinical_factor)
            else:
                outputs = net(img)

            loss = criterion(outputs, torch.max(y_true, 1)[1])

            _, y_pred = torch.max(outputs, 1)
            _, y_true = torch.max(y_true, 1)
            y_score = torch.softmax(outputs, 1)[:, 1]

            y_pred_.append(y_pred.detach().cpu().numpy())
            y_true_.append(y_true.cpu().numpy())

            res['y_test'] += y_true.cpu().numpy().tolist()
            res['y_score'] += y_score.detach().cpu().numpy().tolist()
            res['y_hat'] += y_pred.detach().cpu().numpy().tolist()
            res['CaseIdx'] += list(image_id)

            acc = accuracy_score(y_true.cpu(), y_pred.detach().cpu())

            running_loss += loss.item()
            running_acc += acc.item()

    for k, v in res.items():
        res[k] = np.vstack(v)

    running_loss /= len(test_loader)
    running_acc /= len(test_loader)

    # f1-score
    f1 = f1_score(list(chain(*y_true_)),
                  list(chain(*y_pred_)))

    TN, FP, FN, TP = confusion_matrix(res['y_test'], res['y_hat']).ravel()
    SENS = TP / (TP + FN)
    SPC = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)

    # auroc
    fpr, tpr, thresh = roc_curve(res['y_test'], res['y_score'][:, 0])
    roc_auc = auc(fpr, tpr)

    print(f'[Test] ep : {ep} loss : {running_loss:.4f} \
          acc : {running_acc:.4f} / SENS : {SENS:.4f} / SPC : {SPC:.4f} / PPV : {PPV:.4f} / NPV : {NPV:.4f} / f1 : {f1:.4f} / AUROC : {roc_auc:.4f}')

    global BEST_SCORE
    global BEST_F1
    global BEST_AUROC
    global BEST_EPOCH

    monitor = roc_auc

    return res, running_loss, monitor, BEST_EPOCH
