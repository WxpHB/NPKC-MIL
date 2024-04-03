import numpy as np
import math
from itertools import islice
import collections
from torch.nn import init
import os
import requests
import torch.nn as nn
import torch
import random
import pandas as pd
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import dgl
import pickle
from scipy import ndimage


def save_pkl(filename, save_object):
    writer = open(filename, 'wb')
    pickle.dump(save_object, writer)
    writer.close()


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)





def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
                   seed = 7, label_frac = 1.0, custom_test_ids = None):
    indices = np.arange(samples).astype(int)

    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        if custom_test_ids is not None: # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices)  # all indices of this class
            val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

            remaining_ids = np.setdiff1d(possible_indices, val_ids)  # indices of this class left after validation
            all_val_ids.extend(val_ids)

            if custom_test_ids is None: # sample test split

                test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)

            else:
                sample_num  = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('初始化方法 [%s] 未实现！' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('初始化方法为： %s' % init_type)
    net.apply(init_func)


def get_number_of_classes(class_split):
    return len(class_split.split('VS'))


# 查看网络
def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('参数总数为: %d' % num_params)
    print('可训练参数总数为: %d' % num_params_train)


def check_for_dir(path):
    """dir检查，无则创建"""
    if path and not os.path.exists(path):
        os.makedirs(path)


def download_box_link(url, out_fname='box.file'):
    out_dir = os.path.dirname(out_fname)
    check_for_dir(out_dir)
    if os.path.isfile(out_fname):
        # print('文件已下载！')
        return out_fname

    r = requests.get(url, stream=True)

    with open(out_fname, "wb") as large_file:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                large_file.write(chunk)
    return out_fname


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def seed_torch(seed=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns=['train', 'val', 'test'])

    df.to_csv(filename)
    print()


def get_optim(model, args):
    if args.optimize == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.decay)
    elif args.optimize == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,weight_decay=args.decay)
    else:
        raise NotImplementedError
    return optimizer


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)


def collate_MIL_features(batch):
    # 只返回patch对应特征
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]


def collate_MIL_coords_features(batch):
    # 返回patch左上角坐标及patch对应的特征
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    slide_id=str([item[2] for item in batch])
    coords = torch.cat([item[3] for item in batch], dim=0)

    return [img, label,slide_id,coords]


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_split_loader(split_dataset, training=False, testing=False, weighted=False):
    """
        train或validation dataloader
    """
    kwargs = {'num_workers': 12} if device.type == "cuda" else {}
    if not testing:
        if training:
            # 注意其中的batch都是 "1" !!!
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=1, sampler=WeightedRandomSampler(weights, len(weights)),
                                    collate_fn=collate_MIL_coords_features, **kwargs)
            else:
                loader = DataLoader(split_dataset, batch_size=1, sampler=RandomSampler(split_dataset),
                                    collate_fn=collate_MIL_coords_features, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=1, sampler=SequentialSampler(split_dataset),
                                collate_fn=collate_MIL_coords_features, **kwargs)

    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset) * 0.1)), replace=False)
        loader = DataLoader(split_dataset, batch_size=1, sampler=SubsetSequentialSampler(ids),
                            collate_fn=collate_MIL_coords_features,**kwargs)

    return loader


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


class AccuracyLogger(object):
    """准确率日志"""

    def __init__(self, n_classes):
        super(AccuracyLogger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1  # 统计一张所有patch数
        self.data[Y]["correct"] += (Y_hat == Y)  # 统计pred正确的patch数

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count


def calculate_error(Y_hat, Y):
    # tensor.eq()每一个对应位置上元素是否相等–对应位置相等，就返回一个True；否则返回一个False.
    tmp=Y_hat.float()
    tmp1=tmp.eq(Y.float())
    tmp2=tmp1.float()
    tmp3=tmp2.mean()
    tmp4=tmp3.item()
    error = 1. - tmp4

    return error


def summary(model, loader, n_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = AccuracyLogger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, slide_id_, coords) in enumerate(loader):
        data, label, coords = data.to(device), label.to(device), coords.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data, slide_id=slide_id_, coords=coords,
                                                label=label, instance_eval=True)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        all_zeroFlag = (np.mean(all_labels) == 0)  # 全为类别0
        all_oneFalg = (np.mean(all_labels) == 1)  # 全为类别1
        # val中真值类别不能只有一类！！！！
        if all_zeroFlag:
            all_labels[0] = 1  # 调试用，人为构造两类！！！
        if all_oneFalg:
            all_labels[0] = 0  # 调试用，人为构造两类！！！
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        if auc == None:
            auc = 0.0
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger




"""# 图转到GPU上进行运算"""
def set_graph_on_cuda(graph):
    cuda_graph = dgl.DGLGraph()
    cuda_graph.add_nodes(graph.number_of_nodes())
    cuda_graph.add_edges(graph.edges()[0], graph.edges()[1])
    for key_graph, val_graph in graph.ndata.items():
        tmp = graph.ndata[key_graph].clone()
        cuda_graph.ndata[key_graph] = tmp.cuda()
    for key_graph, val_graph in graph.edata.items():
        cuda_graph.edata[key_graph] = graph.edata[key_graph].clone().cuda()
    return cuda_graph


def remove_small_objects(pred, min_size=64, connectivity=1):
    """Remove connected components smaller than the specified size.

    This function is taken from skimage.morphology.remove_small_objects, but the warning
    is removed when a single label is provided.

    Args:
        pred: input labelled array
        min_size: minimum size of instance in output array
        connectivity: The connectivity defining the neighborhood of a pixel.

    Returns:
        out: output array with instances removed under min_size

    """
    out = pred

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        selem = ndimage.generate_binary_structure(pred.ndim, connectivity)
        ccs = np.zeros_like(pred, dtype=np.int32)
        ndimage.label(pred, selem, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out