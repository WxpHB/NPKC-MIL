"""
每batch数据
"""
import torch
from utils.utils import AccuracyLogger,calculate_error
import os
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc


def train_batch_npkc(args ,cur ,epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):

    # =============================================================
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    # 两个微观的logger，它们的acc低并不意味着最终结果的acc低！！！最终是以WSI级别的指标为准！
    WSI_acc_logger = AccuracyLogger(n_classes=n_classes)
    inst_acc_logger = AccuracyLogger(n_classes=n_classes)
    nuclei_acc_logger =AccuracyLogger(n_classes=n_classes)

    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    train_nuclei_loss =0.
    wsi_Num = 0

    print('\n')
    for batch_idx, (data, label, slide_id, coords) in enumerate(loader):
        # data leaf-tensor，但 required_grd=False
        data, label ,coords = data.to(device), label.to(device) ,coords.to(device)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label ,coords=coords ,slide_id=slide_id, instance_eval=True)

        WSI_acc_logger.log(Y_hat, label)
        # label leaf-tensor，但 required_grd=False
        # logits non-leaf ，但 required_grd=Ture
        loss = loss_fn(logits, label)  # 整个WSI的损失
        loss_value = loss.item()

        instance_loss = instance_dict['instance_loss']
        nuclei_loss = instance_dict['nuclei_loss']
        wsi_Num += 1
        instance_loss_value = instance_loss.item()
        nuclei_loss_value =nuclei_loss.item()
        train_inst_loss += instance_loss_value  # patch级损失
        train_nuclei_loss +=nuclei_loss_value  # nuclei级损失
        # 总的损失就是WSI级与patch级损失、细胞级损失的加权平均数
        # 0.5/0.2/0.3  -> 应该给与细胞级分析更大的权重?
        total_loss = bag_weight * loss + 0.2 * instance_loss + 0.1 * nuclei_loss
        # total_loss = bag_weight* loss + 0.2 * instance_loss + 0.3*nuclei_loss

        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        nuclei_preds =instance_dict['nuclei_preds']
        inst_acc_logger.log_batch(inst_preds, inst_labels)
        nuclei_acc_logger.log_batch(nuclei_preds ,inst_labels)

        # train_loss += loss_value  # 只记录了WSI级的损失？
        train_loss += total_loss.item()
        # default20
        if (batch_idx + 1) % 10 == 0:
            # 每隔一定的batch，输出单张WSI的相关统计数据
            print('batch {}, WSI_loss: {:.4f}, instance_loss: {:.4f}, nuclei_loss: {:.4f},weighted_loss: {:.4f}, '.format(batch_idx, loss_value,
                                                                                                                          instance_loss_value, nuclei_loss_value
                                                                                                                          ,total_loss.item()) + 'label: {}, bag_size: {}'.format
                (label.item(), data.size(0)))


        error = calculate_error(Y_hat, label)
        train_error += error

        # 反向传播
        total_loss.backward()  #
        # step
        optimizer.step()
        optimizer.zero_grad()
    # ==================                       ==========================
    # ==================每个epoch的相关统计信息==========================
    # ==================                       ==========================
    # calculate loss and error for epoch
    if (epoch +10) % 10 == 0:  # 每10个epoch保存模型一次
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint_epoch_{}.pt".format(cur ,epoch)))
    print('\n')  # epoch统计信息与batch信息隔开
    train_loss /= len(loader)  # 加权平均train_loss
    train_error /= len(loader)  # 平均train_error
    train_inst_loss /= wsi_Num  # patch损失均值
    train_nuclei_loss /= wsi_Num  # nuclei损失均值

    print('Epoch:{},train_loss:{:.4f},patch_loss: {:.4f},nuclei_loss: {:.4f}, train_error:{:.4f}'.format(epoch,
                                                                                                         train_loss,
                                                                                                         train_inst_loss,
                                                                                                         train_error,
                                                                                                         train_nuclei_loss))

    for i in range(n_classes):
        # 286/48/48:train/val/test  train中normal 173张；cancer 113张
        # patch层面：normal-3672(normal WSI中16个全是，cancer WSI中至少有8个); cancer-904(只有cancer中的8个！！！)
        # 绝大部分normal被判为cancer；能全部识别是cancer类别；也即漏检(cancer判为normal)很低，错检(normal判为cancer)很高！！！
        # 原因在于，cancer WSI中的top_k个patch可能并不真是cancerous区域！！！！
        # WSI_acc为0(173)张，说明将normal一股脑判为cancer了！！！
        acc_inst, correct_inst, count_inst = inst_acc_logger.get_summary(i)
        acc_nuclei ,correct_nuclei ,count_nuclei =nuclei_acc_logger.get_summary(i)
        wsi_acc, wsi_correct, wsi_count = WSI_acc_logger.get_summary(i)
        # patch/级/nuclei级/wsi级 acc信息
        inst_nuclei_acc_msg ='class {} patch_acc {}: correct {}/{};nuclei_acc {}: correct {}/{}'.format(i, acc_inst, correct_inst,
                                                                                                       count_inst ,acc_nuclei ,correct_nuclei
                                                                                                         ,count_nuclei)
        wsi_acc_msg ='class {}: WSI_acc {}, WSI_correct {}/{}'.format(i, wsi_acc, wsi_correct, wsi_count)
        print(inst_nuclei_acc_msg)
        print(wsi_acc_msg)
        if writer and wsi_acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), wsi_acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)
        writer.add_scalar('train/nuclei_loss' ,train_nuclei_loss ,epoch)


def validate_batch_npkc(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    wsi_acc_logger = AccuracyLogger(n_classes=n_classes)
    inst_acc_logger = AccuracyLogger(n_classes=n_classes)
    nuclei_acc_logger = AccuracyLogger(n_classes=n_classes)

    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    val_nuclei_loss = 0.
    val_nuclei_acc = 0.
    inst_count = 0

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))  # labels = np.zeros(len(loader))
    sample_size = model.k_sample
    with torch.no_grad():
        for batch_idx, (data, label, slide_id, coords) in enumerate(loader):
            data, label, coords = data.to(device), label.to(device), coords.to(device)
            logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, coords=coords, slide_id=slide_id, instance_eval=True)

            wsi_acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            val_loss += loss.item()

            instance_loss = instance_dict['instance_loss']
            nuclei_loss = instance_dict["nuclei_loss"]

            inst_count += 1
            instance_loss_value = instance_loss.item()
            instance_nuclei_loss_value = nuclei_loss.item()
            val_inst_loss += instance_loss_value
            val_nuclei_loss += instance_nuclei_loss_value

            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_nuclei_preds = instance_dict['nuclei_preds']

            inst_acc_logger.log_batch(inst_preds, inst_labels)
            nuclei_acc_logger.log_batch(inst_nuclei_preds, inst_labels)
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        all_zeroFlag = (np.mean(labels) == 0)  # 全为类别0
        all_oneFalg = (np.mean(labels) == 1)  # 全为类别1

        # val中真值类别不能只有一类！！！！
        if all_zeroFlag:
            labels[0] = 1  # 调试用，人为构造两类！！！
        if all_oneFalg:
            labels[0] = 0  # 调试用，人为构造两类！！！
        prob_ = prob[:, 1]
        auc = roc_auc_score(labels, prob_)
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    val_inst_loss /= inst_count  # patch损失均值
    val_nuclei_loss /= inst_count  # nuclei损失均值
    for i in range(n_classes):
        inst_acc, inst_correct, inst_count = inst_acc_logger.get_summary(i)
        acc_nuclei, correct_nuclei, count_nuclei = nuclei_acc_logger.get_summary(i)
        wsi_acc, wsi_correct, wsi_count = wsi_acc_logger.get_summary(i)
        print('class {} patch_acc {}: correct {}/{};nuclei_acc {}: correct {}/{}'.format(i, inst_acc, inst_correct, inst_count,
                                                                                         acc_nuclei, correct_nuclei, count_nuclei))

        print('class {}: acc {}, correct {}/{}'.format(i, wsi_acc, wsi_correct, wsi_count))

        if writer and wsi_acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), wsi_acc, epoch)
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)
        writer.add_scalar('val/nuclei_loss', val_nuclei_loss, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def train_batch(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = AccuracyLogger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label, slide_id, coords) in enumerate(loader):
        data, label,coords = data.to(device), label.to(device),coords.to(device)
    # for batch_idx, (data, label) in enumerate(loader):
    #     data, label = data.to(device), label.to(device)
        # top_instance, Y_prob, Y_hat, y_probs, results_dict
        logits, Y_prob, Y_hat, _, _ = model(data)

        acc_logger.log(Y_hat, label)

        # 由用户初始创建的变量，而不是程序产生的中间结果变量，那么该变量为叶变量。
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('当前batch号 {}, 损失: {:.4f}, 标签: {}, bag大小: {}'.format(batch_idx, loss_value, label.item(),
                                                                           data.size(0)))

        error = calculate_error(Y_hat, label)  # 统计错误个数
        train_error += error

        # 反向传播
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)


def validate_batch(cur, epoch, model, loader, n_classes, early_stopping=None, writer=None, loss_fn=None, results_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = AccuracyLogger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.

    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data)

            acc_logger.log(Y_hat, label)

            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()

            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])

    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}'.format(val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False





