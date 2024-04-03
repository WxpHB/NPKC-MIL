"""
训练入口
"""
import torch
import os
import numpy as np
import torch.nn as nn
from utils.utils import init_weights,seed_torch,print_network,save_splits,get_optim
from utils.utils import get_split_loader,EarlyStopping,summary,save_pkl
from models.npkc_mil import NPKCMIL
from models.CLAM_MB import CLAM_MB
from models.model_mil import MIL_fc,MIL_fc_mc
from utils.train_batch import train_batch_npkc,validate_batch_npkc,train_batch,validate_batch
import pandas as pd


def train_entrance(args,dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    # 精度与准确度
    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)  # 分成了10份，在i上训练完继续在i+1上优化

    """===================模型参数更新========================"""
    model_args={"dropout": args.drop_out, 'num_classes': args.num_classes}
    """二分、多分类"""
    if args.model_choice == 'clam' and args.subtyping:
        model_args.update({'subtyping': True})
    """模型参数规模"""
    if args.model_size is not None and args.model_choice != 'mil':
        model_args.update({"size_arg": args.model_size})
    if args.model_choice in ['npkc_mil', 'clam_mb']:
        if args.subtyping:
            model_args.update({'subtyping': True})
        if args.B > 0:
            model_args.update({'k_sample': args.B})
        """instance损失"""
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes=2)  # 默认tau为1.0
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        if args.model_choice == 'npkc_mil':
            model =NPKCMIL(**model_args, instance_loss_fn=instance_loss_fn)
            model = model.to("cuda")
            # 初始权重是用预训练的模型，还是用其他初始化方式(kaiming初始化)?
            # D:\A_Deal_Pathology\PyTorch_Nets\CLAM\CLAM-master_3\results\None_s1_30epoch_broken1\s_1_checkpoint_epoch_10.pt
            preTrainWeight=r"D:\A_Deal_Pathology\PyTorch_Nets\CLAM\NPKC_MIL\results_deep_reviewer_required_breakPoint\None_s1\s_4_checkpoint_epoch_0.pt"
            #preTrainWeight = r"D:\A_Deal_Pathology\PyTorch_Nets\CLAM\CLAM-master_3\results\None_s1_30epoch_broken1\s_1_checkpoint_epoch_10.pt"
            model.load_state_dict(torch.load(preTrainWeight,map_location=device))
            # init_weights(model, init_type='kaiming')  # 从0开始训练时
        elif args.model_choice == 'clam_mb':
            model = CLAM_MB(**model_args, instance_loss_fn=instance_loss_fn)
            print("hhhh")
        else:
            raise NotImplementedError

    else:  # args.model_choice == 'mil'
        """多分类，未完善！！！"""
        if args.num_classes > 2:
            model = MIL_fc_mc(**model_args)
            model = model.to("cuda")
        else:
            model = MIL_fc(**model_args)

    """ 
    # 不需要另起函数投射到gpu，直接用.to(device)类方法就可以
    # model.relocate()
    """
    print_network(model)  # 网络参数输出
    # ================================================================================
    # ================================================================================
    folds = folds[6:]  # 调试用，先在一个小样本集上调通！！！
    for i in folds:
        seed_torch(args.seed)
        csv_path = '{}/splits_{}.csv'.format(args.split_dir, i)
        """dataset是一个大的class，其中有诸多类方法属性"""
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path=csv_path)

        datasets = (train_dataset, val_dataset, test_dataset)
        train(datasets, i, args, device, model)
        """
         # ================              =============
        # ================调试时以下关闭=============
          # ================              =============
        """
        # results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, args, device, model)
        # # ===================================以下为新增训练过程相关记录==========================================
        # all_test_auc.append(test_auc)
        # all_val_auc.append(val_auc)
        # all_test_acc.append(test_acc)
        # all_val_acc.append(val_acc)
        # # 结果写入pkl
        # filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        # save_pkl(filename, results)
        #
        # final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
        #                          'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc': all_val_acc})
        #
        # if len(folds) != args.k:
        #     save_name = 'summary_partial_{}_{}.csv'.format(start, end)
        # else:
        #     save_name = 'summary.csv'
        # savePath=os.path.join(args.results_dir, save_name)
        # final_df.to_csv(savePath)


def train(datasets,cur,args,device,model):
    """
     一个划分下的训练
    """
    print('\n当前训练数据文件夹为： {}!'.format(cur))
    # 记录训练过程
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=60)  # 刷新间隔，15s

    else:
        writer = None

    train_split, val_split, test_split = datasets
    savePath = os.path.join(args.results_dir, 'splits_{}.csv'.format(cur))
    save_splits(datasets, ['train', 'val', 'test'], savePath)  # 保存.csv文件，记录的是按8/1/1划分的训练/val/test文件名

    print("训练/验证/测试分别为 {}/{}/{} 个样本".format(len(train_split), len(val_split), len(test_split)) + "\n初始化损失函数...", end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.num_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('损失函数初始化完成')
    optimizer = get_optim(model, args)  # 默认‘adma'优化器
    print('优化初始化完成' + '\n数据加载初始化...', end=' ')

    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)
    print('数据加载初始化完成' + '\n设置 EarlyStopping...', end=' ')

    if args.early_stopping:
        early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True)
    else:
        early_stopping = None
    print('EarlyStopping完成!')

    for epoch in range(args.epoches_max):
        if args.model_choice in ['npkc_mil', 'clam_mb'] and not args.no_inst_cluster:
            train_batch_npkc(args,cur,epoch, model, train_loader, optimizer, args.num_classes, args.bag_weight, writer, loss_fn)
            """
             # ================              =============
             # ================调试时以下关闭=============
             # ================              =============
             """
            # stop = validate_batch_npkc(cur, epoch, model, val_loader, args.num_classes,early_stopping, writer, loss_fn, args.results_dir)
            stop=False

        else:   # mil多分类
            train_batch(epoch, model, train_loader, optimizer, args.num_classes, writer, loss_fn)
            stop = validate_batch(cur, epoch, model, val_loader, args.num_classes,early_stopping, writer, loss_fn, args.results_dir)
        if stop:
            break
    #
    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint_break_down.pt".format(cur)))
        # # =======================================================================================================================
    # _, val_error, val_auc, _ = summary(model, val_loader, args.num_classes)
    #
    # print('val 误差: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))
    #
    """test不用做！！！！"""
    #results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.num_classes)
    #print('test 误差 : {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))
    #
    # for i in range(args.num_classes):
    #     acc, correct, count = acc_logger.get_summary(i)
    #     if acc == None:
    #         acc = 0.0
    #     print('类别 {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
    #
    #     if writer:
    #         writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    # if writer:
    #     writer.add_scalar('final/val_error', val_error, 0)
    #     writer.add_scalar('final/val_auc', val_auc, 0)
    #     # writer.add_scalar('final/test_error', test_error, 0)
    #     # writer.add_scalar('final/test_auc', test_auc, 0)
    #     # writer.close()
    # return results_dict, test_auc, val_auc, 1 - test_error, 1 - val_error

