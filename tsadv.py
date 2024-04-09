
from ts_fl import *
from models.fcn import ConvNet, ResNet, MLP
READ_CKPT = True


# helper function because otherwise non-empty strings
# evaluate as True
def bool_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
def load_data(path):
    data = np.loadtxt(path)
    # nan
    mask = np.isnan(data)
    data[mask] = 0
    # Normalize some datasets without normalization在没有归一化的情况下使某些数据集归一化
    # if normalize:
    #        mean = data[:, 1:].mean(axis=1, keepdims=True)
    #        std = data[:, 1:].std(axis=1, keepdims=True)
    #        data[:, 1:] = (data[:, 1:] - mean) / (std + 1e-8)
    return data  # 返回归一化数据


if __name__ == "__main__":
    # Training settings FaceAll MelbournePedestrian ShapesAll SwedishLeaf
    # TwoPatterns UWaveGestureLibraryX
    # ElectricDevices ECG5000
    print('========================> Building model........................')
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=1.00, metavar='M',
                        help='Learning rate step gamma (default: 0.99)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--fraction', type=float or int, default=0.1,
    #                     help='how many fraction of poisoned data inserted')
    parser.add_argument('--local_train_period', type=int, default=1,
                        help='number of local training epochs')
    parser.add_argument('--num_nets', type=int, default=3383,
                        help='number of totally available users')
    parser.add_argument('--part_nets_per_round', type=int, default=30,
                        help='number of participating clients per FL round')
    parser.add_argument('--fl_round', type=int, default=100,
                        help='total number of FL round to conduct')
    parser.add_argument('--fl_mode', type=str, default="fixed-pool",
                        help='fl mode: fixed-freq mode or fixed-pool mode')
    parser.add_argument('--attacker_pool_size', type=int, default=100,
                        help='size of attackers in the population, used when args.fl_mode == fixed-pool only')
    parser.add_argument('--defense_method', type=str, default="no-defense",
                        help='防御方法：defense method used: no-defense|norm-clipping|norm-clipping-adaptive|weak-dp|krum|multi-krum|rfa|')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to set, can take the value of: cuda or cuda:x')
    # parser.add_argument('--attack_method', type=str, default="blackbox",
    #                     help='describe the attack type: blackbox|pgd|graybox|')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset to use during the training process')
    parser.add_argument('--model', type=str, default='fcn',
                        help='model to use during the training process')
    parser.add_argument('--eps', type=float, default=1,
                        help='specify the l_inf epsilon budget')
    parser.add_argument('--norm_bound', type=float, default=1.5,
                        help='describe if there is defense method: no-defense|norm-clipping|weak-dp|')
    parser.add_argument('--adversarial_local_training_period', type=int, default=2,
                        help='specify how many epochs the adversary should train for')
    parser.add_argument('--poison_type', type=str, default='ardis',
                        help='specify source of data poisoning: |ardis|fashion|(for EMNIST) || '
                             '|southwest|southwest+wow|southwest-da|greencar-neo|howto|(for CIFAR-10)')
    parser.add_argument('--rand_seed', type=int, default=7,
                        help='random seed utilize in the experiment for reproducibility.')
    parser.add_argument('--model_replacement', type=bool_string, default=False,
                        help='to scale or not to scale')
    parser.add_argument('--project_frequency', type=int, default=10,
                        help='project once every how many epochs')
    # parser.add_argument('--adv_lr', type=float, default=0.02,
    #                     help='learning rate for adv in PGD setting')
    parser.add_argument('--prox_attack', type=bool_string, default=False,
                        help='use prox attack')
    parser.add_argument('--attack_case', type=str, default="edge-case",
                        help='attack case indicates wheather the honest nodes see the attackers poisoned data points: '
                             'edge-case|normal-case|almost-edge-case')
    parser.add_argument('--stddev', type=float, default=0.025,
                        help='choose std_dev for weak-dp defense')
    parser.add_argument('--target_class', type=float, default=-1,
                        help='target_class')
    parser.add_argument('--cuda', action='store_true', help='it is unnecessary')
    parser.add_argument('--attackmathed', type=str, default="AttackerRandShape",
                        help='attackmathed: Attacker, AttackerRandShape, AttackerRandAll, AttackerOnepoint')

    args = parser.parse_args()
    partition_strategy = "homo"
    partition_strategy = "hetero-dir"

    save_dir = './ts_attack/resnet/' + args.dataset + partition_strategy+ args.fl_mode\
               +'_' + args.defense_method +'_' + str(args.attacker_pool_size) + '_' + args.attackmathed + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")
    datadir = './data/UCR/'
    os.makedirs(save_dir, exist_ok=True)
    sys.stdout = Logger(save_dir + '/print.txt')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    device = torch.device(args.device if use_cuda else "cpu")

    print("Running Attack of the tails with args:\n {}".format(args))
    print(device)
    print('========================> Building model........................')

    torch.manual_seed(args.seed)
    criterion = nn.CrossEntropyLoss()

    # add random seed for the experiment for reproducibility随机种子
    seed_experiment(seed=args.rand_seed)

    import copy

    '''xiugai'''
    net_dataidx_map = partition_tsdata(
        # D:\21120338\code\OOD_TS_FL\data
        args.dataset, './data/UCR/', partition_strategy,
        args.num_nets, 0.5)
    for key in net_dataidx_map:
        print(len(net_dataidx_map[key]), net_dataidx_map[key])

    # print(net_dataidx_map)
    # rounds of fl to conduct
    ## some hyper-params here:
    local_training_period = args.local_train_period  # 5 #1 local training epochs
    adversarial_local_training_period = 5

    # 
    # train_data = np.loadtxt(datadir + dataset +'/'+dataset + '_TRAIN.txt')
    # poisoned_data = np.loadtxt(datadir + args.dataset + '/' + args.dataset + '_attack.txt')
    poisoned_data = load_data(datadir + args.dataset + '/' + args.dataset + '_attack.txt')
    # poisoned_data = poisoned_data[0:30, :]
    #poisoned_data = torch.tensor(poisoned_data[0:50, :], dtype=torch.float)
    poisoned_data = torch.tensor(poisoned_data, dtype=torch.float)
    num_dps_poisoned_dataset = poisoned_data.shape[0]

    # targetted_task_test_data = np.loadtxt(datadir + args.dataset + '/' + args.dataset + '_TEST.txt')
    targetted_task_test_data = load_data(datadir + args.dataset + '/' + args.dataset + '_TEST.txt')
    targetted_task_test_data = torch.tensor(targetted_task_test_data, dtype=torch.float)

    # vanilla_test_data = np.loadtxt(datadir + args.dataset + '/' + args.dataset + '_TEST.txt')
    vanilla_test_data = load_data(datadir + args.dataset + '/' + args.dataset + '_TEST.txt')
    vanilla_test_data = torch.tensor(vanilla_test_data, dtype=torch.float)
    # clean_train_data = load_data(datadir + args.dataset + '/' + args.dataset + '_TRAIN.txt')
    clean_train_data = load_data(datadir + args.dataset + '/' + args.dataset + '_train.txt')

    args.batch_size = int(min(len(clean_train_data) / 10, 16))

    clean_train_data = torch.tensor(clean_train_data, dtype=torch.float)

    seq_len = vanilla_test_data.shape[1] - 1
    n_class = len(np.unique(vanilla_test_data[:, 0]))
    #  train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float), torch.from_numpy(train_labels).to(torch.float))
    poisoned_dataset = TensorDataset(poisoned_data[:, 1:], poisoned_data[:, 0])
    targetted_task_test_dataset = TensorDataset(targetted_task_test_data[:, 1:], targetted_task_test_data[:, 0])
    vanilla_test_dataset = TensorDataset(vanilla_test_data[:, 1:], vanilla_test_data[:, 0])
    clean_train_dataset = TensorDataset(clean_train_data[:, 1:], clean_train_data[:, 0])

    poisoned_train_loader = torch.utils.data.DataLoader(poisoned_dataset,
                                                        batch_size=args.batch_size, shuffle=True, **kwargs)
    targetted_task_test_loader = torch.utils.data.DataLoader(targetted_task_test_dataset,
                                                             batch_size=args.test_batch_size, shuffle=False,
                                                             **kwargs)
    vanilla_test_loader = torch.utils.data.DataLoader(vanilla_test_dataset,
                                                      batch_size=args.test_batch_size, shuffle=False, **kwargs)
    clean_train_loader = torch.utils.data.DataLoader(clean_train_dataset,
                                                     batch_size=args.batch_size, shuffle=True, **kwargs)
    # datadir = 'D:/21120338/code/FL_TSadv_master/data/UCR/'
    advdatapath = datadir + args.dataset + '/' + args.dataset
    dataloader, vanilla_test_loader = get_ts_loader(args.dataset, advdatapath, 6, 1000)
    # _, vanilla_test_loader = get_ts_loader(args.dataset, advdatapath, args.batch_size, args.test_batch_size)
    # poisoned_train_loader, vanilla_test_loader, targetted_task_test_loader, num_dps_poisoned_dataset, clean_train_loader = \
    #     load_poisoned_dataset(args=args)
    # file = open('./dataread.txt', 'w+')
    # file.write(str(len(poisoned_train_loader)) + '\n')
    # file.write(str(len(vanilla_test_loader)) + '\n')
    # file.write(str(len(targetted_task_test_loader)) + '\n')
    # file.write(str(num_dps_poisoned_dataset) + '\n')
    # file.write(str(len(clean_train_loader)) + '\n')
    # file.close()
    READ_CKPT = False #是否直接读取模型
    if READ_CKPT:
        if args.model == "lenet":
            net_avg = Net(num_classes=10).to(device)
            with open("./checkpoint/emnist_lenet_10epoch.pt", "rb") as ckpt_file:
                ckpt_state_dict = torch.load(ckpt_file, map_location=device)
        elif args.model in ("vgg9", "vgg11", "vgg13", "vgg16"):
            net_avg = get_vgg_model(args.model).to(device)
            # net_avg = VGG(args.model.upper()).to(device)
            # load model here
            # with open("./checkpoint/trained_checkpoint_vanilla.pt", "rb") as ckpt_file:
            with open("./checkpoint/Cifar10_{}_10epoch.pt".format(args.model.upper()), "rb") as ckpt_file:
                ckpt_state_dict = torch.load(ckpt_file, map_location=device)
        elif args.model == 'fcn':
            net_avg = ConvNet(n_in=seq_len, n_classes=n_class).to(device)
            with open("./checkpoint/fcn_10epoch.pt", "rb") as ckpt_file:
                ckpt_state_dict = torch.load(ckpt_file, map_location=device)
        net_avg.load_state_dict(ckpt_state_dict)
        logger.info("Loading checkpoint file successfully ...")
    else:#生成新模型
        if args.model == "mlp":
            net_avg = MLP(n_in=seq_len, n_classes=n_class).to(device)
        elif args.model == "resnet":
            net_avg = ResNet(n_in=seq_len, n_classes=n_class).to(device)
        else:
            net_avg = ConvNet(n_in=seq_len, n_classes=n_class).to(device)
    print(net_avg)
    print("Test the model performance on the entire task before FL process ... ")
    # # test return final_acc, task_acc

    overall_acc1, raw_acc1 = advfl_test(net_avg, device, vanilla_test_loader, test_batch_size=args.test_batch_size, criterion=criterion, ts_len=seq_len,
         num_class=n_class, mode="raw-task", dataset=args.dataset)
    # advfl_test(net_avg, device, targetted_task_test_loader, test_batch_size=args.test_batch_size, criterion=criterion,ts_len=seq_len,
    #      num_class=n_class, mode="targetted-task", dataset=args.dataset, poison_type='attack')
    print('\n-----============------训练前的测试 overall_acc1, raw_acc1\n', overall_acc1, raw_acc1)
    # let's remain a copy of the global model for measuring the norm distance:
    vanilla_model = copy.deepcopy(net_avg)

    ######初始化保存模型
    for global_user_idx in range(args.num_nets):
        save_model_path = "./client_model/00/" + str(global_user_idx) + "_round_model.pth"
        torch.save(net_avg.state_dict(), save_model_path)
        save_model_path = "./client_model/01/" + str(global_user_idx) + "_round_model.pth"
        torch.save(net_avg.state_dict(), save_model_path)
        save_model_path = "./client_model/02/" + str(global_user_idx) + "_round_model.pth"
        torch.save(net_avg.state_dict(), save_model_path)
        save_model_path = "./client_model/03/" + str(global_user_idx) + "_round_model.pth"
        torch.save(net_avg.state_dict(), save_model_path)


    starttime = time.time()
    if args.fl_mode == "fixed-pool":
        arguments = {
            "vanilla_model": vanilla_model,
            "net_avg": net_avg,
            "net_dataidx_map": net_dataidx_map,
            "num_nets": args.num_nets,
            "dataset": args.dataset,
            "model": args.model,
            "part_nets_per_round": args.part_nets_per_round,
            "attacker_pool_size": args.attacker_pool_size,
            "fl_round": args.fl_round,
            "local_training_period": args.local_train_period,
            "adversarial_local_training_period": args.adversarial_local_training_period,
            "args_lr": args.lr,
            "args_gamma": args.gamma,
            "num_dps_poisoned_dataset": num_dps_poisoned_dataset,
            "attack_ts_train_loader": poisoned_train_loader,
            "clean_train_loader": clean_train_loader,
            "vanilla_emnist_test_loader": vanilla_test_loader,
            "targetted_task_test_loader": targetted_task_test_loader,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            #"log_interval": args.log_interval,
            "defense_technique": args.defense_method,
            "eps": args.eps,
            "norm_bound": args.norm_bound,
            "poison_type": args.poison_type,
            "device": device,
            "model_replacement": args.model_replacement,
            "project_frequency": args.project_frequency,
            #"adv_lr": args.adv_lr,
            "prox_attack": args.prox_attack,
            "attack_case": args.attack_case,
            "stddev": args.stddev,
            "n_class":n_class,
            "target_class":args.target_class,
            "seq_len":seq_len,
            "cuda":args.cuda,
            "attackmathed":args.attackmathed,
            "save_dir":save_dir,
            "datadir":datadir
        }
        fixed_pool_fl_trainer = FixedPoolFL_TSadv(arguments=arguments)
        fixed_pool_fl_trainer.run()
    # 固定频率攻击 295-340
    if args.fl_mode == "fixed-freq":
        arguments = {
            "vanilla_model": vanilla_model,
            "net_avg": net_avg,
            "net_dataidx_map": net_dataidx_map,
            "num_nets": args.num_nets,
            "dataset": args.dataset,
            "model": args.model,
            "part_nets_per_round": args.part_nets_per_round,
            "attacker_pool_size": args.attacker_pool_size,
            "fl_round": args.fl_round,
            "local_training_period": args.local_train_period,
            "adversarial_local_training_period": args.adversarial_local_training_period,
            "args_lr": args.lr,
            "args_gamma": args.gamma,
            "num_dps_poisoned_dataset": num_dps_poisoned_dataset,
            "attack_ts_train_loader": poisoned_train_loader,
            "clean_train_loader": clean_train_loader,
            "vanilla_emnist_test_loader": vanilla_test_loader,
            "targetted_task_test_loader": targetted_task_test_loader,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            #"log_interval": args.log_interval,
            "defense_technique": args.defense_method,
            "eps": args.eps,
            "norm_bound": args.norm_bound,
            "poison_type": args.poison_type,
            "device": device,
            "model_replacement": args.model_replacement,
            "project_frequency": args.project_frequency,
            #"adv_lr": args.adv_lr,
            "prox_attack": args.prox_attack,
            "attack_case": args.attack_case,
            "stddev": args.stddev,
            "n_class":n_class,
            "target_class":args.target_class,
            "seq_len":seq_len,
            "cuda":args.cuda,
            "attackmathed":args.attackmathed,
            "save_dir":save_dir,
            "datadir":datadir
        }
        fixed_freq_fl_trainer = FixedFreqFL_TSadv(arguments=arguments)
        fixed_freq_fl_trainer.run()
    else:
        print('--------------------------------------over')
    # 结束
    endtime = time.time() - starttime
    print(f"\nFL TS attack time: {endtime} s")
    #sys.stdout.reset()