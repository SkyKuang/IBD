# configs
config_natural = {
    'train': False,
    'adv_mode':args.adv_mode,
    'crd_k':args.crd_k,
    'crd_dim':args.crd_dim,
    'crd_emDim':args.crd_emDim,
    'crd_mode':args.crd_mode,
    'crd_moment':args.crd_moment,
    'crd_temp':args.crd_temp,
    'crd_ndata':n_data,
}

config_fgsm = {
    'train': False,
    'targeted': False,
    'epsilon': 8.0 / 255 * 1,
    'num_steps': 1,
    'step_size': 8.0 / 255 * 1,
    'random_start': True,
    'adv_mode':args.adv_mode,
    'crd_k':args.crd_k,
    'crd_dim':args.crd_dim,
    'crd_emDim':args.crd_emDim,
    'crd_mode':args.crd_mode,
    'crd_moment':args.crd_moment,
    'crd_temp':args.crd_temp,
    'crd_ndata':n_data,
}

config_pgd = {
    'train': False,
    'targeted': False,
    'epsilon': 8.0 / 255 * 1,
    'num_steps': 100,
    'step_size': 2.0 / 255 * 1,
    'random_start': True,
    'loss_func': torch.nn.CrossEntropyLoss(reduction='none'),
    'adv_mode':args.adv_mode,
    'crd_k':args.crd_k,
    'crd_dim':args.crd_dim,
    'crd_emDim':args.crd_emDim,
    'crd_mode':args.crd_mode,
    'crd_moment':args.crd_moment,
    'crd_temp':args.crd_temp,
    'crd_ndata':n_data,
}


config_cw = {
    'train': False,
    'targeted': False,
    'epsilon': 8.0 / 255 * 1,
    'num_steps': 100,
    'step_size': 2.0 / 255 * 1,
    'random_start': True,
    'loss_func': CWLoss(args.num_classes),
    'adv_mode':args.adv_mode,
    'crd_k':args.crd_k,
    'crd_dim':args.crd_dim,
    'crd_emDim':args.crd_emDim,
    'crd_mode':args.crd_mode,
    'crd_moment':args.crd_moment,
    'crd_temp':args.crd_temp,
    'crd_ndata':n_data,
}
