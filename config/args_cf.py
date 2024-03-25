import argparse
import utils

def get_train_args(description='AT'):
    parser = argparse.ArgumentParser(description='Adversarial Training')

    parser.register('type', 'bool', utils.str2bool)

    parser.add_argument('--resume',
                        '-r',
                        action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--net_module',
                        default='base',
                        type=str,
                        help='net module')
    parser.add_argument('--model_dir',
                        default='./checkpoint/',
                        type=str,
                        help='model path')
    parser.add_argument('--init_model_pass',
                        default='-1',
                        type=str,
                        help='init model pass (-1: from scratch; K: checkpoint-K)')
    parser.add_argument('--max_epoch',
                        default=200,
                        type=int,
                        help='max number of epochs')
    parser.add_argument('--save_epochs',
                        default=10,
                        type=int,
                        help='save period')
    parser.add_argument('--decay_epoch1',
                        default=100,
                        type=int,
                        help='learning rate decay epoch one')
    parser.add_argument('--decay_epoch2',
                        default=150,
                        type=int,
                        help='learning rate decay point two')
    parser.add_argument('--decay_epoch3',
                        default=220,
                        type=int,
                        help='learning rate decay point three')
    parser.add_argument('--decay_rate',
                        default=0.1,
                        type=float,
                        help='learning rate decay rate')
    parser.add_argument('--batch_size_train',
                        default=128,
                        type=int,
                        help='batch size for training')
    parser.add_argument('--test_batch',
                        default=100,
                        type=int,
                        help='batch size for testing')
    parser.add_argument('--optim',
                        default='SGD',
                        type=str,
                        help='optimizer')  # concat cascade
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        help='momentum (1-tf.momentum)')
    parser.add_argument('--weight_decay',
                        default=5e-4,
                        type=float,
                        help='weight decay')
    parser.add_argument('--log_step',
                        default=50,
                        type=int,
                        help='log_step')
    parser.add_argument('--num_classes',
                        default=10,
                        type=int,
                        help='num classes')
    parser.add_argument('--image_size',
                        default=32,
                        type=int,
                        help='image size')
    parser.add_argument('--dataset',
                        default='cifar10',
                        type=str,
                        help='dataset')  # concat cascade
    parser.add_argument('--input_size',
                        default=32,
                        type=int,
                        help='depth of model')
    parser.add_argument('--input_channels',
                        default=3,
                        type=int,
                        help='input channels')
    parser.add_argument('--depth',
                        default=28,
                        type=int,
                        help='depth of model')
    parser.add_argument('--aux_depth',
                        default=28,
                        type=int,
                        help='depth of aux model')
    parser.add_argument('--net_type',
                        default='wide',
                        type=str,
                        help='net type')
    parser.add_argument('--aux_type',
                        default='wide',
                        type=str,
                        help='aux net type')
    parser.add_argument('--widen_factor',
                        default=10,
                        type=int,
                        help='width of model')
    parser.add_argument('--use_FNandWN',
                        default=False,
                        type=bool,
                        help='use Feature Norm and Weight Norm')
    parser.add_argument('--save_name',
                        default='resnet',
                        type=str,
                        help='save model name')
    parser.add_argument('--attack_method',
                        default='pgd',
                        type=str,
                        help='attack method')
    parser.add_argument('--black_attack',
                        action='store_true',
                        help='black attack flag')
    parser.add_argument('--black_name',
                        default='wide-res-dual-geometry',
                        type=str,
                        help='black model name')
    parser.add_argument('--multi_gpu',
                        default=False,
                        type=bool,
                        help='is multi gpus run')
    parser.add_argument('--visual',
                        default=False,
                        type=bool,
                        help='visual feature distribution')
    parser.add_argument('--lr_schedule', default='piecewise',
                        choices=['mylr','superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--lr_max', 
                        default=0.1, 
                        type=float)
    parser.add_argument('--lr_one_drop', 
                        default=0.01, 
                        type=float)
    #~~~~~~~~~~~~~~~~~~~~CRD~~~~~~~~~~~~~~~~~#
    parser.add_argument('--c_k',
                        default=128,
                        type=int,
                        help='number of k')
    parser.add_argument('--c_m',
                        default=128,
                        type=int,
                        help='number of m')
    parser.add_argument('--c_dim',
                        default=640,
                        type=int,
                        help='dim of feat')
    parser.add_argument('--c_moment',
                        default=0.1,
                        type=float,
                        help='moment')
    parser.add_argument('--c_temp',
                        default=0.05,
                        type=float,
                        help='temp')
    parser.add_argument('--ls_factor',
                        default=0.5,
                        type=float,
                        help='label smooth factor')
    parser.add_argument('--aux_net', 
                        default=False, 
                        type=bool, 
                        help='axu net')
    parser.add_argument('--aux_name', 
                        default='resnet', 
                        type=str, 
                        help='aux net name')
    parser.add_argument('--swa', 
                        default=False, 
                        type=bool, 
                        help='swa')
    parser.add_argument('--model_name', 
                        default='model', 
                        type=str, 
                        help='model name from zoo')
    parser.add_argument('--zoo_model', default=False, type='bool', help='model from AutoAttack')
    args = parser.parse_args()

    return args

def get_eval_args(description='AT'):
    parser = argparse.ArgumentParser(
    description='Adversarial Training')

    parser.register('type', 'bool', utils.str2bool)
    parser.add_argument('--resume','-r',action='store_true',help='resume from checkpoint')
    parser.add_argument('--attack', default=True, type='bool', help='attack')
    parser.add_argument('--zoo_model', default=False, type='bool', help='model from AutoAttack')
    parser.add_argument('--model_dir', default='./checkpoint/',type=str, help='model path')
    parser.add_argument('--init_model_pass',default='latest',type=str,help='init model pass')
    parser.add_argument('--net_module',default='base',type=str,help='net module')
    parser.add_argument('--attack_method',default='pgd',type=str, help='attack_method (natural, fgsm, pdg)')
    parser.add_argument('--attack_method_list', type=str)
    parser.add_argument('--log_step', default=10, type=int, help='log_step')
    # dataset dependent
    parser.add_argument('--lr_max', default=0.1, type=float)
    parser.add_argument('--epoch', default=100, type=int, help='num epoch')
    parser.add_argument('--num_classes', default=10, type=int, help='num classes')
    parser.add_argument('--dataset', default='cifar10', type=str,help='dataset')  # concat cascade
    parser.add_argument('--test_batch',default=100,type=int,help='batch size for testing')
    parser.add_argument('--image_size', default=32, type=int, help='image size')
    parser.add_argument('--depth', default=28, type=int, help='depth of model')
    parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
    parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
    parser.add_argument('--use_FNandWN',default=False,type=bool,help='use Feature Norm and Weight Norm')
    parser.add_argument('--save_name', default='wide-res-dual-geometry', type=str, help='save model name')
    parser.add_argument('--black_attack', default=False, type=bool, help='black attack')
    parser.add_argument('--black_name', default='wide-res-dual-geometry', type=str, help='black model name')
    parser.add_argument('--input_channels', default=3, type=int, help='depth of model')
    parser.add_argument('--input_size', default=32, type=int, help='model input size')
    parser.add_argument('--aux_net', default=False, type=bool, help='is axu net')
    parser.add_argument('--aux_name', default='resnet', type=str, help='aux net name')
    parser.add_argument('--aux_depth',default=28,type=int,help='depth of aux model')
    parser.add_argument('--aux_type',default='wide', type=str,help='aux net type')
    parser.add_argument('--bpda', default=False, type=bool, help='bpda attack')
    parser.add_argument('--bpda_type', default='aep', type=str, help='bpda defense net name')
    parser.add_argument('--multi_gpu', default=False, type=bool, help='is multi gpus run')
    parser.add_argument('--benchmark', default=False, type=bool, help='benchmark test')
    #~~~~~~~~~~~~~~~~~~~~CRD~~~~~~~~~~~~~~~~~#
    parser.add_argument('--c_k', default=1, type=int, help='number of k')
    parser.add_argument('--c_m', default=4096, type=int, help='number of m')
    parser.add_argument('--n_data', default=128, type=int, help='number of m')
    parser.add_argument('--c_emDim', default=128, type=int, help='dim of feat')
    parser.add_argument('--c_dim', default=512, type=int, help='dim of feat')
    parser.add_argument('--c_moment', default=0.1, type=float, help='moment')
    parser.add_argument('--c_temp', default=0.1, type=float, help='temp')
    parser.add_argument('--ls_factor', default=0.5, type=float, help='label smooth factor')
    parser.add_argument('--version', default='standard', type=str, help='auto attack')
    parser.add_argument('--model_name', default='Carmon2019Unlabeled', type=str, help='model zoo name')

    args = parser.parse_args()
    return args
