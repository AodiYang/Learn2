import os
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))


    if not os.path.exists(args.test_patient.replace('test',args.test_model_name)):
        os.makedirs(args.test_patient.replace('test',args.test_model_name))
        print('Create path : {}'.format(args.test_patient.replace('test',args.test_model_name)))


    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'fig')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

    data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             num_workers=args.num_workers)

    solver = Solver(args, data_loader)
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help="train | test")
    parser.add_argument('--load_mode', type=int, default=0, help="0 | 1")
    parser.add_argument('--test_model_name', type=str, default='test-WGAN-VGG')
    parser.add_argument('--data_path', type=str, default='./data1/')
    parser.add_argument('--saved_path', type=str, default='./npy_img/')
    parser.add_argument('--save_path', type=str, default='./save/')
    parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--result_fig', type=bool, default=True)

    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)

    # if CT is abdomen scans 

    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)
    '''
    # if CT is chest scans 
    parser.add_argument('--trunc_min', type=float, default=-1350.)
    parser.add_argument('--trunc_max', type=float, default=150)
    '''

    parser.add_argument('--transform', type=bool, default=False)
    # if patch training, batch size is (--patch_n x --batch_size)
    parser.add_argument('--patch_n', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--print_iters', type=int, default=100)
    parser.add_argument('--decay_iters', type=int, default=3000)
    parser.add_argument('--save_iters', type=int, default=10000)
    parser.add_argument('--test_iters', type=int, default=26800)

    parser.add_argument('--n_d_train', type=int, default=4)

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lambda_', type=float, default=10.0)

    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--multi_gpu', type=bool, default=False)

    args = parser.parse_args()
    print(args)
    main(args)