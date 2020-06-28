import argparse
import os

fcn_resume = None
pspnet_resume = None
deeplabv3_resum = None


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PaddlePaddle Segmentation')
        # model and dataset 
        parser.add_argument('--model', type=str, default='apcnet',
                            help='model name (default: fcn)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('pretrained', type=bool, default=True,
                            help='load the pretrained model')

        parser.add_argument('--reader', type=str, default='voc',
                            help='reader name (default: voc)')
        parser.add_argument('nclass', type=int, default=21,
                            help='location of pretrained models')
        parser.add_argument('--data-path', type=str, default=r"~/work/datasets",
                            help='reader path')
        parser.add_argument('--checkpoints-path', type=str, default=None,
                            help='checkpoints path')
        parser.add_argument('--base-size', type=int, default=None,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=None,
                            help='crop image size')
        parser.add_argument('--train-split', type=str, default='train',
                            help='dataset train split (default: train)')
        # training hyper params
        parser.add_argument('--epochs', type=int, default=None, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            training (default: auto)')
        parser.add_argument('--test-batch-size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')
        # optimizer params
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar='M', help='w-decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=bool, default=False,
                            help='put the path to resuming file if needed')
        parser.add_argument('--resume-path', type=str, default=None,
                            help='put the path to resuming file if needed')
        # finetuning pre-trained models
        parser.add_argument('--ft', type=bool, default=False,
                            help='finetuning on a different dataset')
        # evaluation option
        parser.add_argument('--split', default='val')
        parser.add_argument('--mode', default='testval')
        parser.add_argument('--ms', action='store_true', default=False,
                            help='multi scale & flip')
        parser.add_argument('--no-val', action='store_true', default=False,
                            help='skip validation during training')
        parser.add_argument('--save-folder', type=str, default='results',
                            help='path to save images')
        # logger
        parser.add_argument('--logger-folder', type=str, default=None,
                            help='path to save images')

        # parallel unit configuration
        parser.add_argument('--dilations', type=tuple, default=(1,2,4,8),
                            help='dilations')
        # graph size
        parser.add_argument('--graphs', type=tuple, default=(256,128,64,32),
                            help='dilations')

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()

        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            epoches = {
                'coco': 30,
                'citys': 240,
                'voc': 50,
                'voc_aug': 50,
                'pcontext': 80,
                'ade20k': 120,
            }
            args.epochs = epoches[args.reader.lower()]
        if args.batch_size is None:
            batch_size = {
                'coco': 8,
                'citys': 8,
                'voc': 1,
                'voc_aug': 8,
                'pcontext': 8,
                'ade20k': 8,
            }
            args.batch_size = batch_size[args.reader.lower()]
        if args.test_batch_size is None:
            args.test_batch_size = args.batch_size
        if args.lr is None:
            lrs = {
                'coco': 0.01,
                'citys': 0.01,
                'voc': 0.0001,
                'voc_aug': 0.001,
                'pcontext': 0.001,
                'ade20k': 0.01,
            }
            args.lr = lrs[args.reader.lower()] / 16 * args.batch_size
        if args.checkpoints_path is None:
            if args.dilations is not None and args.model == "grnet":
                args.checkpoints_path = f"./work/{args.model}_{args.backbone}-{args.backbone_style}_{args.reader}_{args.pu}-{args.dilations}_graphs-{args.graphs}_checkpoint.pkl"
            elif args.dilations is not None and args.model != "grnet":
                args.checkpoints_path = f"./work/{args.model}_{args.backbone}-{args.backbone_style}_{args.reader}_{args.pu}-{args.dilations}_checkpoint.pkl"
            else:
                args.checkpoints_path = f"./work/{args.model}_{args.backbone}-{args.backbone_style}_{args.reader}_checkpoint.pkl"
        if args.resume and args.ft:
            args.resume_path = args.checkpoints_path.replace("voc", "voc_aug")
        else:
            args.resume_path = args.checkpoints_path

        if args.logger_folder is None:
            args.logger_folder = "./work/log"
            if not os.path.exists(args.logger_folder):
                os.makedirs(args.logger_folder)

        if args.base_size is None or args.crop_size is None:
            size = {
                "coco": (520, 512),
                "citys": (1024, 800),
                "voc": (520, 512),
                "voc_aug": (520, 512),
                "pcontext": (520, 512),
                "ade20k": (520, 512)
            }
            args.base_size, args.crop_size = size[args.reader.lower()]

        print(args)
        return args


if __name__ == '__main__':
    option = Options()
    args = option.parse()
    print(args.resume)
    print(args.ft)
    print(args.reader)
    print(args.checkpoints_path)
    print(args.logger_folder)