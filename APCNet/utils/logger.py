# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/27
import time
from datetime import datetime


class Logger(object):
    """
    saving training information.
    """
    def __init__(self, filename, filemode="a"):
        self.filename = filename
        self.filemode = filemode

    def info(self, msg):
        msg = str(datetime.fromtimestamp(time.time())) + "\t" + msg + "\n"
        self.write_msg(msg)

    def info_train(self, msg):
        msg = str(datetime.fromtimestamp(time.time())) + "\t" + msg + "\t"
        self.write_msg(msg)

    def info_val(self, msg):
        msg = str(msg) + "\n"
        self.write_msg(msg)

    def write_msg(self, msg):
        with open(self.filename, self.filemode) as file:
            file.write(msg)
            file.close()

    def save_configuration(self, args):
        with open(self.filename, self.filemode) as file:
            file.write(f"parameters configuration\n")
            file.write(f"\tModel: {args.model} \tbackbone: {args.backbone} \tbackbone_style:{args.backbone_style} \tpu: {args.pu}-{args.dilations} \tdilated: {args.dilated} \tgraphs-{args.graphs}\n")
            file.write(f"\tdataset: {args.reader} \tepochs: {args.epochs} \tbatch_size: {args.batch_size}\n")
            file.write(f"\tbase_size: {args.base_size} \tcrop_size: {args.crop_size}\n")
            file.write(f"\tlr: {args.lr} \tlr_scheduler: {args.lr_scheduler}\n")
            file.write(f"\taux: {args.aux}\n")
            file.close()
