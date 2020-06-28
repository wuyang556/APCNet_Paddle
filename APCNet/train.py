

import os
import time
from tqdm import tqdm
import numpy as np

import paddle.fluid as fluid

from .option import Options
from .models import APCNet
from .reader import get_segmentation_reader
from .utils.loss import SegmentationLosses
from .utils.metrics import SegmentationMetric
from .utils.checkpoints import save_checkpoints, load_checkpoints
from .utils.logger import Logger
from .utils.viz import RecordViz


with fluid.dygraph.guard():
    class Trainer(object):
        def __init__(self, args):
            self.args = args

            # data reader
            kwargs = {
                "base_size": self.args.base_size,
                "crop_size": self.args.crop_size,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5]}
            self.train_reader = get_segmentation_reader(self.args.reader, data_path=args.data_path, mode="train", **kwargs)
            self.val_reader = get_segmentation_reader(self.args.reader, data_path=args.data_path, mode="val", **kwargs)
            self.nclass = self.train_reader.NUM_CLASSES

            # model
            # model = get_segmentation_model(self.args.model, args=args, nclass=self.nclass, backbone=args.backbone,
            #                                backbone_style=args.backbone_style, pu=args.pu,
            #                                dilated=args.dilated, aux=args.aux, aux_weight=args.aux_weight)
            model =APCNet(backbone=self.args.backbone, pretrained=self.args.pretrained, root=self.args.root, nclass=self.args.nclass)
            # lr scheduler and L2 regularization
            self.train_steps = int(len(self.train_reader) / args.batch_size)
            self.val_steps = int(len(self.val_reader) / args.batch_size)
            decay_steps = args.epochs * self.train_steps
            poly_lr = fluid.dygraph.learning_rate_scheduler.PolynomialDecay(
                learning_rate=args.lr,
                decay_steps=decay_steps,
                end_learning_rate=0,
                power=0.9)
            poly_lr_10x = fluid.dygraph.learning_rate_scheduler.PolynomialDecay(
                learning_rate=args.lr*10,
                decay_steps=decay_steps,
                end_learning_rate=0,
                power=0.9)
            l2_regularization = fluid.regularizer.L2DecayRegularizer(regularization_coeff=args.weight_decay)

            # optimizer using different LR
            optimizer_pretrained = fluid.optimizer.MomentumOptimizer(learning_rate=poly_lr,
                                                                     parameter_list=model.pretrained.parameters(),
                                                                     momentum=args.momentum,
                                                                     regularization=l2_regularization)
            optimizer_head = fluid.optimizer.MomentumOptimizer(learning_rate=poly_lr_10x,
                                                               parameter_list=model.head.parameters(),
                                                               momentum=args.momentum,
                                                               regularization=l2_regularization)
            if model.pu:
                poly_lr_10x_pu = fluid.dygraph.learning_rate_scheduler.PolynomialDecay(
                    learning_rate=args.lr*10,
                    decay_steps=decay_steps,
                    end_learning_rate=0,
                    power=0.9)
                self.optimizer_pu = fluid.optimizer.MomentumOptimizer(learning_rate=poly_lr_10x_pu,
                                                                      parameter_list=model.up.parameters(),
                                                                      momentum=args.momentum,
                                                                      regularization=l2_regularization)
            if hasattr(model, "auxlayer"):
                poly_lr_10x_auxlayer = fluid.dygraph.learning_rate_scheduler.PolynomialDecay(
                    learning_rate=args.lr*10,
                    decay_steps=decay_steps,
                    end_learning_rate=0,
                    power=0.9)
                self.optimizer_auxlayer = fluid.optimizer.MomentumOptimizer(learning_rate=poly_lr_10x_auxlayer,
                                                                            parameter_list=model.auxlayer.parameters(),
                                                                            momentum=args.momentum,
                                                                            regularization=l2_regularization)
            # criterions
            self.criterion = SegmentationLosses(aux=args.aux,
                                                aux_weight=args.aux_weight,
                                                nclass=self.nclass, )
            self.model, self.optimizer_pretrained, self.optimizer_head = model, optimizer_pretrained, optimizer_head

            # metrics
            self.metrics = SegmentationMetric(nclass=self.nclass)
            self.best_pred = 0.0

            # resuming checkpoint
            if args.resume:
                if not os.path.isfile(args.checkpoints_path):
                    raise RuntimeError("=> no checkpoint found at '{}'" .format(args.checkpoints_path))
                checkpoint = load_checkpoints(args.checkpoints_path)
                args.start_epoch = checkpoint["epoch"]

                self.model.set_dict(checkpoint["state_dict"])

                if not args.ft:
                    global_step = np.array([(args.start_epoch+1)*self.train_steps])
                    checkpoint["optimizer"]["pretrained"]["global_step"] = global_step
                    checkpoint["optimizer"]["head"]["global_step"] = global_step
                    self.optimizer_pretrained.set_dict(checkpoint["optimizer"]["pretrained"])
                    self.optimizer_head.set_dict(checkpoint["optimizer"]["head"])
                    if self.model.pu:
                        checkpoint["optimizer"][args.pu]["global_step"] = global_step
                        self.optimizer_pu.set_dict(checkpoint["optimizer"][args.pu])
                    if hasattr(self.model, "auxlayer"):
                        checkpoint["optimizer"]["auxlayer"]["global_step"] = global_step
                        self.optimizer_auxlayer.set_dict(checkpoint["optimizer"]["auxlayer"])
                self.best_pred = checkpoint["best_pred"]
                print(f"=> loaded checkpoint '{args.resume}' (epoch {args.epochs})")

            # clear start epoch if fine_tuning
            if args.ft:
                args.start_epoch = 0

            # logger
            # if args.graphs:
            #     self.logger = Logger(filename=os.path.join(args.logger_folder, f"{args.model}_{args.backbone}_{args.backbone_style}_{args.reader}_{args.pu}-{args.dilations}_graphs:{args.graphs}.log"))
            # else:
            #     self.logger = Logger(filename=os.path.join(args.logger_folder,
            #                                                f"{args.model}_{args.backbone}_{args.backbone_style}_{args.reader}_{args.pu}-{args.dilations}.log"))
            self.logger = Logger(filename=os.path.join(args.logger_folder,
                                                       os.path.basename(args.checkpoints_path).replace("_checkpoint.pkl", ".log")))
            if not args.resume:
                self.logger.info(msg="Training Log")
                self.logger.save_configuration(args)
            self.record_vizer = RecordViz(args)

        def training(self, epoch):
            train_loss = 0.0
            self.model.train()
            tbar = tqdm(range(self.train_steps))
            train_batch_reader = self.train_reader.batch(batch_size=self.args.batch_size, shuffle=True)
            st_time = time.time()
            for i in tbar:
                images, targets = next(train_batch_reader)
                images, targets = fluid.dygraph.to_variable(images), fluid.dygraph.to_variable(targets)
                outputs = self.model(images)
                # ce = loss(outputs[0], targets, num_classes=self.nclass)[0]
                if self.args.aux:
                    ce = self.criterion(outputs[0], outputs[1], targets)
                else:
                    ce = self.criterion(outputs[0], targets)
                avg_loss = fluid.layers.mean(ce)
                train_loss += avg_loss.numpy()[0]
                avg_loss.backward()

                self.optimizer_pretrained.minimize(avg_loss)
                self.optimizer_head.minimize(avg_loss)
                if self.model.pu:
                    self.optimizer_pu.minimize(avg_loss)
                if hasattr(self.model, "auxlayer"):
                    self.optimizer_auxlayer.minimize(avg_loss)
                self.model.clear_gradients()
                tbar.set_description('Epoch:%3d Train loss: %.4f lr: %.8f' % (epoch, train_loss / (i + 1), self.optimizer_pretrained.current_step_lr()))
            self.logger.info_train(msg=f"Epoch: {epoch:3d} Train_loss: {train_loss / self.train_steps:.4f} total_time: {time.time()-st_time:.4f}s")
            self.record_vizer.add_data_train(epoch, train_loss / self.train_steps)

            if self.args.no_val:
                # save checkpoint every epoch
                is_best = False
                state_dict = {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": {"pretrained": self.optimizer_pretrained.state_dict(),
                                  "head": self.optimizer_head.state_dict()},
                    "best_pred": self.best_pred,}
                if self.model.pu:
                    state_dict["optimizer"][self.args.pu] = self.optimizer_pu.state_dict()
                if hasattr(self.model, "auxlayer"):
                    state_dict["optimizer"]["auxlayer"] = self.optimizer_auxlayer.state_dict()
                save_checkpoints(state_dict, self.args.checkpoints_path, is_best)

        def validation(self, epoch):
            # Fast test during the training
            is_best = False
            self.metrics.reset()
            self.model.eval()
            tbar = tqdm(range(self.val_steps))
            batch_val_reader = self.val_reader.batch(batch_size=self.args.batch_size, shuffle=False)
            for i in tbar:
                images, targets = next(batch_val_reader)
                images, targets = fluid.dygraph.to_variable(images), fluid.dygraph.to_variable(targets)
                outputs = self.model(images)
                self.metrics.update(targets, outputs[0])
                pixAcc, mIoU = self.metrics.get()
                tbar.set_description('\tVal pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
            self.logger.info_val(msg=f"pixAcc: {pixAcc:.4f} mIoU: {mIoU:.4f}")
            self.record_vizer.add_data_val(pixAcc, mIoU)
            new_pred = (pixAcc+mIoU) / 2
            if new_pred > self.best_pred:
                is_best = True
                self.best_pred = new_pred
                state_dict = {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": {"pretrained": self.optimizer_pretrained.state_dict(),
                                  "head": self.optimizer_head.state_dict()},
                    "best_pred": self.best_pred,}
                if self.model.pu:
                    state_dict["optimizer"][self.args.pu] = self.optimizer_pu.state_dict()
                if hasattr(self.model, "auxlayer"):
                    state_dict["optimizer"]["auxlayer"] = self.optimizer_auxlayer.state_dict()
                save_checkpoints(state_dict, self.args.checkpoints_path, is_best)


    args = Options().parse()
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    print("Model: ", trainer.args.model)
    print("Backbone", trainer.args.backbone)
    print("PU", trainer.args.pu)
    print("Reader: ", trainer.args.reader)
    print("Checkpoint", trainer.args.checkpoints_path)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        # trainer.validation(epoch)
        trainer.training(epoch)
        if not trainer.args.no_val:
            trainer.validation(epoch)
        trainer.record_vizer.plot()

