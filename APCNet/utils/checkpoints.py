# coding = utf-8
# python 3.7.3
# created by wuyang on 2020/3/25
import pickle
from paddle.fluid.framework import Variable
from paddle.fluid import core
import shutil


def save_checkpoints(params: dict, model_path: str, is_best=False):
    def toNumpy(param):
        for k, v, in param.items():
            if isinstance(v, (Variable, core.VarBase)):
                param[k] = v.numpy()
            else:
                param[k] = v

    with open(model_path, "wb") as file:
        toNumpy(params["state_dict"])
        # for k, v in params["state_dict"].items():
        #     if isinstance(v, (Variable, core.VarBase)):
        #         params["state_dict"][k] = v.numpy()
        #     else:
        #         params["state_dict"][k] = v
        for k in params["optimizer"].keys():
            toNumpy(params["optimizer"][k])
        # for k, v in params["optimizer"].items():
        #     if isinstance(v, (Variable, core.VarBase)):
        #         params["optimizer"][k] = v.numpy()
        #     else:
        #         params["optimizer"][k] = v

        pickle.dump(params, file=file)
        file.close()
    if is_best:
        shutil.copy(model_path, model_path.replace("checkpoint", "best_pred"))


def load_checkpoints(model_path):
    with open(model_path, "rb") as file:
        checkpoints = pickle.load(file)
    return checkpoints


if __name__ == '__main__':
    import paddle
    import paddle.fluid as fluid
    from PdSeg.models.pspnet import PSPNet

    with fluid.dygraph.guard():
        model = PSPNet(nclass=21, backbone="resnet50")
        optimizer = fluid.optimizer.SGDOptimizer(learning_rate=.1, parameter_list=model.parameters())
        print(type(model.state_dict()))
        print(optimizer.state_dict())
        print(model.state_dict().keys())
        print(model.state_dict()['pretrained.conv1.0.weight'])
        state_dict = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_pred": 1
        }
        save_checkpoints(state_dict, "./checkpoints.pdparams")

        checkpoints = load_checkpoints("./checkpoints.pdparams")

        model.set_dict(checkpoints["state_dict"])
        optimizer.set_dict(checkpoints["optimizer"])
        # print(optimizer.state_dict())
        print(model.state_dict()['pretrained.conv1.0.weight'])
