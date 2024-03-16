import csv
import math
import os
import pickle
import random
from functools import partial
from typing import Union

import hydra
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR

import CustomResNet

## [TODO]: Commenting and documentation


# region Neural Networks
class Init_NN:
    # "kaiming_uniform_" is the default initialization method from PyTroch
    __kwargs_scaling_factors = {
        "kaiming_uniform_": {"a": math.sqrt(5)},
        "kaiming_normal_": {"a": math.sqrt(5)},
        "xavier_uniform_": {"gain": 1},
        "xavier_normal_": {"gain": 1},
    }

    def __init_weights_apply(
        self, module: nn.Module, init_method_class: any, kwargs: dict[str, any]
    ) -> None:
        init_method_class(module.weight, **kwargs)

        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)

    def _init_weights(self, module: nn.Module, init_method: str) -> None:
        init_method_class = getattr(nn.init, init_method)

        # Apply init_method to weight/bias on Linear and Conv2D Layer.
        if isinstance(module, nn.Sequential):
            for sub_module in module:
                if isinstance(sub_module, (nn.Linear, nn.Conv2d)):
                    self.__init_weights_apply(
                        sub_module,
                        init_method_class,
                        self.__kwargs_scaling_factors[init_method],
                    )
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            self.__init_weights_apply(
                module, init_method_class, self.__kwargs_scaling_factors[init_method]
            )


class Custom_Network(nn.Module, Init_NN):
    """
    Build custom Neural Network describe in "./configs/model/custom"
    """

    def __init__(self, criterion, layers, init_method):
        super().__init__()

        self.criterion = getattr(nn, criterion)()
        self.init = init_method
        layers_ = []

        # Create the layer based on the configuration
        for layer_cfg in layers:
            if layer_cfg.arguments is not None:
                layer = getattr(nn, layer_cfg.type)(**layer_cfg.arguments)
            else:
                layer = getattr(nn, layer_cfg.type)()

            layers_.append(layer)

            if layer_cfg.activation is not None:
                layers_.append(getattr(nn, layer_cfg.activation)())

        self.sequential = nn.Sequential(*layers_)

        init_weight_with_config = partial(self._init_weights, init_method=init_method)
        self.apply(init_weight_with_config)

    def forward(self, input):
        x = self.sequential(input)
        return x


class McMahan_2NN(nn.Module, Init_NN):
    """
    Define a 2 hidden layer MLP for MNIST digit recognition.

    Attributes
    ----------
    flatten : nn.Flatten
        Flatten images.
    sequential : nn.Sequential
        Define internal structure of the MLP.
    softmax : nn.Softmax
        Softmax the output of the MLP.
    """

    def __init__(
        self, data_size, label_size, hidden_layer_size, activation, init_method
    ) -> None:
        super().__init__()

        # loss(output, target), target's format logits
        self.criterion = nn.CrossEntropyLoss()
        self.init = init_method

        # alias for verbal parameters
        activation = getattr(nn, activation)
        hl_size = hidden_layer_size
        data = data_size

        # input.shape => (batch_size, 1, 28, 28)
        self.sequential = nn.Sequential(
            # x.shape => (batch_size, 1*28*28)
            nn.Flatten(),
            # x.shape => (batch_size, 200)
            nn.Linear(data.c * data.x * data.y, hl_size),
            activation(),
            # x.shape => (batch_size, 200)
            nn.Linear(hl_size, hl_size),
            activation(),
            # x.shape => (batch_size, 10)
            nn.Linear(hl_size, label_size),
            nn.Softmax(dim=1),
        )

        init_weight_with_config = partial(self._init_weights, init_method=init_method)
        self.apply(init_weight_with_config)

    def forward(self, input):
        x = self.sequential(input)
        return x


class Custom_ResNet(nn.Module, Init_NN):
    def __init__(self, data_size, label_size, layers, activation, init_method):
        super().__init__()

        block_type, layer_structure = CustomResNet.resnet_depths_to_config(layers)

        self.resnet = CustomResNet.ResNet(
            channels=data_size.c,
            classes=label_size,
            block=block_type,
            layers=layer_structure,
            nonlin=activation,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.init = init_method

        init_weight_with_config = partial(self._init_weights, init_method=init_method)
        self.apply(init_weight_with_config)

    def forward(self, input):
        x = self.resnet(input)
        return x


class McMahan_CNN(nn.Module, Init_NN):
    """
    Define a convolutional neural network for MNIST digit recognition.

    Attributes
    ----------
    sequential : nn.Sequential
        Define internal structure of the CNN.
    softmax : nn.Softmax
        Softmax the output of the CNN.
    """

    def __init__(self, data_size, label_size, layers, activation, init_method):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.init = init_method

        # alias for verbal parameters
        activation = getattr(nn, activation)
        hl_size = layers.hidden_layer_size
        data = data_size
        conv1 = layers.conv1
        conv2 = layers.conv2

        # Con2d -> (B, C', (H-kernel_size+1), (W-kernel_size+1))
        # MaxPool2d -> (B, C, (H/kernel_size), (W/kernel_size))
        size_x = (data.x - conv1.kernel_size + 3) / 2 + 1
        size_x = (size_x - conv2.kernel_size + 3) / 2 + 1
        size_x = int(size_x)

        size_y = (data.y - conv1.kernel_size + 3) / 2 + 1
        size_y = (size_y - conv2.kernel_size + 3) / 2 + 1
        size_y = int(size_y)

        # input.shape => (batch_size, 1, 28, 28)
        self.sequential = nn.Sequential(
            # (batch_size, 32, 26, 26)
            nn.Conv2d(
                data.c,
                conv1.out_channels,
                conv1.kernel_size,
                padding=1,
            ),
            nn.BatchNorm2d(conv1.out_channels),
            activation(),
            # (batch_size, 32, 14, 14)
            nn.MaxPool2d(2, padding=1),
            # (batch_size, 64, 12, 12)
            nn.Conv2d(
                conv1.out_channels,
                conv2.out_channels,
                conv2.kernel_size,
                padding=1,
            ),
            nn.BatchNorm2d(conv2.out_channels),
            activation(),
            # (batch_size, 64, 7, 7)
            nn.MaxPool2d(2, padding=1),
            # (batch_size, 64 * 7 * 7)
            nn.Flatten(),
            # (batch_size, 512)
            nn.Linear(
                conv2.out_channels * size_x * size_y,
                hl_size,
            ),
            activation(),
            # (batch_size, 10)
            nn.Linear(
                hl_size,
                label_size,
            ),
        )

        init_weight_with_config = partial(self._init_weights, init_method=init_method)
        self.apply(init_weight_with_config)

    def forward(self, input):
        x = self.sequential(input)
        return x


# endregion


# region Dataset
class Custom_Dataset:
    """
    Load dataset from custom settings (Only dataset from pytorch)
    """

    def __init__(
        self,
        name,
        download,
        shuffle,
        mean,
        var,
        batch_size,
        data_size: None,
        label_size: None,
    ):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, var)]
        )  # normalise dataset

        self.train_dataset = getattr(torchvision.datasets, name)(
            root="./data", download=download, train=True, transform=transform
        )

        self.custom_trainLoader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle
        )

    def get_trainLoader(self):
        return self.custom_trainLoader


class MnistDataset:
    """
    MNIST Dataset loader.

    Attributes
    ----------
    mnist_train : MNIST
        Load train data from MNIST dataset.
    mnist_loader : DataLoader
        Training data loader.

    Methods
    -------
    get_trainloader()
        Return the train loader.
    """

    def __init__(
        self,
        download,
        shuffle,
        mean,
        var,
        batch_size,
        data_size: None,
        label_size: None,
    ):
        mnist_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, var),
            ]  # normalise MNIST dataset
        )

        self.mnist_train = torchvision.datasets.MNIST(
            root="./data",
            download=download,
            train=True,
            transform=mnist_transform,
        )

        self.mnist_trainLoader = torch.utils.data.DataLoader(
            self.mnist_train, batch_size=batch_size, shuffle=shuffle
        )

    def get_trainLoader(self):
        return self.mnist_trainLoader


# endregion


# region Client
class SimpleClient:
    """
    Define a client for federated learning.

    Attributes
    ----------
    trainLoader : DataLoader
        Training data loader.
    model : any
        Model of NN.
    loss : nn.CrossEntropyLoss
        Loss function used for the learning

    Methods
    -------
    compute_true_gradient()
        Compute the gradient of model parameters based on the loss.
    """

    def __init__(
        self,
        prune,
        model,
        trainLoader,
        label_size,
        device="cpu",
    ) -> None:
        """
        Parameters
        ----------
        trainLoader : DataLoader
            Training data loader.
        model : any
            Model of NN.
        device : torch.device
            Device on which model will run.
        """
        self.trainLoader = iter(trainLoader)
        self.model = model
        self.device = device
        self.label_class = label_size
        self.prune = prune.type
        self.percentage = prune.percentage

    def compute_true_gradient(self):
        """
        Compute the gradient of model parameters based on the loss.
        Randomly set a percentage of gradients to zero.
        """
        # Load data and label from the MNIST dataset and send them
        # to the device memory.
        data, label = next(self.trainLoader)
        data, label = data.to(self.device), label.to(self.device)

        # The data is normalized within the whole dataset.
        # We re-normalize each image individually (maybe use Layer Norm or Instance Norm ?)
        data = normalize_torch(data)
        self.model.eval()  # 在eval模式下均值和方差均为0
        preds = self.model(data)

        # loss = self.model.criterion(
        #     preds, F.one_hot(label, num_classes=self.label_class).to(torch.float32)
        # )

        loss = self.model.criterion(preds, label)

        gradient = torch.autograd.grad(loss, self.model.parameters())
        gradient = [layer.detach().clone() for layer in gradient]

        print("CLIENT INFO: [ ##################### ] 100%\n")

        ## Why prunning an entire layer ??
        ## [TODO] Check how prunning works
        if self.prune == "random":
            print(
                f"Pruning mode: {self.prune} | Pruning percentage: {self.percentage}\n"
            )

            # Randomly set a percentage of gradients to zero
            for layer_grad in gradient:
                if torch.rand(1).item() < self.percentage:
                    layer_grad.zero_()

        if self.prune == "small":
            print(
                f"Pruning mode: {self.prune} | Pruning percentage: {self.percentage}\n"
            )

            # Prune the gradient by setting some elements to zero by magnitudes order.
            gradient_magnitudes = [layer.norm().item() for layer in gradient]

            sorted_gradients = sorted(
                zip(gradient_magnitudes, gradient), key=lambda x: x[0]
            )

            retain_threshold = sorted_gradients[
                int(len(sorted_gradients) * self.percentage)
            ][0]

            # for i, (grad_mag, layer_grad) in enumerate(sorted_gradients):
            #     if grad_mag <= retain_threshold:
            #         gradient[i].zero_()

            # Zero-out gradients below the threshold
            [
                gradient[i].zero_()
                for i, (grad_mag, _) in enumerate(sorted_gradients)
                if grad_mag <= retain_threshold
            ]

        return gradient, data, label


# endregion


# region Attacker


# stop training when the distance continue to increase
class SimpleEarlyStopping:
    LoopStopCrit = {
        "iteration": 0,
        "threshold": 1,
        "variation": 2,
        "best": 3,
    }

    def __init__(self, type: str, value: Union[int, float], max_iter: int = 500):
        self.loop_criterion = self.LoopStopCrit[type]
        self.loop_value = value
        self.loop_max_iter = self.loop_value if self.loop_criterion == 0 else max_iter

        self.reset()

    def reset(self):
        # init loop start and stop values
        # NB: In mode "iteration", the first condition is alway true;
        #     the second condition is the stop condition.
        self.early_stop = False
        self.loop_current = self.loop_value + 1
        self.loss_prec = 0
        self.best_score = None
        self.counter = 0
        self.j = 0

    def update(self, loss):
        # Update loop current value
        if self.loop_criterion == 1:  # Threshold based distance
            self.loop_current = loss
        elif self.loop_criterion == 2:  # Variation of distance
            self.loop_current = abs(self.loss_prec - loss)
        elif self.loop_criterion == 3:  # Best
            if self.best_score is None:
                self.best_score = loss
            elif loss > self.best_score:
                self.counter += 1
            else:
                self.best_score = loss
                self.counter = 0

        self.loss_prec = loss
        self.j += 1

        self.early_stop = (
            (self.loop_current < self.loop_value)
            or (self.j >= self.loop_max_iter)
            or (self.counter >= self.loop_value)
        )


class GradAttacker:
    """
    Attack on differentiable NN to reconstruct private data.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(
        self,
        attacker_criterion,
        optimizer,
        loss_function,
        dummy_batch_size,
        dummy_data_size,
        dummy_label_size,
        model,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model = model
        self.optimizer = getattr(torch.optim, optimizer.type)
        self.lr = optimizer.learning_rate
        self.scheduler = optimizer.scheduler

        self.batch_size = dummy_batch_size
        self.cfg_data = dummy_data_size
        self.cfg_label = dummy_label_size

        self.early_stopping = SimpleEarlyStopping(
            type=attacker_criterion.type,
            value=attacker_criterion.value,
            max_iter=attacker_criterion.max_iteration,
        )

        self.iter = attacker_criterion.value

        # Status Check
        print()
        print("ATTACKER INFO [ ##################### ] 100%")
        print(f"Early Stop Type: {attacker_criterion.type}")
        print(f"Max Iteration: {self.iter}")
        print(f"Optimizer Used: {optimizer.type}")
        print(f"Initial Learning Rate: {self.lr}")
        print(f"Scheduler Status: {self.scheduler}")
        print(f"Loss Function: {loss_function}")

        if loss_function == "cosine_similarity":
            self.compute_distance_between_grad = self.cosine_similarity
        elif loss_function == "MSE":
            self.compute_distance_between_grad = self.MSE
        else:
            self.compute_distance_between_grad = self.cosine_similarity
            print(
                'WARN : Invalid Loss function. Should be one of "cosine_similarity" or "MSE"'
                + '\nDefault is set to "consine_similarity"'
            )

    def attack_gradient(self, true_gradient, shared_data={}):
        # PyTorch automatically calculates the gradient of the loss with respect to dummy_data
        # and stores these gradients in the .grad attribute of dummy_data.
        # call the backward() method, PyTorch automatically calculates the gradient of the loss relative to
        # all tensors with the requires_grad=True setting.

        # init data
        dummy_data, dummy_label = self.init_dummy_data()

        # init optimizer
        if "true_label" in shared_data:
            optimizer = self.optimizer([dummy_data], lr=self.lr)
            print("True Label is shared")
        else:
            optimizer = self.optimizer([dummy_data, dummy_label], lr=self.lr)
            print("True Label is not shared")
        # use scheduler
        if self.scheduler:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.iter, eta_min=0.00001)

        # init history data
        history_loss = np.array([])

        history_img = np.zeros(
            (0, self.batch_size, self.cfg_data.c, self.cfg_data.x, self.cfg_data.y)
        )

        self.early_stopping.reset()
        counter = 0
        loss = None

        print()
        print("START TRAINING ATTACK GRADIENT...")

        while not self.early_stopping.early_stop:
            counter += 1
            if counter % 5 == 0:
                print(
                    f"Current Iteration: {counter} | Current Learning Rate: {optimizer.param_groups[0]['lr']:.3f} | Current Grad Loss: {loss:.4f}",
                    end="\r",
                )

            def closure():
                optimizer.zero_grad()

                # compute_fake
                fake_grad = self.compute_fake_gradient(
                    dummy_data, dummy_label, shared_data=shared_data
                )

                # compute_distance
                loss = self.compute_distance_between_grad(true_gradient, fake_grad)

                # Compute backward gradient
                loss.backward(retain_graph=True)
                return loss

            # Optimise dummy data & dummy label
            loss = optimizer.step(closure).item()
            
            # Update scheduler
            if self.scheduler:
                scheduler.step()


            # Update history data
            history_loss = np.append(history_loss, loss)

            history_img = np.append(
                history_img,
                to_numpy_array(dummy_data.clone().unsqueeze(0)),
                axis=0,
            )

            self.early_stopping.update(loss)

        history = {
            "distance": history_loss,
            "data": history_img,
        }

        return dummy_data, dummy_label, history

    def init_dummy_data(self):
        # initialize random images (B,C,W,H) from a normal distribution

        dummy_data = torch.randn(
            self.batch_size,
            self.cfg_data.c,
            self.cfg_data.x,
            self.cfg_data.y,
            requires_grad=True,
            device=self.device,
        )

        dummy_label = torch.randn(
            self.batch_size,
            self.cfg_label,
            requires_grad=True,
            device=self.device,
        )

        return dummy_data, dummy_label

    def compute_fake_gradient(self, dummy_data, dummy_label, shared_data={}):
        def total_variation_loss(c):
            x = c[:, :, 1:, :] - c[:, :, :-1, :]
            y = c[:, :, :, 1:] - c[:, :, :, :-1]
            loss = torch.sum(torch.abs(x)) + torch.sum(torch.abs(y))
            return loss

        preds = self.model(dummy_data)

        if "true_label" in shared_data:
            loss = self.model.criterion(
                preds,
                F.one_hot(shared_data["true_label"], num_classes=self.cfg_label).to(
                    torch.float32
                ),
            )
        else:
            loss = self.model.criterion(
                preds,
                F.softmax(dummy_label, dim=-1),
            )

        # Added total variation loss
        loss += total_variation_loss(dummy_data)

        gradient = torch.autograd.grad(
            loss,
            self.model.parameters(),
            create_graph=True,
        )

        return gradient

    def MSE(self, true_grad, fake_grad):
        layer_distance = [
            ((fg - tg) ** 2).sum() for tg, fg in zip(true_grad, fake_grad)
        ]

        loss = sum(layer_distance)
        return loss

    def cosine_similarity(self, gradient_data, gradient_rec):
        scalar_product = gradient_rec[0].new_zeros(1)
        rec_norm = gradient_rec[0].new_zeros(1)
        data_norm = gradient_rec[0].new_zeros(1)
        count = 0

        for rec, data in zip(gradient_rec, gradient_data):
            if data.norm().item() != 0:  # Check if data gradient is non-zero
                scalar_product += (rec * data).sum()
                rec_norm += rec.pow(2).sum()
                data_norm += data.pow(2).sum()
                count += 1

        if count == 0:
            return 0  # If no non-zero gradients in gradient_data, return 0 similarity

        objective = 1 - scalar_product / (rec_norm.sqrt() * data_norm.sqrt())

        return objective


# endregion


# region Utils
def set_seed(seed=42):
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_nb_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def to_numpy_array(var):
    return var.detach().cpu().numpy()


def normalize_torch(data):
    for i in range(data.shape[0]):
        min_ = data[i].min()
        max_ = data[i].max()
        data[i] = (data[i] - min_) / (max_ - min_)
    return data


def normalize(data):
    min_ = data.min(axis=(1, 2, 3))
    max_ = data.max(axis=(1, 2, 3))
    data = np.array(
        [
            (data[i] - min_) / (max_ - min_)
            for i, (min_, max_) in enumerate(zip(min_, max_))
        ]
    )
    return data


def PSNR(A, B, dynamic):
    EQM = np.mean((A - B) ** 2, axis=(1, 2, 3))
    return np.where(EQM == 0, -np.inf, 10 * np.log10(dynamic**2 / EQM))


def visualisation(cfg, fake_data, data, history=None, model=None):
    # Copy fake_data to host memory if cuda is used and
    # convert to a numpy array.

    data = np.transpose(data, (0, 2, 3, 1))
    fake_data = np.transpose(fake_data, (0, 2, 3, 1))
    # difference = np.abs(np.sum(fake_data - data, axis=3))
    difference = normalize(np.abs(fake_data - data))

    if history is not None:
        x = np.arange(1, history["distance"].shape[0] + 1)
        yscale = "log" if cfg.yscale_log else "linear"

        if cfg.total_distance:
            plt.figure()
            plt.plot(x, history["distance"])
            plt.title("Difference between True Grad and Fake Grad")
            plt.xlabel("Iteration")
            plt.xlim(1, history["distance"].shape[0])
            plt.ylabel(yscale)
            plt.yscale(yscale)
            plt.grid()

        if cfg.display_images:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

            if cfg.normalize:
                data = normalize(data)
                fake_data = normalize(fake_data)
                history["data"] = np.array(
                    [normalize(elem) for elem in history["data"]]
                )

            square_size = int(math.sqrt(data.shape[0]))

            data_as_grid = (
                torchvision.utils.make_grid(
                    torch.from_numpy(data.transpose(0, 3, 1, 2)), nrow=square_size
                )
                .numpy()
                .transpose(1, 2, 0)
            )
            ax1.imshow(data_as_grid, cmap="gray", vmin=0, vmax=1)
            ax1.set_title("Original data")
            ax1.axis("off")

            image_diff_as_grid = (
                torchvision.utils.make_grid(
                    torch.from_numpy(difference.transpose(0, 3, 1, 2)),
                    # torch.from_numpy(difference[..., np.newaxis].transpose(0, 3, 1, 2)),
                    nrow=square_size,
                )
                .numpy()
                .transpose(1, 2, 0)
            )

            image_diff = ax2.imshow(image_diff_as_grid, cmap="gray")

            fig.colorbar(image_diff, ax=ax2)
            ax2.set_title("Difference")
            ax2.axis("off")

            fake_data_as_grid = (
                torchvision.utils.make_grid(
                    torch.from_numpy(fake_data.transpose(0, 3, 1, 2)), nrow=square_size
                )
                .numpy()
                .transpose(1, 2, 0)
            )
            ax3.imshow(fake_data_as_grid, cmap="gray", vmin=0, vmax=1)
            ax3.set_title("Recovered data")
            ax3.axis("off")

            first_history_data_as_grid = (
                torchvision.utils.make_grid(
                    torch.from_numpy(history["data"][0]), nrow=square_size
                )
                .numpy()
                .transpose(1, 2, 0)
            )
            img = ax4.imshow(first_history_data_as_grid, cmap="gray", animated=True)
            ax4.axis("off")

            def animation(frame):
                frame_hist_data_as_grid = (
                    torchvision.utils.make_grid(
                        torch.from_numpy(history["data"][frame]), nrow=square_size
                    )
                    .numpy()
                    .transpose(1, 2, 0)
                )
                img.set_array(frame_hist_data_as_grid)  # Update the image data
                ax4.set_title(
                    f"Update: {frame*1 + 1}"
                )  # Update the title with the frame number
                return (img,)

            # Animation must be kept in memory
            ani = anim.FuncAnimation(  # noqa: F841
                fig,
                animation,
                frames=history["data"].shape[0],
                interval=cfg.duration // history["data"].shape[0],
            )

            fig.tight_layout()
    else:
        if cfg.display_images:
            fig = plt.figure()

            if cfg.normalize:
                data = normalize(data)
                fake_data = normalize(fake_data)

            plt.subplot(2, 2, 1)
            plt.imshow(data[0], cmap="gray", vmin=0, vmax=1)
            plt.title("Original data")
            plt.axis("off")

            plt.subplot(2, 2, (2, 4))

            image_diff = plt.imshow(difference, cmap="gray")
            plt.colorbar()
            plt.title("Difference")
            plt.axis("off")

            plt.subplot(2, 2, 3)
            plt.imshow(fake_data[0], cmap="gray", vmin=0, vmax=1)
            plt.title("Recovered data")
            plt.axis("off")

            fig.tight_layout()

    plt.show()


def zero_percentage(gradient, conv=False):
    if conv:
        print(gradient[4])
        zero_count = gradient[4].eq(0).sum().item()
        total_elements = gradient[4].numel()
        zero_percentage = (zero_count / total_elements) * 100
    else:
        zero_count = sum(layer.eq(0).sum().item() for layer in gradient)
        total_elements = sum(layer.numel() for layer in gradient)
        zero_percentage = (zero_count / total_elements) * 100
    return zero_percentage


def write_to_csv(data, file_path, fieldnames):
    try:
        file_exists = os.path.exists(file_path)

        with open(file_path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=';')

            # Write header only if the file is created
            if not file_exists:
                writer.writeheader()

            # Write data
            writer.writerow(data)
    except Exception as e:
        print(f"Error: {e}")


def iDLG(true_gradient):
    last_tensor = true_gradient[-1]
    negative_indices = [index for index, value in enumerate(last_tensor) if value < 0]
    return torch.tensor(negative_indices)


def load_previous_gradients():
    try:
        with open("./Client_Server/previous_gradients.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def save_gradients(gradients):
    with open("previous_gradients.pkl", "wb") as f:
        pickle.dump(gradients, f)
        f.close()


# endregion


# region Main
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    seed = cfg.seed if "seed" in cfg else 42
    set_seed(seed)

    visualisation_enabled = "visualisation" in cfg and cfg.visualisation is not None

    if visualisation_enabled:
        print(f"\nConfiguration used :\n\n{OmegaConf.to_yaml(cfg)}")

    device_name = torch.device(device())
    if visualisation_enabled:
        print(f"Using {device_name} device\n")

    ## Model
    # Instanciate the model class from the config
    model = instantiate(cfg.model).to(device_name)

    if visualisation_enabled:
        print("############ Model Architecture ############\n")
        print(f"Model Architecture: {model}\n")
        print(f"Model Parameters: {count_nb_parameters(model)}\n")
        print("############################################\n")

    ## Data Set
    data = instantiate(cfg.dataset)
    trainLoader = data.get_trainLoader()

    ## Client
    client = instantiate(
        cfg.client,
        model=model,
        trainLoader=trainLoader,
        label_size=cfg.dataset.label_size,
        device=device_name,
    )

    ## Attacker
    attacker = instantiate(cfg.attacker, model=model, device=device_name)

    ## Compute true gradient
    true_gradient, data, label = client.compute_true_gradient()
    zero_gradient = zero_percentage(true_gradient)
    
    # conv2_grad = true_gradient[4]  # for McMahan_CNN
    # with open('conv2_grad_train.pickle', 'wb') as f:
    #     pickle.dump(conv2_grad, f)
   
    if cfg.batch_size == 1:
        assert int(iDLG(true_gradient)) == int(label), "Failed to Extract True Label!"
        print("Label is", int(label))
        print("Success to Extract True Label!")

    shared_data = {"true_label": label}

    ## Gradient descent
    fake_data, fake_label, history = attacker.attack_gradient(
        true_gradient, shared_data=shared_data
    )

    ## Visualize data
    if "true_label" not in shared_data:
        recovered = to_numpy_array(fake_label).argmax(axis=1)
        original = to_numpy_array(label)
        print(f"Recover Label : {recovered}, original was : {original}")

    data = to_numpy_array(data)
    fake_data = to_numpy_array(fake_data)
    RL = PSNR(data, fake_data, dynamic=1)

    if visualisation_enabled:
        print(f"PSNR: {RL}")
        visualisation(cfg.visualisation, fake_data, data, history=history, model=model)

    ## Save Results
    if cfg.save_data.enable:
        path = (
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/"
        )  # Hydra output directory

        if cfg.save_data.true_gradient:
            torch.save(true_gradient, path + "true_gradient.pt")
        if cfg.save_data.data:
            torch.save(data, path + "data.pt")
        if cfg.save_data.label:
            torch.save(label, path + "label.pt")
        if cfg.save_data.fake_data:
            torch.save(fake_data, path + "fake_data.pt")
        if cfg.save_data.fake_label:
            torch.save(fake_label, path + "fake_label.pt")
        if cfg.save_data.history:
            torch.save(history, path + "history.pt")

    if cfg.save_data.log:
        # Initialization Results Logger
        csv_file_path = "Raw_Data/Data.csv"
        field_names = [
            "Seed",
            "Model_Architecture",
            "Batch_Size",
            "Init_Method",
            "Pruning",
            "Activation",
            "Iterations",
            "Gradient_Loss",
            "PSNR",
        ]
        raw_data = {
            "Seed": seed,
            "Model_Architecture": cfg.model._target_.split(".")[-1],
            "Batch_Size": cfg.batch_size,
            "Init_Method": cfg.model.init_method,
            "Pruning": cfg.client.prune.type,
            "Activation": cfg.model.activation,
            "Iterations": len(history["distance"]),
            "Gradient_Loss": history["distance"][-1],
            "PSNR": RL.mean(),
        }
        write_to_csv(raw_data, csv_file_path, field_names)

        # Model Mode Results Logger
        csv_file_path = "Raw_Data/Mode.csv"
        field_names = [
            "Seed",
            "Model_Architecture",
            "Mode",
            "Pruning",
            "Gradient_Loss",
            "Zero_Percentage",
        ]
        raw_data = {
            "Seed": seed,
            "Model_Architecture": cfg.model._target_.split(".")[-1],
            "Mode": "model.train()",
            "Pruning": cfg.client.prune.type,
            "Gradient_Loss": history["distance"][-1],
            "Zero_Percentage": zero_gradient,
        }
        write_to_csv(raw_data, csv_file_path, field_names)

        # Dummy Label Results Logger
        csv_file_path = "Raw_Data/iDLG.csv"
        field_names = [
            "Seed",
            "Model_Architecture",
            "Label",
            "Pruning",
            "Gradient_Loss",
            "PSNR",
        ]
        raw_data = {
            "Seed": seed,
            "Model_Architecture": cfg.model._target_.split(".")[-1],
            "Label": "DLG",
            "Pruning": cfg.client.prune.type,
            "Gradient_Loss": min(history["distance"]),
            "PSNR": RL.mean(),
        }
        write_to_csv(raw_data, csv_file_path, field_names)


# endregion


if __name__ == "__main__":
    main()
