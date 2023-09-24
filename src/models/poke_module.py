from typing import Any, Dict, Tuple

import torch
import torchvision.utils as vutils
import wandb
from lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class PokeGanModule(LightningModule):
    def __init__(
        self,
        gen: torch.nn.Module,
        disc: torch.nn.Module,
        criterion: torch.nn.Module = torch.nn.BCELoss(),
        lr=0.001,
        optimizer_gen: torch.optim.Optimizer = torch.optim.Adam,
        optimizer_disc: torch.optim.Optimizer = torch.optim.Adam,
        b1=0.5,
        b2=0.999,
        scheduler_gen: torch.optim.lr_scheduler = None,
        scheduler_disc: torch.optim.lr_scheduler = None,
        noise_dist="normal",
        compile: bool = False,
        disc_iters: int = 1,
        gradient_clip_val: float = None,
        gradient_penalty: bool = False,
        lambda_gp: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.gen = gen

        self.disc = disc

        self.criterion = criterion

        self.d_val_acc = Accuracy(task="binary")
        self.d_train_acc = Accuracy(task="binary")
        self.d_test_acc = Accuracy(task="binary")

        # for averaging loss across batches
        self.g_train_loss = MeanMetric()
        self.d_train_loss = MeanMetric()
        self.g_val_loss = MeanMetric()
        self.d_val_loss = MeanMetric()
        self.g_test_loss = MeanMetric()
        self.d_test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

        self.automatic_optimization = False

    def sample_noise(self, batch_size):
        if self.hparams.noise_dist == "uniform":
            noise_gen = torch.rand
        elif self.hparams.noise_dist == "normal":
            noise_gen = torch.randn
        return noise_gen(batch_size, self.gen.input_shape, 1, 1, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.g_train_loss.reset()
        self.d_train_loss.reset()
        self.g_val_loss.reset()
        self.d_val_loss.reset()
        self.g_test_loss.reset()
        self.d_test_loss.reset()
        self.test_noise_batch = self.sample_noise(16)

    def g_model_step(self, X_gen, real_label) -> float:
        batch_size = X_gen.shape[0]
        predictions = self.disc(X_gen)
        return self.criterion(predictions.view(batch_size, -1), real_label.view(batch_size, -1))

    def d_model_step(
        self, X, X_gen, real_label, fake_label
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = X_gen.shape[0]
        d_x = self.disc(X).view(batch_size, -1)
        real_label = real_label.view(batch_size, -1)
        fake_label = fake_label.view(batch_size, -1)
        loss_real = self.criterion(d_x, real_label)

        d_z = self.disc(X_gen.detach()).view(batch_size, -1)
        loss_fake = self.criterion(d_z, fake_label)
        loss_total = loss_real + loss_fake
        return (
            loss_total,
            torch.cat((d_x, d_z)),
            torch.cat((real_label, fake_label)),
        )

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: Nothing.
        """
        # update and log metrics
        g_opt, d_opt = self.optimizers()

        X, _ = batch

        batch_size = X.shape[0]

        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)
        for _ in range(self.hparams.disc_iters):
            X_gen = self.gen(self.sample_noise(batch_size))

            ##########################
            # Optimize Discriminator #
            ##########################

            d_loss, preds, targets = self.d_model_step(X, X_gen, real_label, fake_label)

            d_opt.zero_grad()
            self.manual_backward(d_loss, retain_graph=True)
            if self.hparams.gradient_clip_val is not None:
                self.clip_gradients(
                    d_opt,
                    gradient_clip_val=self.hparams.gradient_clip_val,
                    gradient_clip_algorithm="value",
                )
            d_opt.step()

        ######################
        # Optimize Generator #
        ######################

        g_loss = self.g_model_step(X_gen, real_label)
        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()

        # Log the variables in a single step using self.log_dict
        self.log_dict(
            {
                "train/g_loss": self.g_train_loss(g_loss),
                "train/d_loss": self.d_train_loss(d_loss),
                "train/d_acc": self.d_train_acc(preds, targets),
            },
            prog_bar=True,
        )
        if batch_idx % 50 == 0:
            with torch.no_grad():
                # self.test_noise_batch.to(self.device)
                generated_images = self.gen(self.test_noise_batch)
                self.logger.log_image(
                    key="samples",
                    images=[x.squeeze(0) for x in torch.split(generated_images, 1, dim=0)],
                )

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: Nothing.
        """
        X, _ = batch

        batch_size = X.shape[0]

        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)

        X_gen = self.gen(self.sample_noise(batch_size))

        # Evaluate Discriminator
        d_loss, preds, targets = self.d_model_step(X, X_gen, real_label, fake_label)

        # Evaluate Generator (if needed for metrics)
        g_loss = self.g_model_step(X_gen, real_label)

        # Log the variables in a single step using self.log_dict
        self.log_dict(
            {
                "val/g_loss": self.g_val_loss(g_loss),
                "val/d_loss": self.d_val_loss(d_loss),
                "val/d_acc": self.d_val_acc(preds, targets),
            },
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        d_acc = self.d_val_acc.compute()  # get current val acc
        self.val_acc_best(d_acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        self.log(
            "val/val_acc_best",
            self.val_acc_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: Nothing.
        """
        X, _ = batch

        batch_size = X.shape[0]

        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)

        X_gen = self.gen(self.sample_noise(batch_size))

        # Evaluate Discriminator
        d_loss, preds, targets = self.d_model_step(X, X_gen, real_label, fake_label)

        # Evaluate Generator (if needed for metrics)
        g_loss = self.g_model_step(X_gen, real_label)

        # Log the variables in a single step using self.log_dict
        self.log_dict(
            {
                "test/g_loss": self.g_test_loss(g_loss),
                "test/d_loss": self.d_test_loss(d_loss),
                "test/d_acc": self.d_test_acc(preds, targets),
            },
            prog_bar=True,
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.gen = torch.compile(self.gen)
            self.disc = torch.compile(self.disc)

    def configure_optimizers(self) -> Any:
        optim_gen_kwargs = {"params": self.gen.parameters(), "lr": self.hparams.lr}
        optim_disc_kwargs = {"params": self.disc.parameters(), "lr": self.hparams.lr}
        if self.hparams.b1 is not None and self.hparams.b2 is not None:
            optim_gen_kwargs["betas"] = (self.hparams.b1, self.hparams.b2)
            optim_disc_kwargs["betas"] = (self.hparams.b1, self.hparams.b2)
        optimizer_G = self.hparams.optimizer_gen(**optim_gen_kwargs)
        optimizer_D = self.hparams.optimizer_disc(**optim_disc_kwargs)

        return optimizer_G, optimizer_D


class PokeWGanModule(PokeGanModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def d_model_step(
        self, X, X_gen, real_label, fake_label
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_x = self.disc(X).reshape(X.shape[0], -1)
        d_z = self.disc(X_gen).reshape(X.shape[0], -1)
        return (
            torch.mean(d_z) - torch.mean(d_x),
            torch.cat((d_x, d_z)),
            torch.cat((real_label, fake_label)),
        )

    def g_model_step(self, X_gen, real_label) -> float:
        d_z = self.disc(X_gen).reshape(X_gen.shape[0], -1)
        return torch.mean(d_z)

    def gradient_penalty(self, real, fake):
        pass
        # TODO


if __name__ == "__main__":
    _ = PokeGanModule(None, None, None, None, None, None)
