#%%

from deeplabv3plus_custom.models.DeepLabV3Plus_Dropout import DeepLabV3Plus
import torch

model = DeepLabV3Plus(in_channels=3, 
                      classes=6, 
                      encoder_name='tu-resnest50d',
                      encoder_weights=None)

model_weight = 'deeplabv3plus_custom/model_ckpts/heart_seg_dropout.ckpt' 
model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]

for k in list(model_weight.keys()):
    k_new = k.replace(
        "model.", "", 1
    )  # e.g. "model.conv.weight" => conv.weight"
    model_weight[k_new] = model_weight.pop(k)

model.load_state_dict(model_weight)

#%%

total_params = sum([p.numel() for p in model.parameters()])
print(f"Total number of parameters: {total_params}")


trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

print(f"Total number of trainable parameters: {trainable_params}")
print(f"Total number of non-trainable parameters: {non_trainable_params}")


#%%

import os
import tempfile
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger

from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import NLL, BNN_VI_ELBO_Regression
from lightning_uq_box.viz_utils import (
    plot_calibration_uq_toolbox,
    plot_predictions_regression,
    plot_toy_regression_data,
    plot_training_metrics,
)

plt.rcParams["figure.figsize"] = [14, 5]

seed_everything(0)  # seed everything for reproducibility


my_temp_dir = tempfile.mkdtemp()

dm = ToyHeteroscedasticDatamodule(batch_size=50)

X_train, y_train, train_loader, X_test, y_test, test_loader = (
    dm.X_train,
    dm.y_train,
    dm.train_dataloader(),
    dm.X_test,
    dm.y_test,
    dm.test_dataloader(),
)

#fig = plot_toy_regression_data(X_train, y_train, X_test, y_test)


network = MLP(n_inputs=1, n_hidden=[50, 50], n_outputs=2, activation_fn=nn.Tanh())

total_params = sum([p.numel() for p in network.parameters()])
print(f"Total number of parameters: {total_params}")


trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
non_trainable_params = sum(p.numel() for p in network.parameters() if not p.requires_grad)

print(f"Total number of trainable parameters: {trainable_params}")
print(f"Total number of non-trainable parameters: {non_trainable_params}")

#%%

network

#%%


bbp_model = BNN_VI_ELBO_Regression(
    network,
    optimizer=partial(torch.optim.Adam, lr=1e-2),
    criterion=NLL(),
    stochastic_module_names=[0,1,2],
    num_mc_samples_train=10,
    num_mc_samples_test=25,
    burnin_epochs=20,
)
network

#%%


logger = CSVLogger(my_temp_dir)
trainer = Trainer(
    max_epochs=100,  # number of epochs we want to train
    logger=logger,  # log training metrics for later evaluation
    log_every_n_steps=20,
    enable_checkpointing=False,
    enable_progress_bar=False,
    default_root_dir=my_temp_dir,
)

trainer.fit(bbp_model, dm)


#%%
fig = plot_training_metrics(
    os.path.join(my_temp_dir, "lightning_logs"), ["train_loss", "trainRMSE"]
)
preds = bbp_model.predict_step(X_test)

#%%

named_params = [name for name in bbp_model.named_parameters()]
param_names = [named_params[i][0] for i in range(len(named_params))]
print(param_names)

#%%

print('Parameters: ', bbp_model.get_parameter('model.model.3.mu_weight'))
print('Parameter shape: ', bbp_model.get_parameter('model.model.3.mu_weight').shape)

#%%

total_params = sum([p.numel() for p in bbp_model.parameters()])
print(f"Total number of parameters: {total_params}")


trainable_params = sum(p.numel() for p in bbp_model.parameters() if p.requires_grad)
non_trainable_params = sum(p.numel() for p in bbp_model.parameters() if not p.requires_grad)

print(f"Total number of trainable parameters: {trainable_params}")
print(f"Total number of non-trainable parameters: {non_trainable_params}")



#%%

fig = plot_predictions_regression(
    X_train,
    y_train,
    X_test,
    y_test,
    preds["pred"].squeeze(-1),
    preds["pred_uct"],
    epistemic=preds["epistemic_uct"],
    aleatoric=preds["aleatoric_uct"],
    title="Bayes By Backprop MFVI",
    show_bands=False,
)


#%%

fig = plot_calibration_uq_toolbox(
    preds["pred"].cpu().numpy(),
    preds["pred_uct"].cpu().numpy(),
    y_test.cpu().numpy(),
    X_test.cpu().numpy(),
)


#%%



#%%



#%%