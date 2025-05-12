import torch
from torchview import draw_graph
from torch import nn
import torch.nn.functional as F
from torch.export import export, export_for_training, ExportedProgram
from executorch.exir import ExecutorchBackendConfig, ExecutorchProgramManager
import executorch.exir as exir
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import glob
import os
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from helper import generate_memory_trace

class FCN(pl.LightningModule):
    def __init__(
            self,
            input_channels,
            output_channels_1,
            output_channels_2,
            output_channels_3,
            kernel_size_1,
            kernel_size_2,
            kernel_size_3,
            padding_1,
            padding_2,
            padding_3,
            lr):
        super().__init__()
        self.model_name = "FCN"

        self.save_hyperparameters()

        # If we don’t pad: the signal shrinks every layer.
        #If we pad: we maintain the temporal resolution, which seems to be the intention here — especially since it ends in Global Pooling.
        self.conv1d_1 = nn.Conv1d(input_channels, output_channels_1, kernel_size=kernel_size_1, padding=padding_1)
        self.bn1 = nn.BatchNorm1d(output_channels_1)
                
        self.conv1d_2 = nn.Conv1d(output_channels_1, output_channels_2, kernel_size=kernel_size_2, padding=padding_2)
        self.bn2 = nn.BatchNorm1d(output_channels_2)

        self.conv1d_3 = nn.Conv1d(output_channels_2, output_channels_3, kernel_size=kernel_size_3, padding=padding_3)
        self.bn3 = nn.BatchNorm1d(output_channels_3)


        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(output_channels_3, 2)
        
        self.loss_fn = nn.BCELoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=2)
        self.lr = lr


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1d_1(x)))
        x = F.relu(self.bn2(self.conv1d_2(x)))
        x = F.relu(self.bn3(self.conv1d_3(x)))
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        x = F.softmax(x,dim=1)
        return x

    def _common_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = self.loss_fn(output, y.float()) # Example loss
        y_true = torch.argmax(y, dim=1)        # class index
        y_pred = torch.argmax(output, dim=1)   # predicted class
        return loss, output, y_true, y_pred

    def training_step(self, batch, batch_idx):
        loss, scores, y, y_pred = self._common_step(batch, batch_idx)
        # print(f"scores: {scores}")
        accuracy = self.accuracy(y_pred, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "train_accuracy": accuracy, 
                "train_scores": scores, "train_y": y}

    def validation_step(self, batch, batch_idx):
        loss, scores, y, y_pred = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, y)
        self.log_dict(
            {
                "val_loss": loss,
                "val_accuracy": accuracy
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return {"val_loss": loss, "val_accuracy": accuracy}

    def test_step(self, batch, batch_idx):
        loss, scores, y, y_pred = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(y_pred, y)
        f1_score = self.f1_score(y_pred, y)
        self.log_dict(
            {
                "test_loss": loss,
                "test_accuracy": accuracy,
                "test_f1_score": f1_score,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )        
        return {"test_loss": loss, "test_accuracy": accuracy, "test_f1_score": f1_score}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    can_delegate = False

    @property
    @torch.jit.ignore  # This tells TorchScript to ignore this property
    def trainer(self):
        # Return None or a dummy trainer instance; returning None is usually fine for inference
        return None
    
    can_delegate = False

model = FCN.load_from_checkpoint(
    "/home/chris/watchplant_classification_dl/pipline_test/FCN_temp_10min_input/lightning_logs/version_966/checkpoints/epoch=312-step=626000.ckpt",
    output_channels_1 = 4,
    output_channels_2 = 32,
    output_channels_3 = 16,
    kernel_size_1 = 9,
    kernel_size_2 = 7,
    kernel_size_3 = 9,
    padding_1 = 4,
    padding_2 = 3,
    padding_3 = 4,
    lr=0.00036632491019339495)
    
model = model.cpu()
model.eval()

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model.pt') 

# Example Input
final_example_input = (torch.randn(1, 1, 100),)  # [batch_size, input_channels, seq_length]

# Define fixed input tensor should be classified as HEAT=1
fixed_input = torch.tensor([[[ 0.4289, 0.4291, 0.4288, 0.4287, 0.4288, 0.4288, 0.4286, 0.4286, 0.4286,
         0.4286, 0.4288, 0.4288, 0.4288, 0.4287, 0.4284, 0.4282, 0.4280, 0.4279,
         0.4278, 0.4278, 0.4279, 0.4279, 0.4281, 0.4282, 0.4283, 0.4284, 0.4284,
         0.4283, 0.4281, 0.4279, 0.4279, 0.4278, 0.4277, 0.4280, 0.4281, 0.4281,
         0.4281, 0.4281, 0.4284, 0.4284, 0.4285, 0.4286, 0.4286, 0.4287, 0.4288,
         0.4287, 0.4287, 0.4285, 0.4283, 0.4281, 0.4280, 0.4280, 0.4280, 0.4280,
         0.4280, 0.4279, 0.4281, 0.4282, 0.4283, 0.4284, 0.4286, 0.4288, 0.4292,
         0.4295, 0.4295, 0.4296, 0.4296, 0.4295, 0.4293, 0.4291, 0.4290, 0.4290,
         0.4291, 0.4295, 0.4295, 0.4295, 0.4294, 0.4293, 0.4290, 0.4288, 0.4287,
         0.4287, 0.4286, 0.4286, 0.4285, 0.4286, 0.4286, 0.4286, 0.4287, 0.4288,
         0.4289, 0.4291, 0.4293, 0.4294, 0.4295, 0.4295, 0.4294, 0.4294, 0.4294,
         0.4293]]],)

# [[9.9968e-01, 3.1811e-04]]
#output = model(fixed_input)  
# print("Model Input:", fixed_input)
#print("Model Output:", output)

numpy_input = fixed_input[0].detach().cpu().numpy()

# Flatten to a 1D list
flattened_input = numpy_input.flatten().tolist()

# Print the vector representation
print("C++ std::vector<float> representation:")
print(flattened_input)

# Export the Model
pre_autograd_aten_dialect = export_for_training(
        model,
        final_example_input
    ).module()


aten_dialect: ExportedProgram = export(pre_autograd_aten_dialect, final_example_input)

edge_program: exir.EdgeProgramManager = exir.to_edge(aten_dialect)


executorch_program: exir.ExecutorchProgramManager = edge_program.to_executorch(
    ExecutorchBackendConfig(
        passes=[],  # User-defined passes
    )
)

generate_memory_trace(executorch_program_manager=executorch_program, chrome_trace_filename="test.json", enable_memory_offsets=True)

print(executorch_program.exported_program())

print(executorch_program.executorch_program.execution_plan[0].non_const_buffer_sizes)

with open("model.pte", "wb") as file:
    file.write(executorch_program.buffer)

print("CNN model saved as model.pte")