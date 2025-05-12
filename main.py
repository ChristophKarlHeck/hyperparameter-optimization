import csv
import os
import glob
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import OneHotEncoder  # Required for df_to_tensor()
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


# Define CSV log file
LOG_FILE = "test.csv"

# Write headers if the file does not exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "trial",
            "output_channels_1",
            "output_channels_2",
            "output_channels_3",
            "kernel_size_1",
            "kernel_size_2",
            "kernel_size_3",
            "padding_1",
            "padding_2",
            "padding_3",
            "lr",
            "batch_size",
            "weight_decay",
            "test_accuracy",
            "path"])

def downsample_last_300(df):
    if df.empty:
        print("Warning: DataFrame is empty!")
        return df

    if df.shape[1] < 300:
        print("Error: DataFrame has fewer than 600 columns")
        return df

    last_600_columns = df.iloc[:, -300:].to_numpy()

    if last_600_columns.shape[1] % 3 != 0:
        raise ValueError("Error: Number of columns to downsample is not a multiple of 6.")

    downsampled_values = last_600_columns.reshape(df.shape[0], 100, 6).mean(axis=2)
    new_columns = [f"downsampled_{i}" for i in range(100)]
    downsampled_df = pd.DataFrame(downsampled_values, columns=new_columns, index=df.index)

    df = pd.concat([df.iloc[:, :-300], downsampled_df], axis=1)
    return df

def load_and_downsample_data(path_pattern="/home/chris/experiment_data/*/rolling_window/training_data_zscore_1_10_local"):
    """Loads CSV files from the experiment directories and downsamples to 100 values."""
    training_dirs = glob.glob(path_pattern)
    df_list = []
    
    for training_dir in training_dirs:
        if os.path.isdir(training_dir):  # Check if it exists
            csv_files = glob.glob(os.path.join(training_dir, "*.csv"))  # Get all CSV files
            
            for file in csv_files:
                df = pd.read_csv(file)  # Read CSV file
                df_list.append(df)  # Add to list

    # Concatenate all data into a single DataFrame
    final_df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    
    # Downsample the last 600 values to 100
    #final_df = downsample_last_300(final_df)  
    print(final_df.describe())
    #final_df.to_csv("/home/chris/watchplant_classification_dl/pipline_test/traing.csv")
    
    return final_df

def get_representative_sample(df, sample_size=50000, target_col='Heat'):
    # Ensure that the dataset has at least sample_size entries
    if len(df) < sample_size:
        raise ValueError("The dataset has fewer entries than the requested sample size.")
    
    # Set up the stratified shuffle split for one split with the desired sample size
    sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
    
    # Get the indices for the sample that preserves class proportions
    for train_index, _ in sss.split(df, df[target_col]):
        sample_df = df.iloc[train_index]
    
    return sample_df

def split_data(df):
    print("🔍 Checking class balance before splitting...")
    print(df["Heat"].value_counts(normalize=True))  # ✅ Print class distribution

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    analysis_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Heat'])
    train_df, val_df = train_test_split(analysis_df, test_size=0.2, random_state=42)

    return train_df, val_df, test_df


def df_to_tensor(df, y_column):
    if df.empty:
        raise ValueError(f"Error: DataFrame is empty when converting to tensor for {y_column}.")

    x = torch.tensor(df.iloc[:, -100:].values, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    
    y = df[[y_column]].values
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)

    y_tensor = torch.tensor(y_encoded, dtype=torch.float32)

    # Check for NaN or infinite values in input tensors
    if torch.isnan(x).any() or torch.isnan(y_tensor).any():
        print("❗ Warning: NaN detected in input tensors!")

    if torch.isinf(x).any() or torch.isinf(y_tensor).any():
        print("❗ Warning: Infinite values detected in input tensors!")

    return x, y_tensor


def prepare_dataloaders(df, batch_size):
    if df.empty:
        raise ValueError("Error: DataFrame is empty after loading and downsampling!")

    df = get_representative_sample(df)
    train_df, val_df, test_df = split_data(df)

    x_train, y_train = df_to_tensor(train_df, "Heat")
    x_val, y_val = df_to_tensor(val_df, "Heat")
    x_test, y_test = df_to_tensor(test_df, "Heat")

    if x_train.shape[0] == 0 or x_val.shape[0] == 0 or x_test.shape[0] == 0:
        raise ValueError("Error: One of the datasets has zero samples.")

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=7)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=7)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=7)

    return train_loader, val_loader, test_loader


# Define the model class
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


def memory_kbytes(C1, C2, C3, k1, k2, k3):
    prog  = 6792 + 8*(C1 + C2 + C3)
    convs = 4*(C1*k1 + 3*C1
             + C2*C1*k2 + 3*C2
             + C3*C2*k3 + 5*C3 + 2)
    pool  = 2**14
    buf   = (max(C1, C2, C3)*100)*2
    return (prog + convs + pool + buf) / 1000

def objective(trial):
    """Optuna objective function for hyperparameter tuning with parameter restrictions."""
    
    # Hyperparameter search space
    output_channels_1 = trial.suggest_categorical("output_channels_1", [4, 8, 16, 32, 64])
    output_channels_2 = trial.suggest_categorical("output_channels_2", [4, 8, 16, 32, 64])
    output_channels_3 = trial.suggest_categorical("output_channels_3", [4, 8, 16, 32, 64])

    kernel_size_1 = trial.suggest_categorical("kernel_size_1", [3, 5, 7, 9])
    kernel_size_2 = trial.suggest_categorical("kernel_size_2", [3, 5, 7, 9])
    kernel_size_3 = trial.suggest_categorical("kernel_size_3", [3, 5, 7, 9])

    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)  
    batch_size = trial.suggest_categorical("batch_size", [8,16,32,64])  
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True) 

    padding_1 = kernel_size_1 // 2
    padding_2 = kernel_size_2 // 2
    padding_3 = kernel_size_3 // 2

    mem_kb = memory_kbytes(
        output_channels_1,
        output_channels_2,
        output_channels_3,
        kernel_size_1,
        kernel_size_2,
        kernel_size_3)
    
    # Optionally discard configurations that exceed the 130 kB limit
    if mem_kb >= 130:
        raise optuna.TrialPruned()

    # Load dataset
    df = load_and_downsample_data()
    train_loader, val_loader, test_loader = prepare_dataloaders(df, batch_size)

    # Initialize model
    model = FCN(
        input_channels=1,
        output_channels_1 = output_channels_1,
        output_channels_2 = output_channels_2,
        output_channels_3 = output_channels_3,
        kernel_size_1 = kernel_size_1,
        kernel_size_2 = kernel_size_2,
        kernel_size_3 = kernel_size_3,
        padding_1 = padding_1,
        padding_2 = padding_2,
        padding_3 = padding_3,
        lr=lr
    )

    # Modify optimizer to include weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Callbacks: Checkpoint & Early Stopping
    checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", mode="max", save_top_k=1)
    early_stopping_callback = EarlyStopping(monitor="val_accuracy", patience=100, mode="max", verbose=True)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=1,  
        callbacks=[
            checkpoint_callback,
            early_stopping_callback],
        enable_progress_bar=True
    )

    trainer.fit(model, train_loader, val_loader)

    val_accuracy = trainer.callback_metrics.get("val_accuracy")
    if val_accuracy is not None:
        val_acc_value = val_accuracy.item() if isinstance(val_accuracy, torch.Tensor) else float(val_accuracy)
        trial.report(val_acc_value, step=trainer.current_epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()


    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        model = FCN.load_from_checkpoint(
            best_model_path, 
            input_channels=1,
            output_channels_1 = output_channels_1,
            output_channels_2 = output_channels_2,
            output_channels_3 = output_channels_3,
            kernel_size_1 = kernel_size_1,
            kernel_size_2 = kernel_size_2,
            kernel_size_3 = kernel_size_3,
            padding_1 = padding_1,
            padding_2 = padding_2,
            padding_3 = padding_3,
            lr=lr
        )

    # Evaluate on test set
    test_results = trainer.test(model, test_loader)
    test_accuracy = test_results[0]["test_accuracy"]

    # Log results
    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            trial.number,
            output_channels_1,
            output_channels_2,
            output_channels_3,
            kernel_size_1,
            kernel_size_2,
            kernel_size_3,
            padding_1,
            padding_2,
            padding_3,
            lr,
            batch_size,
            weight_decay,
            test_accuracy,
            best_model_path
        ])


    return test_accuracy



def stop_when_good_enough(study, trial):
    if study.best_value is not None and study.best_value > 0.98:
        print("Target accuracy achieved. Stopping hyperparameter search.")
        study.stop()


def main():

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),  # Start with random exploration
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)  # Prune bad trials early
    )
    study.optimize(objective, n_trials=1, callbacks=[stop_when_good_enough])

    print("Best hyperparameters:", study.best_params)
    print("Best test accuracy:", study.best_value)

if __name__ == "__main__":
    main()
