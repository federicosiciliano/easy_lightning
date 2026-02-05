import time
import pytorch_lightning as pl
import platform, subprocess, time
import numpy as np
import torch

class TimeCallback(pl.callbacks.Callback):
    def __init__(self, log_params={}):
        self.custom_log = lambda name, value: self.log(name, value, **log_params)

    def on_epoch_start(self):
        self.start_time = time.time()

    def on_epoch_end(self, split_name):
        self.elapsed_time = time.time() - self.start_time
        self.custom_log(split_name+"_time", self.elapsed_time)

    def on_train_epoch_start(self, trainer, pl_module):
        self.on_epoch_start()

    def on_train_epoch_end(self, trainer, pl_module):
        self.on_epoch_end("train")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.on_epoch_start()
        
    def on_validation_epoch_end(self, trainer, pl_module):
        self.on_epoch_end("val")

    def on_test_epoch_start(self, trainer, pl_module):
        self.on_epoch_start()

    def on_test_epoch_end(self, trainer, pl_module):
        self.on_epoch_end("test")

class TemperatureSlowdownCallback(pl.callbacks.Callback):
    def __init__(self, threshold=80, sleep_time=10, every_n_epochs=5, devices=slice(None), nvidia_smi_path=r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        self.epoch = 0
        self.threshold = threshold
        self.sleep_time = sleep_time
        self.every_n_epochs = every_n_epochs
        self.devices = devices
        if platform.system() == "Windows":
            self.command = f'"{nvidia_smi_path}" --query-gpu=temperature.gpu --format=csv,noheader,nounits'
        elif platform.system() == "Linux":
            self.command = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits"
        else:
            raise ValueError(f"Unsupported OS: {platform.system()}. Only Windows and Linux are supported.")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.epoch += 1
        if self.epoch % self.every_n_epochs == 0:
            try:
                max_temp = self.threshold + 1
                while max_temp > self.threshold:
                    output = subprocess.check_output(self.command, shell=True, text=True, stderr=subprocess.PIPE, timeout=15)
                    temps = np.array([int(t.strip()) for t in output.strip().split('\n') if t.strip().isdigit()])

                    max_temp = temps[self.devices].max()
                    if max_temp > self.threshold:
                        print(f"GPU temperature {max_temp}°C exceeds threshold {self.threshold}°C. Sleeping for {self.sleep_time} seconds.")
                        time.sleep(self.sleep_time)
            except Exception as e:
                print(f"Error checking GPU temperature: {e}")

class TerminateOnNaNCallback(pl.callbacks.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is not None and torch.isnan(loss):
            print(f"NaN loss detected at batch {batch_idx}. Stopping training.")
            trainer.should_stop = True