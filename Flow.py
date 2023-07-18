from collections import namedtuple
import pickle
from copy import deepcopy
import time
import os
from typing import Union, Tuple
import torch
from tqdm import tqdm
from DataLoaders import HDFDataset

from Model import create_model
from Preprocessing import create_HDF_file
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from Generator import DrugGeneration
import os


from util import write_molecules


class Drug():

    def __init__(self):

        # dataset files
        self.test_h5_path = None
        self.train_h5_path = None
        self.valid_h5_path = None

        self.checkpoint_path = None

        self.test_smi_path = './test.smi'
        self.train_smi_path = './train.smi'
        self.valid_smi_path = 'validation.smi'

        # general paramters
        self.optimizer = None
        self.scheduler = None
        self.analyzer = None
        self.current_epoch = None
        self.restart_epoch = None

        # hyper parameters
        self.batch_size = 7
        self.dataloader_workers_count = 2

        self.learning_rate = 3e-4

        # learning parameters
        self.model = None
        self.test_dataloader = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.likelihood_per_action = None

        self.lr_scheduler = None
        # Helper Constants

        self.start_epoch = 0
        self.end_epoch = 1
        self.current_epoch = 0

        self.gradient_accumulation_steps = 1

        self.device = torch.device("cpu")

        self.loss_logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.loss_kldiverence = torch.nn.KLDivLoss(reduction="batchmean")

        self.tb_writer = SummaryWriter()

        self.model_name = "test_debug"
        self.model_save_path = './'

    def save(self, save_model=True, save_optimizer=False):
        if save_model:
            torch.save(self.model.state_dict(), os.path.join(
                self.model_save_path, self.model_name+".pt"))
            print("Model Weights Saved")
        if save_model:
            torch.save(self.optimizer.state_dict(), os.path.join(
                self.model_save_path, self.model_name+"_optimizer.pt"))
            print("Optimizer Weights Saved")

    def load(self, load_optimizer=True):

        self.model.load_state_dict(torch.load(os.path.join(
            self.model_save_path, self.model_name+".pt")))

        print("Model Weights Loaded")

        if load_optimizer:
            self.optimizer.load_state_dict(torch.load(os.path.join(
                self.model_save_path, self.model_name+"_optimizer.pt")))

            print("Optimizer Weights Loaded")

    def log(self, key, value):
        self.tb_writer.add_scalar(key, value)

    def load_data(self):

        # ===========Train==========
        if os.path.exists(self.train_smi_path[:-3] + "h5"):
            print("Train H5 Exist Just Load it")
            self.train_h5_path = self.train_smi_path[:-3] + "h5"
        else:
            print("Create Training set h5 File")
            self.train_h5_path = create_HDF_file(path=self.train_smi_path)

        print("Create Train DataLoader")
        self.train_dataloader = self.create_dataloader(
            self.train_h5_path, is_train=True)

        # =======Validation======
        if os.path.exists(self.valid_smi_path[:-3] + "h5"):
            print("Validation H5 Exist Just Load it")
            self.valid_h5_path = self.valid_smi_path[:-3] + "h5"
        else:
            print("Create Validation set h5 File")
            self.valid_h5_path = create_HDF_file(path=self.valid_smi_path)

        print("Create Validation DataLoader")
        self.valid_dataloader = self.create_dataloader(
            self.valid_h5_path, is_train=False)

        # =======Test Set======
        if os.path.exists(self.test_smi_path[:-3] + "h5"):
            print("Test H5 Exist Just Load it")
            self.test_h5_path = self.test_smi_path[:-3] + "h5"
        else:
            print("Create Test set h5 File")
            self.test_h5_path = create_HDF_file(path=self.test_smi_path)

        print("Create Test DataLoader")

        self.test_h5_path = self.create_dataloader(
            self.test_h5_path, is_train=False)

    def create_dataloader(self, h5_path, is_train=True):
        dataset = HDFDataset(h5_path)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=is_train,)
        
        return dataloader

    def create_model_and_optimizer(self):

        self.model = create_model()
        if self.checkpoint_path != None:
            
            print("Loading Model Checkpoint Weights")
            
            self.model.load_state_dict(torch.load(self.checkpoint_path))

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    def train_model(self):

        for epoch in range(self.start_epoch, self.end_epoch):
            self.current_epoch = epoch
            print(f"Start Training Epoch number {epoch}")
            training_loss = self.run_one_train_epoch()

            print(training_loss.item())

    def calculate_loss(self, model_output, target_output):

        model_output = self.loss_logsoftmax(model_output)

        loss = self.loss_kldiverence(target=target_output, input=model_output)

        return loss

    def run_one_validation_epoch(self):
        validation_loss_tensor = torch.zeros(
            len(self.valid_dataloader), device=self.device)

        self.model.eval()

        with torch.no_grad():
            for batch_index, batch in tqdm(enumerate(self.valid_dataloader), total=len(self.valid_dataloader)):

                batch = [b.to(self.device) for b in batch]

                nodes, edges, target_output = batch

                model_output = self.model(nodes, edges)
                step_loss = self.calculate_loss(
                    model_output=model_output, target_output=target_output)

                validation_loss_tensor[batch_index] = step_loss

        validation_epoch_loss = torch.mean(validation_loss_tensor)
        self.log("ValidationLoss", validation_epoch_loss)
        return validation_epoch_loss

    def run_one_train_epoch(self):

        current_step = 0

        self.model.zero_grad()
        self.optimizer.zero_grad()

        train_loss_tensor = torch.zeros(
            len(self.train_dataloader), device=self.device)

        pbar = tqdm(total=len(self.train_dataloader),
                    desc=f"Training Epoch {self.current_epoch}")

        self.model.train()

        for batch_index, batch in enumerate(self.train_dataloader):

            # convert All Tensors To Device Type Tensord
            batch = [b.to(self.device) for b in batch]

            nodes, edges, target_output = batch

            model_output = self.model(nodes, edges)

            step_loss = self.calculate_loss(
                model_output=model_output, target_output=target_output)

            train_loss_tensor[batch_index] = step_loss

            pbar.set_postfix_str(f"Loss: {step_loss:.4f}")

            step_loss.backward()

            if current_step % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.lr_scheduler != None:
                self.lr_scheduler.step()

            current_step += 1

            pbar.update(1)

        epoch_loss = torch.mean(train_loss_tensor)
        self.log("TrainLoss", epoch_loss)
        return epoch_loss

    def generate(self, n_samples):
        # Create model And Load Last Time Saved Weights To Model
        self.create_model_and_optimizer()
        self.load()

        generation_batch_size = min(self.batch_size, n_samples)

        n_generation_batches = int(n_samples/generation_batch_size)

        generator = DrugGeneration(model=self.model,
                                   batch_size=6)
        # generate graphs in batches

        generated_graphs = []
        generated_action_likehoods = []
        generated_final_loglikelihood = []
        generated_termination = []

        for idx in range(0, n_generation_batches + 1):
            # generate one batch of graphs
            (graphs, action_likelihoods, final_loglikelihoods,
             termination) = generator.sample()

            generated_graphs.extend(graphs)
            generated_action_likehoods.extend(action_likelihoods)
            generated_final_loglikelihood.extend(final_loglikelihoods)
            generated_termination.extend(termination)

        fraction_valid, validity, uniqueness = write_molecules(
            molecules=generated_graphs[:n_samples],
            name="Ahora",

        )


if __name__ == "__main__":
    drug = Drug()

    # drug.load_data()
    # drug.create_model_and_optimizer()
    # drug.load()
    # drug.train_model()

    drug.generate(n_samples=10)

    # drug.save()

    print("Done")
