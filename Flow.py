import argparse
import os
import pickle
import time
from copy import deepcopy
from tqdm import tqdm
from typing import Union, Tuple
from collections import namedtuple

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from Preprocessing import create_HDF_file
from Model import create_model
from Generator import DrugGeneration
from dataloader import HDFDataset
from util import write_molecules

from Parameters import Parameters

class Drug():

    def __init__(self):

        # dataset files
        self.test_h5_path = None
        self.train_h5_path = None
        self.valid_h5_path = None

        self.checkpoint_path = None

        self.train_smi_path = './data/train.smi'
        self.valid_smi_path = './data/validation.smi'
        self.test_smi_path = './data/test.smi'

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
        self.current_lr = 0
        
        # Helper Constants

        self.start_epoch = 0
        self.end_epoch = 1
        self.current_epoch = 0

        self.gradient_accumulation_steps = 1

        self.device = Parameters.device

        self.loss_logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.loss_kldiverence = torch.nn.KLDivLoss(reduction="batchmean")

        self.tb_writer = SummaryWriter()

        self.model_name = "test_debug"
        self.model_save_path = './model_saves'


    def write_train_log(self , training_loss , validation_loss ,lr , epoch):
        self.tb_writer.add_scalar("Training/training_loss", training_loss, epoch)
        self.tb_writer.add_scalar("Training/validation_loss", validation_loss, epoch)
        self.tb_writer.add_scalar("Training/lr", lr, epoch)
        
    def write_evaluation_log(self):
        pass
    
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

    def preprocess_data(self):

        # ===========Train==========
        if not os.path.exists(self.train_smi_path[:-3] + "h5"):
            print("Create Train HD5")
            create_HDF_file(path=self.train_smi_path)

        # =======Validation======
        if not os.path.exists(self.valid_smi_path[:-3] + "h5"):
            print("Create Validation HD5")
            create_HDF_file(path=self.valid_smi_path)

        # =======Test======
        if os.path.exists(self.test_smi_path[:-3] + "h5"):
            print("Create Test HD5")
            create_HDF_file(path=self.test_smi_path)

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
        
        self.load_data()
        for epoch in range(self.start_epoch, self.end_epoch):
            self.current_epoch = epoch
            print(f"Start Training Epoch number {epoch}")
            training_loss = self.run_one_train_epoch()
            validation_loss = self.run_one_validation_epoch()
            
            self.write_train_log(training_loss=training_loss ,validation_loss=validation_loss , lr=self.current_lr , epoch=epoch )
            
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
    parser = argparse.ArgumentParser(description='Drug Generation and Training Script')
    

    parser.add_argument('--preprocess', action='store_true', help='Flag to preprocess the data')
    parser.add_argument('--train', action='store_true', help='Flag to train the model')
    parser.add_argument('--generate', action='store_true', help='Flag to generate samples')
   
    args = parser.parse_args()

    drug = Drug()

    if args.preprocess:
        drug.preprocess_data()

    if args.train:
        drug.train_model()

    if args.generate:
        drug.generate(n_samples=args.n_samples)

    