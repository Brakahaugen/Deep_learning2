import pathlib
import matplotlib.pyplot as plt
import torch
import utils
import time
import typing
import collections
from torch import nn
from dataloaders import load_cifar10 
from task2 import *


class Deeper_model(nn.Module):
 
    def __init__(self,
                image_channels,
                    num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = [64, 128, 256, 512, 512, 1024]  # Set number of filters in conv layers
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters[0],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters[0],
                out_channels=num_filters[1],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2,
            ),
            
            nn.Conv2d(
                in_channels=num_filters[1],
                out_channels=num_filters[2],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters[2],
                out_channels=num_filters[3],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2,
            ),

            nn.Conv2d(
                in_channels=num_filters[3],
                out_channels=num_filters[4],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_filters[4]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters[4],
                out_channels=num_filters[5],
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2,
            ),
            nn.Flatten(),
        )
    
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = num_filters[5] * 4 * 4
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features,  2048),
            nn.ReLU(),
            nn.Linear(2048,  512),
            nn.ReLU(),
            nn.Linear(512,  128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            #    nn.Softmax(),
        )

    
    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
    
        batch_size = x.shape[0]
    
        out = self.feature_extractor.forward(x)
        
        out = self.classifier(out)
        
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out



if __name__ == "__main__":

    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = Deeper_model(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )

    #Overriding optimizer
    # trainer.optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    

    trainer.train()
    
    create_plots(trainer, "task3d")
    calculate_final_loss(trainer)
    