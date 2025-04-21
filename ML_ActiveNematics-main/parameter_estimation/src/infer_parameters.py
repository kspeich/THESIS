import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import glob
import argparse
import numpy as np      # Sometimes DataLoaders are really slow if you import numpy first?
import pandas as pd

from pathlib import Path

from dataset import *
from parameter_estimator_convnext import ConvNextParameterEstimator

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# This script can be run on command line by: python estimate_parameters.py --model_name='MODELNAME' --dataset='DATASET'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)           # This is be the folder path to the trained model (must train the model first)
    parser.add_argument('--directory', type=str)            # This is the file path to the input directory with all the datasets
    parser.add_argument('--output_directory', type=str, default='../../../data/inferences')     # This is where the code will output a CSV file

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()

    # This is the list of all of the HDF5 files in the input directory
    paths = glob.glob(f'{args.directory}/*.hdf5')

    # GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Training on {device}')
    
    # Create the model
    model = ConvNextParameterEstimator()    # Here we are using the default values

    # Load the weights from the trained model
    logger.info(f'Loading weights for {args.model_name}')
    info = torch.load(f'{args.model_name}/model_weight.ckpt', map_location=device, weights_only=True)
    model.load_state_dict(info['state_dict'])           # Put the weights into the model object
    model.to(device)

    # Set up the output directory (we want to categorize it by the training model just so we know what gave what numbers)
    output_dir = Path(args.output_directory) / Path(args.model_name).name
    output_dir.mkdir(parents=True, exist_ok=True)         # Create output_dir if it doesn't exist already

    # Then we loop over every HDF5 file in our input directory
    for path in paths:
        logger.info(f'Processing data file: {path}')

        # Now we want to read the data
        inferences = []

        # We include the non-random transformations that we used in training in train_parameter_estimator.py
        transform = transforms.Compose([
            Sin2t(),
            ToTensor(),
        ])

        # Pass the data into the NematicsSequenceDataset object (defined in dataset.py)
        dataset = NematicsSequenceDataset(path=path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)     # We create dataloader to process in batches below

        logger.info('Starting to infer parameters')
        model.eval()        # Make sure that the model is in evaluation mode

        # Now we iterate over the batches in dataloader
        with torch.no_grad():
            counter = 0
            for batch in dataloader:
                counter += 1
                logger.info(f'Processing batch {counter} out of {len(dataloader)}')
                inputs = batch[0].to(device)
                logger.debug(f'Input shape: {inputs.shape}, dtype: {inputs.dtype}')

                outputs = model(inputs)
                inferences.append(outputs.cpu().numpy())

        inferences = np.concatenate(inferences, axis=0)

        # Write the parameter estimates as a DataFrame object, then save as a CSV 
        df_infer = pd.DataFrame(inferences, columns=['k', 'z'])
        output_file = f'{output_dir / Path(path).stem}.csv'
        logger.info(f'Saving inferences to {output_file}')
        df_infer.to_csv(output_file)