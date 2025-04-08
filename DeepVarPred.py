import math
import os
import random
import cv2
import numpy as np
from Bio import SeqIO
from sklearn import preprocessing
from torch.utils.data.sampler import Sampler
from typing import Callable
import pandas as pd
import torch
from torch import nn
import torch.utils.data
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score, f1_score
import onnxruntime as ort
import argparse

# Define a dictionary to map nucleotide bases to integers
# This encoding will be used to convert DNA sequences into numerical representations
encoders = {
    "A": 1,  # Adenine
    "T": 2,  # Thymine
    "C": 3,  # Cytosine
    "G": 4  # Guanine
}


def get_image_size(bed_path: str):
    """
    get the image size based on the start and end positions in the bed file.

    Parameters:
    bed_path (str): path to the bed file containing genomic coordinate information.

    Returns:
    tuple: IMAGE_HEIGHT (int), IMAGE_WIDTH (int), START_POSITION (int)
    """
    f = open(bed_path, 'r')
    read = f.readline().strip().split('\t')
    f.close()
    start = int(read[1])
    end = int(read[2])
    size = round(math.sqrt(end - start))
    IMAGE_HEIGHT = size + 1
    IMAGE_WIDTH = size + 1
    START_POSITION = start + 1
    return IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION


def prepare_mutation(mutation_path):
    """[summary]
    prepare mutation data by scaling scores and combining with info columns

    Args:
        mutation_path (str): path to the mutation data file

    Returns:
        pd.DataFrame: transformed mutation data with scaled scores and info columns
    """
    # Read the mutation data from a tab-separated file into a DataFrame
    score_df = pd.read_csv(mutation_path, sep='\t')

    # Define the columns that contain information about the mutation
    info_index = ['#chr', 'pos(1-based)', 'ref', 'alt', 'hg19_chr', 'hg19_pos(1-based)',
                  'hg18_chr', 'hg18_pos(1-based)', 'clinvar_clnsig']

    # Extract the information columns into a separate DataFrame
    score_df_info = score_df[info_index]

    # Drop the information columns from the original DataFrame to get only the score columns
    score_df_trans = score_df.drop(columns=info_index)

    # Get the column names of the score DataFrame
    column_names = score_df_trans.columns

    # Create a Min-Max Scaler to scale the scores to a range between 0 and 255
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 255), copy=True)

    # Apply the scaler to the score DataFrame and convert the scaled data to integers
    df_minmax = min_max_scaler.fit_transform(score_df_trans.values)
    df_minmax = df_minmax.astype(int)

    # Convert the scaled and integer-converted data back into a DataFrame
    df_minmax_trans = pd.DataFrame(df_minmax, columns=column_names)

    # Concatenate the scaled scores with the information columns
    df_trans_res = pd.concat([score_df_info, df_minmax_trans], axis=1)

    # Return the combined DataFrame
    return df_trans_res


def split_list(split, s):
    """splite a list to sub list contain s"""
    return [split[i:i + s] for i in range(len(split)) if i % s == 0]


def read_gene_ref(path):
    """
    Read a gene reference sequence from a FASTA file and return it as a list of nucleotides.

    This function uses Biopython's SeqIO to parse a FASTA file and extract the DNA sequence.
    The sequence is converted to uppercase and returned as a list of individual nucleotides.

    Parameters:
    path (str): Path to the FASTA file containing the gene reference sequence
    Returns:
    list: A list of uppercase nucleotide characters representing the gene sequence
    """
    # Parse the FASTA file using SeqIO.parse which returns a generator of SeqRecord objects
    # We take the first record (typical for FASTA files containing a single sequence)
    for record in SeqIO.parse(path, "fasta"):
        # Convert the sequence to uppercase string and split into individual characters
        seq = list(str(record.seq).upper())
    return seq


def create_singleton_image_score(each, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION):
    """Create a single-channel score image from mutation data.

    This function converts mutation score data into a 2D image representation where:
    - Each pixel represents a score value from the mutation data
    - The image dimensions are defined by IMAGE_HEIGHT and IMAGE_WIDTH
    - Scores are arranged row-wise with zero-padding if needed

    Args:
        each: A pandas Series or similar object containing mutation scores (first 9 elements are info columns)
        IMAGE_HEIGHT: Height of the output image (number of rows)
        IMAGE_WIDTH: Width of the output image (number of columns)
        START_POSITION: Starting genomic position (unused in this function but kept for interface consistency)

    Returns:
        numpy.ndarray: A 2D array of integers representing the score image
    """
    # Initialize empty image with specified dimensions
    score_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH)).astype(int)

    # Extract score values (skip first 9 info columns)
    each_ = each.to_list()[9:]

    # Handle case where scores don't fill a full image row
    if len(each_) < IMAGE_WIDTH:
        # Pad with zeros to match image width
        each_.extend([0] * (IMAGE_WIDTH - len(each_)))
        # Place scores in first row
        score_image[0] = each_
    else:
        # Split scores into chunks matching image width
        score_splite = split_list(each_, IMAGE_WIDTH)

        # Process each row of the image
        for line in range(len(score_splite)):
            tem = score_splite[line]

            # Pad partial rows with zeros
            if len(tem) != IMAGE_WIDTH:
                tem.extend([0] * (IMAGE_WIDTH - len(tem)))
            else:
                # Assign complete rows directly to image
                score_image[line] = tem

    return score_image.astype(int)


def create_singleton_image_ref(ref_path, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION):
    """
    Create a single-channel reference image from genomic sequence data.

    This function converts a DNA sequence into a 2D image representation where:
    - Each pixel represents an encoded nucleotide value (A=1, T=2, C=3, G=4)
    - The image dimensions are defined by IMAGE_HEIGHT and IMAGE_WIDTH
    - Sequence is arranged row-wise with zero-padding if needed

    Args:
        ref_path (str): Path to the FASTA file containing the reference sequence
        IMAGE_HEIGHT (int): Height of the output image (number of rows)
        IMAGE_WIDTH (int): Width of the output image (number of columns)
        START_POSITION (int): Starting genomic position (unused in this function but kept for interface consistency)

    Returns:
        numpy.ndarray: A 2D array of integers representing the reference image
    """
    # Initialize empty image with specified dimensions
    variant_image_ref = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH)).astype(int)

    # Read the reference sequence from FASTA file
    seq = read_gene_ref(ref_path)

    # Split the sequence into chunks matching image width
    seq_splite = split_list(seq, IMAGE_WIDTH)

    # Process each row of the image
    for line in range(len(seq_splite)):
        # Encode nucleotides to numerical values (A=1, T=2, C=3, G=4)
        tem = [encoders[x] for x in seq_splite[line]]

        # Handle case where sequence doesn't fill a full image row
        if len(tem) != IMAGE_WIDTH:
            # Pad with zeros to match image width
            tem.extend([0] * (IMAGE_WIDTH - len(tem)))
            variant_image_ref[line] = tem
        else:
            # Assign complete encoded sequence directly to image row
            variant_image_ref[line] = [encoders[x] for x in seq_splite[line]]

    return variant_image_ref.astype(int)


def create_singleton_image_alt(row, ref_path, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION):
    """
    Create a single-channel alternate allele image by modifying the reference image at the mutation position.

    This function:
    1. Creates a base reference image using the reference sequence
    2. Identifies the mutation position in the image
    3. Replaces the reference nucleotide with the alternate allele at that position

    Args:
        row: A pandas Series containing mutation information (must include 'hg19_pos(1-based)' and 'alt')
        ref_path (str): Path to the FASTA file containing the reference sequence
        IMAGE_HEIGHT (int): Height of the output image (number of rows)
        IMAGE_WIDTH (int): Width of the output image (number of columns)
        START_POSITION (int): Starting genomic position (used to calculate relative position in image)

    Returns:
        numpy.ndarray: A 2D array of integers representing the alternate allele image,
                      which is identical to the reference image except at the mutation position
    """
    # Create the reference image first (will be modified to create alternate image)
    variant_image_ref_ = create_singleton_image_ref(ref_path, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION)

    # Initialize an empty image (not actually used since we modify the reference image directly)
    variant_image_alt = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH)).astype(int)

    # Get the genomic position and encoded alternate allele value
    chr_pos = row['hg19_pos(1-based)']  # 1-based genomic position
    alt_ = encoders[row['alt']]  # Encode alternate allele (A=1, T=2, C=3, G=4)

    # Calculate the image position by:
    # 1. Getting relative position from start (0-based)
    # 2. Converting to 2D coordinates (row, column)
    image_location = int(chr_pos - START_POSITION)  # Convert to 0-based index
    variant_image_ref_[image_location // IMAGE_WIDTH, image_location % IMAGE_WIDTH] = alt_

    return variant_image_ref_


def create_rgb_image(row, ref_path, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION):
    """create a rgb image"""
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    image[:, :, 0] = create_singleton_image_score(row, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION)
    image[:, :, 1] = create_singleton_image_ref(ref_path, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION)
    image[:, :, 2] = create_singleton_image_alt(row, ref_path, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION)
    return image


def create_image(data, base_dir, cls, p, ref_path, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION):
    """
    Create and save RGB images for all mutations in the dataset.

    This function:
    1. Creates a directory to save images if it doesn't exist
    2. Iterates through each mutation in the input DataFrame
    3. Generates an RGB image for each mutation using score, reference, and alternate allele data
    4. Saves each image with a descriptive filename containing mutation information

    Args:
        data (pd.DataFrame): DataFrame containing mutation data (must include 'hg19_pos(1-based)', 'ref', 'alt')
        base_dir (str): Base directory where dataset will be saved
        cls (str): Classification type ('train', 'test', or 'val')
        p (str): Identifier for the mutation type ('p' or 'b')
        ref_path (str): Path to the reference sequence FASTA file
        IMAGE_HEIGHT (int): Height of the output image
        IMAGE_WIDTH (int): Width of the output image
        START_POSITION (int): Starting genomic position (used for image creation)

    Returns:
        None: Images are saved to disk in the specified directory
    """
    # Create directory for saving images if it doesn't exist
    save_dir = os.path.join(base_dir, 'dataset', cls, p)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Creating {cls} images in directory: {save_dir}")

    count = 0
    total_mutations = len(data)
    print(f"Processing {total_mutations} mutations for {cls} set...")

    for idx, row in data.iterrows():
        # Create RGB image combining score, reference and alternate allele channels
        image = create_rgb_image(row, ref_path, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION)

        # Extract mutation information for filename
        chr_pos = str(row['hg19_pos(1-based)'])
        ref = row['ref']
        alt = row['alt']

        # Generate descriptive filename
        name = os.path.join(save_dir, f"{chr_pos}_{ref}_{alt}_{count}_{p}.png")

        # Save image as PNG file
        cv2.imwrite(name, image)

        # Print progress every 100 mutations
        if (count + 1) % 100 == 0 or (count + 1) == total_mutations:
            print(f"Processed {count + 1}/{total_mutations} mutations")

        count += 1

    print(f"Successfully created {count} images in {save_dir}")


def split_train_test_val(df):
    """
    Split a DataFrame into training, testing, and validation sets with a fixed ratio.

    This function:
    1. Takes a pandas DataFrame as input
    2. Randomly shuffles the data with a fixed random state for reproducibility
    3. Splits the data into three subsets with ratios:
       - 60% training data (last portion)
       - 20% testing data (first portion)
       - 20% validation data (middle portion)

    Args:
        df (pd.DataFrame): Input DataFrame containing the data to be split

    Returns:
        tuple: Three DataFrames in the order (train_data, test_data, validation_data)
               with sizes 60%, 20%, 20% of original data respectively
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    data = df.copy()

    # Shuffle the data randomly with a fixed random state for reproducibility
    # frac=1.0 means return all rows in random order
    data: pd.DataFrame = data.sample(frac=1.0, random_state=123)

    # Get the dimensions of the DataFrame
    rows, cols = data.shape

    # Calculate split indices:
    # First 20% for test set (0 to split_index_1)
    split_index_1 = int(rows * 0.2)
    # Next 20% for validation set (split_index_1 to split_index_2)
    split_index_2 = int(rows * 0.4)

    # Extract test set (first 20%)
    data_test: pd.DataFrame = data.iloc[0:split_index_1, :]
    # Extract validation set (next 20%)
    data_validate: pd.DataFrame = data.iloc[split_index_1:split_index_2, :]
    # Extract training set (remaining 60%)
    data_train: pd.DataFrame = data.iloc[split_index_2:rows, :]

    return data_train, data_test, data_validate


def create_train_image(mutation_path, ref_path, base_dir, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION):
    """Process mutation data and create training/test/validation image sets."""

    # Extract mutation type from filename (last character before extension)
    p = os.path.basename(mutation_path).split('.')[0][-1]

    # Prepare mutation data (scale scores, combine with info columns)
    df_trans_res = prepare_mutation(mutation_path)

    # Split data into train/test/validation sets (60%/20%/20%)
    data_train, data_test, data_validate = split_train_test_val(df_trans_res)

    # Create image datasets for each split
    create_image(data_train, base_dir, 'train', p, ref_path, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION)
    create_image(data_test, base_dir, 'test', p, ref_path, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION)
    create_image(data_validate, base_dir, 'val', p, ref_path, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION)


class ImbalancedDatasetSampler(Sampler):
    """
    From https://github.com/ufoym/imbalanced-dataset-sampler

    Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
            self,
            dataset,
            labels: list = None,
            indices: list = None,
            num_samples: int = None,
            callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


def export_onnx_model(model, image_channels, image_height, image_width, output_path):
    x = torch.randn(
        1,
        image_channels,
        image_height,
        image_width,
        requires_grad=True,
    )

    bs = "batch_size"
    output_names = ["output"]
    dynamic_axes = {
        "input": {0: bs},  # variable length axes
        "output": {0: bs},
    }

    torch.onnx.export(
        model,  # model being run
        x.cuda(),  # model input (or a tuple for multiple inputs)
        output_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=False,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=output_names,  # the model's output names
        dynamic_axes=dynamic_axes,
        training=torch.onnx.TrainingMode.EVAL,
    )


def ort_inference(session, tensor):
    """
    Perform inference using ONNX Runtime with IO binding for optimized performance.

    This function:
    1. Creates an IO binding object for the ONNX Runtime session
    2. Binds the input tensor (converted to CPU numpy array) to the session input
    3. Binds the output tensor using the specified output name
    4. Executes the model with the bound inputs/outputs
    5. Copies the outputs back from the binding and converts to PyTorch tensor

    Args:
        session (ort.InferenceSession): The ONNX Runtime inference session
        tensor (torch.Tensor): Input tensor for inference (will be moved to CPU if not already)

    Returns:
        torch.Tensor: The model predictions as a PyTorch tensor
    """
    # Create IO binding object for the session
    io_binding = session.io_binding()

    # Bind input tensor (convert to CPU numpy array first)
    io_binding.bind_cpu_input(session.get_inputs()[0].name, tensor.cpu().numpy())

    # Bind output tensor by name
    io_binding.bind_output("output")

    # Execute the model with the bound inputs/outputs
    session.run_with_iobinding(io_binding)

    # Copy outputs from binding to CPU and convert to PyTorch tensor
    predictions = torch.from_numpy(io_binding.copy_outputs_to_cpu()[0])

    return predictions


def train(root_folder, base_dir, batch_size=32, num_epochs=10):
    img_size = 300
    new_size = (img_size, img_size)
    transform = transforms.Compose([
        transforms.Resize(new_size),
        transforms.ToTensor(),
    ])

    path_to_train_images = os.path.join(root_folder, "train")
    trainset = torchvision.datasets.ImageFolder(
        root=path_to_train_images,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=ImbalancedDatasetSampler(trainset),
        num_workers=4,
    )
    path_to_val_images = os.path.join(root_folder, "val")
    valset = torchvision.datasets.ImageFolder(
        root=path_to_val_images,
        transform=transform
    )
    valloader = torch.utils.data.DataLoader(valset, batch_size=1,
                                            shuffle=False, num_workers=4)
    path_to_test_images = os.path.join(root_folder, "test")
    testset = torchvision.datasets.ImageFolder(
        root=path_to_test_images,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=4)
    model = torchvision.models.resnet34(num_classes=2)
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for (inputs, labels) in tqdm(valloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                y_true.append(labels.detach().cpu()[0])
                y_pred.append(predicted.detach().cpu().numpy()[0])

        recall = round(recall_score(y_true, y_pred), 4)
        precision = round(precision_score(y_true, y_pred), 4)
        acc = round(accuracy_score(y_true, y_pred), 4)
        print(f"Epoch: {epoch} - recall={recall} precision={precision} accuracy={acc}")
    print('Finished Training')

    # Export model
    workdir = os.path.join(base_dir, 'model')
    os.makedirs(workdir, exist_ok=True)

    outname = "all_resnet34_300"
    os.makedirs(workdir, exist_ok=True)

    model_path = os.path.join(workdir, f"{outname}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved at {model_path}")
    output_path = os.path.join(workdir, f"{outname}.onnx")
    export_onnx_model(model, 3, img_size, img_size, output_path)
    print(f"Final model exported at {output_path}")
    providers = ('CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider')
    session = ort.InferenceSession(
        output_path, providers=[providers]
    )
    y_true = []
    y_pred = []
    for data in tqdm(testloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = ort_inference(session, inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_true.append(labels.detach().cpu()[0])
        y_pred.append(predicted.detach().cpu().numpy()[0])
    recall = round(recall_score(y_true, y_pred), 4)
    precision = round(precision_score(y_true, y_pred), 4)
    acc = round(accuracy_score(y_true, y_pred), 4)
    auc = round(roc_auc_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred), 4)
    print(
        f"Perfomance on ONNX model: recall={recall} precision={precision} accuracy={acc} auc={auc} f1={f1}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(description='Process gene mutation data')
    parser.add_argument('--gene', default='PKD1', help='Gene name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    args = parser.parse_args()

    gene = args.gene
    batch_size = args.batch_size
    num_epochs = args.epochs

    base_dir = os.getcwd()  # Current directory (BRCA2)
    train_dir = base_dir
    ref_dir = os.path.join(base_dir, 'ref')
    # Define paths using os.path.join for better cross-platform compatibility
    mutation_path_p = os.path.join(ref_dir, f'{gene}_p.txt')
    mutation_path_b = os.path.join(ref_dir, f'{gene}_b.txt')
    ref_path = os.path.join(ref_dir, f'{gene}_ref.fa')
    bed_path = os.path.join(ref_dir, f'{gene}_.bed')
    # Process the data
    IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION = get_image_size(bed_path)
    print(f"start create {gene} gene images:")
    create_train_image(mutation_path_p, ref_path, train_dir, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION)
    create_train_image(mutation_path_b, ref_path, train_dir, IMAGE_HEIGHT, IMAGE_WIDTH, START_POSITION)
    print(f"Create {gene} gene images complete!")
    root_folder = os.path.join(base_dir, "dataset")
    print(f"start train {gene} gene model with batch_size={batch_size}, epochs={num_epochs}:")
    train(root_folder, base_dir, batch_size, num_epochs)
    print(f"train {gene} gene model complete!")


if __name__ == '__main__':
    main()