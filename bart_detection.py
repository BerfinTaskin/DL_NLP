import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel
from sklearn.metrics import matthews_corrcoef
from optimizer import AdamW
import json
import os # Ensure os is imported for makedirs

TQDM_DISABLE = False

class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=26):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained("facebook/bart-large", local_files_only=False)
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        # Use the BartModel to obtain the last hidden state
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        # Add an additional fully connected layer to obtain the logits
        logits = self.classifier(cls_output)

        # Return the probabilities
        probabilities = self.sigmoid(logits)
        return probabilities


# This mapping should be defined globally or passed appropriately if needed elsewhere.
# It maps original ETPC paraphrase type IDs to a 0-indexed label space of 26.
DROPPED_TYPES = {12, 19, 20, 23, 27}
ORIGINAL_TYPE_TO_INDEX_MAP = {}
current_idx = 0
for i in range(1, 32): # Assuming original types go up to 31
    if i not in DROPPED_TYPES:
        ORIGINAL_TYPE_TO_INDEX_MAP[i] = current_idx
        current_idx += 1
# Expected: current_idx should be 26 after this loop.

def transform_data(dataset: pd.DataFrame, tokenizer_name: str = "facebook/bart-large", max_length: int = 512, batch_size: int = 16):
    """
    dataset: pd.DataFrame

    Turn the data to the format you want to use.

    1. Extract the sentences from the dataset. We recommend using the already split
    sentences in the dataset.
    2. Use the AutoTokenizer from_pretrained to tokenize the sentences and obtain the
    input_ids and attention_mask.
    3. Currently, the labels are in the form of [6, 6, 6, 25, 25, 29]. This means that
    the sentence pair contains type 6, 25, and 29. Turn this into a binary form, where the
    label becomes [0, 0, 0, 0, 0, 1, ..., 1, 0, 0, 1, 0, 0].
    IMPORTANT: You will find that the dataset contains types up to 31, but some are not
    assigned. You need to drop 12, 19, 20, 23 and 27 when creating the binary labels.
    This way you should end up with a binary label of size 26.
    Be careful that the test-student.csv does not
    have the paraphrase_types column. You should return a DataLoader without the labels.
    4. Use the input_ids, attention_mask, and binary labels to create a TensorDataset.
    Return a DataLoader with the TensorDataset. You can choose a batch size of your
    choice.
    """
    # 1. Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=False)

    # 2. Prepare lists to store processed data
    all_input_ids = []
    all_attention_masks = []
    all_binary_labels = []

    # 3. Check if 'paraphrase_type_ids' column exists for label processing
    has_labels = 'paraphrase_type_ids' in dataset.columns

    # 4. Iterate over each row in the DataFrame
    for index, row in dataset.iterrows():
        # a. Combine sentence1 and sentence2
        # BART uses </s> as a separator.
        # For sentence pair tasks, typical input is: <s> sentence1 </s> </s> sentence2 </s>
        # The tokenizer.encode_plus handles adding the initial <s> and final </s>.
        # We need to ensure the two sentences are distinctly represented.
        # A common way is to provide them as a pair to the tokenizer.
        sentence1 = str(row['sentence1'])
        sentence2 = str(row['sentence2'])

        # b. Tokenize the sentence pair
        encoded_dict = tokenizer.encode_plus(
            sentence1,
            sentence2, # text_pair
            add_special_tokens=True,     # Add '<s>' and '</s>'
            max_length=max_length,       # Pad & truncate all sentences.
            padding='max_length',        # Pad to max_length
            truncation=True,             # Truncate to max_length if necessary
            return_attention_mask=True, # Construct attn. masks.
            return_tensors='pt',         # Return pytorch tensors.
        )

        # c. Append tokenized inputs
        all_input_ids.append(encoded_dict['input_ids'].squeeze(0)) # Remove batch dimension
        all_attention_masks.append(encoded_dict['attention_mask'].squeeze(0))

        # d. Process labels if they exist
        if has_labels:
            # i. Parse the string representation of the list
            try:
                # Ensure the string is valid JSON. Sometimes they might be single-quoted.
                type_ids_str = str(row['paraphrase_type_ids']).replace("'", '"')
                original_types = json.loads(type_ids_str)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse paraphrase_type_ids for row {index}: {row['paraphrase_type_ids']}")
                original_types = [] # Default to no types if parsing fails

            # ii. Create a binary vector (size 26)
            binary_label_vector = torch.zeros(26, dtype=torch.float) # Use float for BCELoss

            # iii. Populate the binary vector based on ORIGINAL_TYPE_TO_INDEX_MAP
            for pt_type in set(original_types): # Use set to handle duplicates like [6,6,6]
                if pt_type in ORIGINAL_TYPE_TO_INDEX_MAP:
                    mapped_idx = ORIGINAL_TYPE_TO_INDEX_MAP[pt_type]
                    if 0 <= mapped_idx < 26:
                        binary_label_vector[mapped_idx] = 1.0
                    # else:
                    #     print(f"Warning: Original type {pt_type} not in mapping or out of 0-25 range after mapping.")

            all_binary_labels.append(binary_label_vector)

    # 5. Convert lists to PyTorch tensors
    all_input_ids_tensor = torch.stack(all_input_ids)
    all_attention_masks_tensor = torch.stack(all_attention_masks)

    # 6. Create TensorDataset and DataLoader
    if has_labels:
        all_binary_labels_tensor = torch.stack(all_binary_labels)
        tensor_dataset = TensorDataset(all_input_ids_tensor, all_attention_masks_tensor, all_binary_labels_tensor)
        # Shuffle training data
        data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    else:
        # For the test set (no labels)
        tensor_dataset = TensorDataset(all_input_ids_tensor, all_attention_masks_tensor)
        # Do not shuffle test data
        data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)

    return data_loader

def train_model(model, train_data, dev_data, device, num_epochs=5, learning_rate=1e-5, early_stopping_patience=3):
    """
    Train the model. You can use any training loop you want. We recommend starting with
    AdamW as your optimizer. You can take a look at the SST training loop for reference.
    Think about your loss function and the number of epochs you want to train for.
    You can also use the evaluate_model function to evaluate the
    model on the dev set. Print the training loss, training accuracy, and dev accuracy at
    the end of each epoch.
    """
    # 1. Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 2. Set up the loss function
    # BCELoss is suitable for multi-label binary classification problems
    # where the model outputs probabilities (after a sigmoid).
    criterion = nn.BCELoss()
    criterion = criterion.to(device) # Move loss function to device

    print(f"Starting training for {num_epochs} epochs on {device}...")

    # --- Early Stopping Variables ---
    best_dev_loss = float('inf')
    epochs_no_improve = 0
    patience = early_stopping_patience # Use the patience passed as an argument
    # --- End Early Stopping Variables ---

    for epoch_i in range(num_epochs):
        print(f"\n======== Epoch {epoch_i + 1} / {num_epochs} ========")
        print("Training...")

        # Reset the total loss for this epoch.
        total_train_loss = 0
        model.train() # Put model in training mode.

        for step, batch in enumerate(tqdm(train_data, desc="Batch Progress", disable=TQDM_DISABLE)):
            # Move batch to device
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad() # Clear previously calculated gradients

            # Perform a forward pass (evaluate the model on this training batch).
            # This will return the probabilities.
            probabilities = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            # Calculate the loss.
            # BCELoss expects target labels to be float.
            loss = criterion(probabilities, b_labels.float())
            total_train_loss += loss.item()

            loss.backward() # Perform a backward pass to calculate the gradients.
            optimizer.step() # Update parameters

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_data)
        print(f"  Average training loss: {avg_train_loss:.4f}")

        # --- Validation ---
        print("Running Validation...")
        dev_accuracy, dev_mcc = evaluate_model(model, dev_data, device)

        # --- Calculate Validation Loss for Early Stopping ---
        # Temporarily set model to eval mode to calculate validation loss
        model.eval()
        total_dev_loss = 0
        with torch.no_grad():
            for batch in dev_data:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)
                probabilities = model(input_ids=b_input_ids, attention_mask=b_input_mask)
                loss = criterion(probabilities, b_labels.float())
                total_dev_loss += loss.item()
        avg_dev_loss = total_dev_loss / len(dev_data)
        model.train() # Set model back to training mode
        # --- End Calculate Validation Loss ---

        print(f"  Development Accuracy: {dev_accuracy:.4f}")
        print(f"  Development Matthews Correlation Coefficient: {dev_mcc:.4f}")
        print(f"  Average Development Loss: {avg_dev_loss:.4f}") # Print development loss

        # --- Early Stopping Logic ---
        if avg_dev_loss < best_dev_loss:
            best_dev_loss = avg_dev_loss
            epochs_no_improve = 0
            # Optional: Save the best model state here
            # torch.save(model.state_dict(), 'best_model.pth')
            print("  Development loss improved!")
        else:
            epochs_no_improve += 1
            print(f"  Development loss did not improve. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch_i + 1} epochs. No improvement in development loss for {patience} consecutive epochs.")
            break # Exit the training loop
        # --- End Early Stopping Logic ---

    print("\nTraining complete!")
    return model


def test_model(model, test_data, test_ids, device):
    """
    Test the model. Predict the paraphrase types for the given sentences and return the results in form of
    a Pandas dataframe with the columns 'id' and 'Predicted_Paraphrase_Types'.
    The 'Predicted_Paraphrase_Types' column should contain the binary array of your model predictions.
    Return this dataframe.
    """
    print("Starting testing...")
    model.eval() # Put model in evaluation mode.

    all_predictions_binary = []

    with torch.no_grad(): # Saves memory and computations, as no gradients are needed.
        for batch in tqdm(test_data, desc="Testing Batch", disable=TQDM_DISABLE):
            # Test data loader should only yield input_ids and attention_mask
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)

            # Get probabilities from the model
            probabilities = model(input_ids=b_input_ids, attention_mask=b_input_mask)

            # Convert probabilities to binary predictions (0 or 1)
            # A common threshold is 0.5
            predicted_labels_binary = (probabilities > 0.5).int()

            all_predictions_binary.append(predicted_labels_binary.cpu()) # Move to CPU before appending

    # Concatenate predictions from all batches
    all_predictions_tensor = torch.cat(all_predictions_binary, dim=0)

    # Convert the tensor of predictions to a list of lists for the DataFrame
    # Each inner list will be a binary array of 26 elements
    predictions_list_of_lists = all_predictions_tensor.numpy().tolist()

    # Create the results DataFrame as required
    results_df = pd.DataFrame({
        'id': test_ids, # This should be a pd.Series or list of IDs from the test set
        'Predicted_Paraphrase_Types': predictions_list_of_lists
    })

    print("Testing complete!")
    return results_df


def evaluate_model(model, test_data, device):
    """
    This function measures the accuracy of our model's prediction on a given train/validation set
    We measure how many of the 26 paraphrase types the model has predicted correctly for each data point..
    """
    all_pred = []
    all_labels = []
    model.eval() # Put model in evaluation mode

    with torch.no_grad(): # Deactivate autograd for evaluation
        for batch in test_data: # Assuming test_data here is actually dev_data with labels
            # Dev data loader should yield input_ids, attention_mask, AND labels
            if len(batch) == 3:
                input_ids, attention_mask, labels = batch
                labels = labels.to(device) # Move labels to device
            else: # Should not happen if dev_data is correctly prepared
                print("Warning: Batch in evaluate_model does not contain labels.")
                continue

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Convert probabilities to binary predictions
            predicted_labels = (outputs > 0.5).int()

            all_pred.append(predicted_labels.cpu()) # Move to CPU
            all_labels.append(labels.cpu()) # Move to CPU

    if not all_pred: # Handle empty dev set or issues
        print("Warning: No predictions made in evaluate_model. Returning 0 for metrics.")
        return 0.0, 0.0

    all_predictions_tensor = torch.cat(all_pred, dim=0)
    all_true_labels_tensor = torch.cat(all_labels, dim=0)

    true_labels_np = all_true_labels_tensor.numpy()
    predicted_labels_np = all_predictions_tensor.numpy()

    # Compute the accuracy for each label
    accuracies = []
    matthews_coefficients = []
    for label_idx in range(true_labels_np.shape[1]): # Iterate over each of the 26 labels
        correct_predictions = np.sum(true_labels_np[:, label_idx] == predicted_labels_np[:, label_idx])
        total_samples = true_labels_np.shape[0]
        if total_samples == 0: # Avoid division by zero
            label_accuracy = 0.0
            matth_coef = 0.0
        else:
            label_accuracy = correct_predictions / total_samples
            # matthews_corrcoef can raise ValueError if predictions are all same for a label
            try:
                matth_coef = matthews_corrcoef(true_labels_np[:, label_idx], predicted_labels_np[:, label_idx])
            except ValueError:
                matth_coef = 0.0 # Or handle as appropriate, e.g. by checking for constant arrays

        accuracies.append(label_accuracy)
        matthews_coefficients.append(matth_coef)

    # Calculate the average accuracy over all labels
    accuracy = np.mean(accuracies) if accuracies else 0.0
    matthews_coefficient = np.mean(matthews_coefficients) if matthews_coefficients else 0.0

    model.train() # Set model back to training mode
    return accuracy, matthews_coefficient

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dev_split_ratio", type=float, default=0.1)
    parser.add_argument("--model_name", type=str, default="facebook/bart-base")
    parser.add_argument("--early_stopping_patience", type=int, default=4)

    # ⬇️ Add these two new arguments
    parser.add_argument("--approach", type=str, required=True,
                        help="Short description/name of the experiment")
    parser.add_argument("--job_id", type=str, required=True,
                        help="SLURM job ID for tracking")

    return parser.parse_args()



def finetune_paraphrase_detection(args):
    print(f"Using arguments: {args}")
    model = BartWithClassifier(num_labels=26) # Pass model_name
    device = torch.device("cuda") if args.use_gpu and torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)

    print("Loading data...")
    full_train_dataset_df = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset_df = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv")

    # --- Train/Validation Split ---
    # Shuffle the dataset
    full_train_dataset_df = full_train_dataset_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    num_dev_samples = int(len(full_train_dataset_df) * args.dev_split_ratio)
    if num_dev_samples == 0 and len(full_train_dataset_df) > 0 : # ensure at least 1 sample for dev if possible
        num_dev_samples = 1
    if num_dev_samples > len(full_train_dataset_df) -1 : # ensure at least 1 sample for train
        num_dev_samples = len(full_train_dataset_df) -1 if len(full_train_dataset_df) > 0 else 0


    if num_dev_samples > 0 :
        dev_df = full_train_dataset_df.iloc[:num_dev_samples]
        train_df = full_train_dataset_df.iloc[num_dev_samples:]
    else: # Not enough data to split
        dev_df = full_train_dataset_df.copy() # Use full for dev if too small
        train_df = full_train_dataset_df.copy() # Use full for train

    print(f"Training with {len(train_df)} samples, validating with {len(dev_df)} samples.")

    # Transform data into DataLoaders
    print("Transforming training data...")
    train_dataloader = transform_data(train_df, tokenizer_name=args.model_name, max_length=args.max_length, batch_size=args.batch_size)
    print("Transforming development data...")
    dev_dataloader = transform_data(dev_df, tokenizer_name=args.model_name, max_length=args.max_length, batch_size=args.batch_size)
    print("Transforming test data...")
    test_dataloader = transform_data(test_dataset_df, tokenizer_name=args.model_name, max_length=args.max_length, batch_size=args.batch_size)

    print(f"Loaded {len(train_df)} training samples for DataLoader.") # Corrected to reflect split

    # Train the model
    # Pass early stopping patience from args
    model = train_model(model, train_dataloader, dev_dataloader, device,
                        num_epochs=args.num_epochs,
                        learning_rate=args.learning_rate,
                        early_stopping_patience=args.early_stopping_patience) # Pass patience to train_model

    print("\nTraining finished.")
    print("Evaluating on the development set one last time...")
    accuracy, matthews_corr = evaluate_model(model, dev_dataloader, device) # Evaluate on dev_dataloader
    print(f"Final Development Accuracy: {accuracy:.3f}")
    print(f"Final Development Matthews Correlation Coefficient: {matthews_corr:.3f}")

    # Test the model
    test_ids = test_dataset_df["id"] # Get IDs from the original test DataFrame
    test_results_df = test_model(model, test_dataloader, test_ids, device) # Pass test_dataloader and ids

    # Save test predictions
    output_dir = "predictions/bart/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    test_results_df.to_csv(os.path.join(output_dir, "etpc-paraphrase-detection-test-output.csv"), index=False)
    print(f"Test predictions saved to {os.path.join(output_dir, 'etpc-paraphrase-detection-test-output.csv')}")

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
