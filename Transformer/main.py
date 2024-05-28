import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from transformer import TransformerClassifier, TransformerLanguageModel
from torchsummary import summary
from utilities import Utilities


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _, _ = classifier((X, None))
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        _, loss, _ = decoderLMmodel((X, None), Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    # Classification Task
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    # Test Loader
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    # Language Modelling
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    # Test Loaders
    inputfile = "speechesdataset/test_LM_obama.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtestText = f.read()
    test_LM_dataset_obama = LanguageModelingDataset(tokenizer, lmtestText,  block_size)
    test_LM_loader_obama = DataLoader(test_LM_dataset_obama, batch_size=batch_size, shuffle=True)

    inputfile = "speechesdataset/test_LM_hbush.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtestText = f.read()
    test_LM_dataset_hbush = LanguageModelingDataset(tokenizer, lmtestText,  block_size)
    test_LM_loader_hbush = DataLoader(test_LM_dataset_hbush, batch_size=batch_size, shuffle=True)

    inputfile = "speechesdataset/test_LM_wbush.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtestText = f.read()
    test_LM_dataset_wbush = LanguageModelingDataset(tokenizer, lmtestText,  block_size)
    test_LM_loader_wbush = DataLoader(test_LM_dataset_wbush, batch_size=batch_size, shuffle=True)


    print("\nChoose a task")
    print("1. Classification Task")
    print("2. Language Modelling")
    print("3. Exploration")
    print("4. Exit")
    task = input("Enter a number: ")
    while True:
        if task == "1":
            # Load the classification model
            classifier_model = TransformerClassifier(tokenizer.vocab_size, block_size, n_embd, n_head,
                                                    n_input, n_hidden, n_output, n_layer, device).to(device)
            
            # Define the optimizer
            optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=learning_rate)

            # summary(classifier_model, tuple([32]))

            # for the classification  task, you will train for a fixed number of epochs like this:
            print("\nPart 1: Classification Task")
            for epoch in range(epochs_CLS):
                # break
                for xb, yb in train_CLS_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    # CLS training code here
                    # print(xb[0])
                    y_pred, loss, attention_maps = classifier_model((xb, None), yb)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                print(f"Epoch {epoch}: Train Loss {loss.item():.3f}; Train Accuracy {compute_classifier_accuracy(classifier_model, train_CLS_loader):.3f}; Test Accuracy {compute_classifier_accuracy(classifier_model, test_CLS_loader):.3f}")

            print("\nSanity check for classification")
            sentence1 = "Health care costs are rising at the slowest rate in 50 years."
            sentence2 = "It is fair to hold our views up to scrutiny."
            utility_cls = Utilities(tokenizer, classifier_model)
            print(f"Sentence 1: {sentence1}")
            utility_cls.sanity_check(sentence1, block_size, "Encoder_sentence1_attention_map")
            print(f"Sentence 2: {sentence2}")
            utility_cls.sanity_check(sentence2, block_size, "Encoder_sentence2_attention_map")
            print("Sanity check completed!")

            task = input("\nEnter other numbers to run another task (1,2,3,4): ")


        elif task == "2":
            # Load the language model
            language_model = TransformerLanguageModel(tokenizer.vocab_size, block_size, n_embd, n_head,
                                                    n_input, n_hidden, n_layer, device).to(device)
            
            # Define the optimizer
            optimizer = torch.optim.AdamW(language_model.parameters(), lr=learning_rate)

            # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
            print("\nPart 2: Language Modelling")
            for i, (xb, yb) in enumerate(train_LM_loader):
                if i > max_iters:
                    break

                xb, yb = xb.to(device), yb.to(device)
                # LM training code here
                y_pred, loss, attention_maps = language_model((xb, None), yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if i%eval_interval == 0:
                    print(f"Iter {i}: Train Loss {loss.item():.3f}; Train Perplexity {compute_perplexity(language_model, train_LM_loader):.4f}")

            print(f"Obama Perplexity {compute_perplexity(language_model, test_LM_loader_obama):.4f}; H. Bush Perplexity {compute_perplexity(language_model, test_LM_loader_hbush):.4f}; W. Bush Perplexity {compute_perplexity(language_model, test_LM_loader_wbush):.4f}")

            print("\nSanity check for language modelling")
            sentence1 = "Health care costs are rising at the slowest rate in 50 years."
            sentence2 = "It is fair to hold our views up to scrutiny."
            utility_lm = Utilities(tokenizer, language_model)
            print(f"Sentence 1: {sentence1}")
            utility_lm.sanity_check(sentence1, block_size, "Decoder_sentence1_attention_map")
            print(f"Sentence 2: {sentence2}")
            utility_lm.sanity_check(sentence2, block_size, "Decoder_sentence2_attention_map")
            print("Sanity check completed!")

            task = input("\nEnter other numbers to run another task (1,2,3,4): ")


        elif task == "3":
            # Architectural Exploration
            print("\nPart 3a: Architectural Exploration")
            print("Using ALiBi for Classification")

            # Load the classification model
            classifier_model = TransformerClassifier(tokenizer.vocab_size, block_size, n_embd, n_head,
                                                    n_input, n_hidden, n_output, n_layer, device, alibi=True).to(device)
            
            # Define the optimizer
            optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=learning_rate)

            for epoch in range(epochs_CLS):
                for xb, yb in train_CLS_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    # CLS training code here
                    # print(xb[0])
                    y_pred, loss, attention_maps = classifier_model((xb, None), yb)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                print(f"Epoch {epoch}: Train Loss {loss.item():.3f}; Train Accuracy {compute_classifier_accuracy(classifier_model, train_CLS_loader):.3f}; Test Accuracy {compute_classifier_accuracy(classifier_model, test_CLS_loader):.3f}")

            print("\nUsing ALiBi for Language Modelling")
            language_model = TransformerLanguageModel(tokenizer.vocab_size, block_size, n_embd, n_head,
                                                    n_input, n_hidden, n_layer, device).to(device)
            
            optimizer = torch.optim.AdamW(language_model.parameters(), lr=learning_rate)

            for i, (xb, yb) in enumerate(train_LM_loader):
                if i > max_iters:
                    break
                xb, yb = xb.to(device), yb.to(device)
                y_pred, loss, attention_maps = language_model((xb, None), yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if i%eval_interval == 0:
                    print(f"Iter {i}: Train Loss {loss.item():.3f}; Train Perplexity {compute_perplexity(language_model, train_LM_loader):.4f}")

            print(f"Obama Perplexity {compute_perplexity(language_model, test_LM_loader_obama):.4f}; H. Bush Perplexity {compute_perplexity(language_model, test_LM_loader_hbush):.4f}; W. Bush Perplexity {compute_perplexity(language_model, test_LM_loader_wbush):.4f}")


            # Performance Improvement
            print("\nPart 3b: Performance Improvement of Classification Task")

            # New Hyperparameters
            new_learning_rate = 3e-3
            new_n_head = 4
            new_n_hidden = 128
            new_weight_decay = 0
            new_dropout = 0.2
            new_n_layer = 4

            print(f"Using ALiBi with new hyperparameters - lr={new_learning_rate}; heads={new_n_head}; hidden_size={new_n_hidden}; weight_decay={new_weight_decay}; n_layer={new_n_layer}")

            classifier_model = TransformerClassifier(tokenizer.vocab_size, block_size, n_embd, new_n_head,
                                                    n_input, new_n_hidden, n_output, new_n_layer, device, new_dropout, alibi=True).to(device)
        
            optimizer = torch.optim.Adam(classifier_model.parameters(), lr=new_learning_rate, weight_decay=new_weight_decay)
            # optimizer = torch.optim.SGD(classifier_model.parameters(), lr=new_learning_rate, momentum=0.5, nesterov=True)

            for epoch in range(epochs_CLS):
                for xb, yb in train_CLS_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    y_pred, loss, attention_maps = classifier_model((xb, None), yb)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                print(f"Epoch {epoch}: Train Loss {loss.item():.3f}; Train Accuracy {compute_classifier_accuracy(classifier_model, train_CLS_loader):.3f}; Test Accuracy {compute_classifier_accuracy(classifier_model, test_CLS_loader):.3f}")

            task = input("\nEnter other numbers to run another task (1,2,3,4): ")


        elif task == "4":
            print("\nExiting...")
            break


        else:
            task = input("Enter a valid number (1,2,3,4): ")


if __name__ == "__main__":
    main()
