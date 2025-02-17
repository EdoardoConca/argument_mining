import os
import pandas as pd
from transformers import AutoTokenizer
import random
import gc
import progressbar
import os
import nltk
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from itertools import chain, cycle
from IPython.display import display_html,display, HTML
import urllib.request
import zipfile
import re
from collections import OrderedDict
from collections import Counter
from tabulate import tabulate
import textwrap
from tqdm import tqdm
import os
from typing import *
import torch
from transformers import AutoTokenizer
from tabulate import tabulate
import pandas as pd
from collections import Counter
import plotly.express as px
#MODEL
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, AutoModel
from torchcrf import CRF
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report,precision_recall_fscore_support, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from models.ac_detector import BertGRUCRF
from models.relation_classifier import BERTSentClf

pbar = None

def show_progress(block_num, block_size, total_size):
    """
    Displays a progress bar to track the download progress of a file.

    Parameters:
        block_num (int): The current block number being downloaded.
        block_size (int): The size of each block.
        total_size (int): The total size of the file being downloaded.

    Returns:
        None
    """
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

def display_special_tokens(tokenizer_name):
    """
    Mostra i token speciali e le loro codifiche per un tokenizer Hugging Face.

    Args:
        tokenizer_name (str): Il nome o percorso del tokenizer Hugging Face (es. "bert-base-uncased").

    Returns:
        dict: Dizionario contenente i token speciali e i loro ID.
    """
    # Carica il tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Ottieni i token speciali e i loro ID
    special_tokens = tokenizer.special_tokens_map
    token_ids = {key: tokenizer.convert_tokens_to_ids(token) for key, token in special_tokens.items()}

    # Stampa i token speciali
    print("\nSpecial Tokens and their IDs:")
    for token_name, token_value in special_tokens.items():
        token_id = token_ids[token_name]
        print(f"{token_name}: {token_value} (ID: {token_id})")

    return special_tokens, token_ids

def set_reproducibility(seed):
    """
    Sets the random seed for reproducibility in PyTorch experiments.

    Parameters:
        seed (int): The seed value to set.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_mappings(tag_list: List[str]) -> Dict[str, int]:
    """
    Creates two mappings: one from tags to indices and one from indices to tags.

    Parameters:
        tag_list (list): A list of tags.

    Returns:
        tuple: A tuple containing two dictionaries:
            - tag_to_idx (dict): A dictionary mapping each tag to a unique index.
            - idx_to_tag (dict): A dictionary mapping each index to its corresponding tag.

    """
    tag_to_idx = {tag: idx for idx, tag in enumerate(sorted(set(tag_list)))}
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}
    return tag_to_idx, idx_to_tag

def convert_tags_to_ids(tags: List[str], tag_to_id: Dict[str, int]) -> List[int]:
    """
    Converts a list of tags to their corresponding IDs.

    Parameters:
        tags (list): List of tags to convert.
        tag_to_id (dict): Dictionary mapping tags to IDs.

    Returns:
        list: List of tag IDs.
    """
    return [tag_to_id.get(tag, -1) for tag in tags]

def convert_ids_to_tags(tag_ids: List[int], id_to_tag: Dict[int, str]) -> List[str]:
    """
    Converts a list of IDs to their corresponding tags.

    Parameters:
        tags_ids (list): List of ids to convert.
        id_to_tag (dict): Dictionary mapping IDs to tags-

    Returns:
        list: List of tags.
    """

    if isinstance(tag_ids[0], list):
        return [[id_to_tag.get(tag_id, "UNK") for tag_id in sub_list] for sub_list in tag_ids]
    else:
        return [id_to_tag.get(tag_id, "UNK") for tag_id in tag_ids]

def create_bio_labels(annotations, encoding, tokenizer):
    offset_mapping = encoding['offset_mapping']
    tokenized_text = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    bio_labels = ['O'] * len(offset_mapping)  # initialize all labels to 'O'
    start_end_positions = {}

    for i in range(len(annotations)):
        id_component, label_type, start, end, _ , _= annotations[i].values()
        annotation_start, annotation_end = None, None
        for i, (token_start, token_end) in enumerate(offset_mapping):
            if token_start >= start and token_end <= end:
                if token_start == start:
                    bio_labels[i] = f"B-{label_type}"  # assign 'B' label to the first token
                    annotation_start = i
                else:
                    bio_labels[i] = f"I-{label_type}"  # assign 'I' label to the rest of the tokens
                    annotation_end = i
        if annotation_start is not None and annotation_end is not None:
            start_end_positions[id_component] = (annotation_start, annotation_end)

    #special tokens 
    for i, token in enumerate(tokenized_text):
        if token == '[CLS]' or token == '[SEP]' or token == ['PAD']: 
            bio_labels[i] = 'O'  

    aligned_labels = [
        f"I-{label.split('-')[1]}" if token.startswith("##") and label != "O" else label
        for token, label in zip(tokenized_text, bio_labels)
    ]

    return tokenized_text, aligned_labels, start_end_positions


def parse_ann_file(ann_file, file_name, task_type="relation_classification"):
    """
    Parses the .ann file based on task type.

    Parameters:
        ann_file (str): The path to the .ann file.
        file_name (str): The name of the file for reference in the DataFrame.
        task_type (str): Specifies the task - "relation_classification" or "component_detection".

    Returns:
        list: A list of dictionaries with parsed data for each task type.
    """
    components = {}
    relations = {}

    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts[0].startswith('T'):
                comp_id = parts[0]
                comp_type, start, end = parts[1].split()[:3]
                if comp_type == "MajorClaim":
                    comp_type = "Claim"
                text = parts[2]
                components[comp_id] = {
                    'type': comp_type, 'start': int(start), 'end': int(end), 'text': text
                }
            elif parts[0].startswith('R'):
                rel_type = parts[1].split()[0]
                if rel_type == 'Partial-Attack':
                    rel_type = 'Attack'
                arg1 = parts[1].split()[1].split(':')[1]
                arg2 = parts[1].split()[2].split(':')[1]
                relations[(arg1, arg2)] = rel_type

    rows = []

    if task_type == "relation_classification":
        component_ids = list(components.keys())
        all_combinations = list(itertools.permutations(component_ids, 2))

        for arg1_id, arg2_id in all_combinations:
            rel_type = relations.get((arg1_id, arg2_id), 'NoRelation')
            label = 'Relation' if rel_type in ['Attack', 'Partial-Attack', 'Support'] else 'NoRelation' #comment for multiclass
            rows.append({
                'Arg1_Text': components[arg1_id]['text'],
                'Arg2_Text': components[arg2_id]['text'],
                'Arg1_ID': arg1_id,
                'Arg2_ID': arg2_id,
                'Label': label,
                'Arg1_Type': components[arg1_id]['type'],
                'Arg2_Type': components[arg2_id]['type'],
                'File': file_name
            })
    else:
        for comp_id, comp_data in components.items():
            rows.append({
                'Component_ID': comp_id,
                'Label_Type': comp_data['type'],
                'Start': comp_data['start'],
                'End': comp_data['end'],
                'Text': comp_data['text'],
                'File': file_name
            })

    return rows

def create_dataframe_from_directory(directory, tag_to_idx, tokenizer, task_type="relation_classification", max_length=512):
    """
    Creates a DataFrame from the files in a directory for the specified task type.

    Parameters:
        directory (str): Path to the directory containing .txt and .ann files.
        tag_to_idx (dict): Dictionary mapping tags to indices.
        task_type (str): "relation_classification" or "component_detection".
        tokenizer: Tokenizer for creating BIO labels.
        max_length (int): Max token length for tokenizer.

    Returns:
        pd.DataFrame: DataFrame with parsed data.
    """
    dataset = []
    for file in os.listdir(directory):
        if file.endswith(".ann"):
            ann_file = os.path.join(directory, file)
            txt_file = os.path.join(directory, file.replace(".ann", ".txt"))
            file_name = os.path.basename(file)

            if task_type == "relation_classification":
                rows = parse_ann_file(ann_file, file_name, task_type)
                dataset.extend(rows)
            elif task_type == "component_detection":
                with open(txt_file, 'r', encoding='utf-8') as f:
                    raw_text = f.read().lower()

                annotations = parse_ann_file(ann_file, file_name, task_type)

                encoding = tokenizer(raw_text, return_offsets_mapping=True,
                                     truncation=True, max_length=max_length, padding='max_length')

                tokenized_text, tags, new_positions = create_bio_labels(annotations, encoding, tokenizer)

                encoded_tags = convert_tags_to_ids(tags, tag_to_idx)

                dataset.append((raw_text, tokenized_text, encoding['input_ids'], encoding['attention_mask'], tags, encoded_tags, new_positions, file_name))

    if task_type == "relation_classification":
        columns = ['Arg1_Text', 'Arg2_Text', 'Arg1_ID', 'Arg2_ID', 'Label', 'Arg1_Type', 'Arg2_Type', 'File']
    else:
        columns = ['raw_text', 'tokenized_text', 'input_ids', 'attention_mask', 'tags', 'encoded_tags', 'new_start_end_positions', 'file_name']

    return pd.DataFrame(dataset, columns=columns)


def load_model(model_name, model_path, device, **kwargs):
    """
    Loads a pre-trained model from a file.

    Parameters: 
        model_name (str): The name of the model class.
        model_path (str): The path to the model file.
        device (torch.device): The device to load the model on.
        **kwargs: Additional keyword arguments for model configuration.

    Returns:
        nn.Module: The loaded model.
    """
    ac_detection_kgwargs = kwargs.get('ac_detector', {})
    relation_classifier_kwargs = kwargs.get('relation_classifier', {})

    if model_name == BertGRUCRF:

        model = model_name(bert_model_name=ac_detection_kgwargs.get('model_card', None), 
                           num_labels=ac_detection_kgwargs.get('num_labels', 5),
                           gru_hidden_size=ac_detection_kgwargs.get('gru_hidden_size', 128),
                           gru_num_layers=ac_detection_kgwargs.get('gru_num_layers', 1),
                           dropout_prob=ac_detection_kgwargs.get('dropout_prob', None))
    
    elif model_name == BERTSentClf:
        model = model_name(bert_model_name=relation_classifier_kwargs.get('model_card', None), 
                           num_labels=relation_classifier_kwargs.get('num_labels', 2),
                           dropout=relation_classifier_kwargs.get('dropout_prob', None))
    
    else:
        raise ValueError(f"Model {model_name} not supported.")

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def compute_norm_class_weights(
    df, 
    task_name, 
    label_column=None, 
    special_tokens=None
):
    """
    Computes normalized class weights for the specified task, handling special cases for token exclusion.

    Args:
        df (pandas.DataFrame): The DataFrame containing the dataset.
        task_name (str): Name of the task, options are:
            - 'arguments_component_classification'
            - 'relation_classification'
        label_column (str): Name of the column containing labels.
            - For 'arguments_component_classification': Default is 'encoded_tags'.
            - For 'relation_classification': Default is 'Label'.
        special_tokens (list, optional): List of token IDs to exclude (default: None, typical values: [101, 102, 0]).

    Returns:
        dict: A dictionary where keys are class indices and values are normalized weights.
    """
    from collections import Counter

    # Set default values for specific tasks
    if task_name == 'seqtag':
        if label_column is None:
            label_column = 'encoded_tags'
        if special_tokens is None:
            special_tokens = [101, 102, 0]  # [CLS], [SEP], [PAD]

        # Collect valid labels excluding special tokens
        all_labels = []
        for _, row in df.iterrows():
            labels = row[label_column]
            input_ids = row['input_ids']
            attention_mask = row['attention_mask']

            valid_labels = [
                label for label, token_id, mask in zip(labels, input_ids, attention_mask)
                if mask != 0 and token_id not in special_tokens
            ]
            all_labels.extend(valid_labels)
    elif task_name == 'rel_class':
        if label_column is None:
            label_column = 'Label'
        # Collect all labels directly from the column
        all_labels = df[label_column].tolist()
    else:
        raise ValueError(f"Unsupported task_name: {task_name}. Use 'arguments_component_classification' or 'relation_classification'.")

    # Calculate class frequencies
    label_counts = Counter(all_labels)
    total_labels = sum(label_counts.values())

    # Compute normalized class weights
    weights = {label: total_labels / count for label, count in label_counts.items()}
    sum_weights = sum(weights.values())
    normalized_weights = {label: weight / sum_weights for label, weight in weights.items()}

    return normalized_weights


##### FIN QUI CORRETTE ######

# Functions to display data AC Classification
def plot_value_counts(df, name):
    """
    Plots a histogram to visualize the occurrences of tags.

    Parameters:
        df (pandas DataFrame): The DataFrame containing the data.
        name (str): Name of the dataset.

    Returns:
        None
    """
    all_tags = [tag for tags in df['tags'] for tag in tags]

    tag_counts = Counter(all_tags)

    counts_df = pd.DataFrame.from_dict(tag_counts, orient='index', columns=['count']).reset_index()
    counts_df.columns = ['Tag', 'Count']

    counts_df = counts_df.sort_values('Count', ascending=False)

    fig = px.bar(counts_df, x='Tag', y='Count',
                 title=f'Tag Occurrences in {name} Dataset',
                 labels={'Count': 'Number of Occurrences'},
                 color='Count', color_continuous_scale='Viridis')

    fig.update_layout(xaxis_title='Tag',
                      yaxis_title='Number of Occurrences',
                      xaxis_tickangle=-45)

    fig.show()

## DISPLAY FUNCTIONS
def generate_random_color():
    """Generates a random color excluding white (#FFFFFF) and black (#000000)."""
    while True:
        # Random RGB values
        r = random.randint(70, 200)
        g = random.randint(70, 200)
        b = random.randint(70, 200)

        # Exclude black (r = g = b = 0) and white (r = g = b = 255)
        if (r, g, b) != (255, 255, 255) and (r, g, b) != (0, 0, 0):
            return "#{:02x}{:02x}{:02x}".format(r, g, b)

def print_colored(text, color):
    """Formats text with a color."""
    return f"<span style='color: {color};'>{text}</span>"

def assign_color_to_span(span_id, color_dict):
    """Assign a new color to each occurrence of a span."""
    if span_id not in color_dict:
        color_dict[span_id] = generate_random_color()
    return color_dict[span_id]


def print_annotated_example(df, predicted_labels=None, show_special_tokens=False, tokens=None, sample_idx=1):
    """
    Prints examples of annotated sentences with different colors for components.

    Parameters:
    df (pandas DataFrame): The DataFrame containing the data.
    tokens (list): List of tokens
    predicted_labels (list, optional): Predicted labels
    show_special_tokens (bool): Whether to show special tokens like [CLS], [SEP], [PAD]
    sample_idx (int): Index of the sample to show
    """
    print(f"\nExample of an annotated sentence:")

    tokens_df = df.iloc[sample_idx]['tokenized_text']

    if tokens is not None:
        tokens = tokens
    else:
        tokens = tokens_df

    tags = df.iloc[sample_idx]['tags']

    if not show_special_tokens:
        special_tokens = ['[CLS]', '[SEP]', '[PAD]']
        # Filter original tokens and tags by removing special tokens
        filtered_pairs = [(t, tag) for t, tag in zip(tokens, tags) if t not in special_tokens]
        tokens, tags = zip(*filtered_pairs) if filtered_pairs else ([], [])

        # Filter prediction tokens and labels if they exist
        if predicted_labels is not None:
            filtered_pred_pairs = [(t, label) for t, label in zip(tokens, predicted_labels) if t not in special_tokens]
            tokens, predicted_labels = zip(*filtered_pred_pairs) if filtered_pred_pairs else ([], [])

    print("\nRaw text:")
    print(df.iloc[sample_idx]['raw_text'].split())

    print("\nTokenized text with colored components:")

    # Dictionary to store colors for each span
    color_dict = {}
    span_id = None  # To track the current span (e.g., B-Claim, B-Premise)
    annotated = []

    for idx, (token, label) in enumerate(zip(tokens, tags)):
        if label in ['B-Claim', 'B-Premise']:
            span_id = label + str(idx)  # Unique span ID per component
            color = assign_color_to_span(span_id, color_dict)  # Assign a color to the span
        elif label in ['I-Claim', 'I-Premise'] and span_id:
            # Continue coloring with the same color for I-Claim or I-Premise part of the same span
            color = color_dict.get(span_id, "black")
        else:
            color = None

        annotated.append(print_colored(f"{token}({label})", color))

    display(HTML(' '.join(annotated)))

    if predicted_labels is not None:
        # Repeat the process for predicted labels
        color_dict_pred = {}
        span_id = None
        pred_annotated = []

        # Teniamo traccia dei token per ogni span
        current_span_tokens = []

        for idx, (token, label) in enumerate(zip(tokens, predicted_labels)):
            # Se troviamo l'inizio di un nuovo span
            if label in ['B-Claim', 'B-Premise']:
                # Se c'era uno span precedente, processiamolo
                if current_span_tokens:
                    # Decidiamo se colorare lo span precedente in base alla sua lunghezza
                    should_color = len(current_span_tokens) >= 5
                    color = color_dict_pred.get(span_id, "black") if should_color else None
                    for t, l in current_span_tokens:
                        pred_annotated.append(print_colored(f"{t}({l})", color))
                    current_span_tokens = []

                # Iniziamo un nuovo span
                span_id = label + str(idx)
                color_dict_pred[span_id] = assign_color_to_span(span_id, {})
                current_span_tokens.append((token, label))

            elif label in ['I-Claim', 'I-Premise'] and span_id:
                # Continuiamo lo span corrente
                current_span_tokens.append((token, label))
            else:
                # Se c'era uno span precedente, processiamolo
                if current_span_tokens:
                    should_color = len(current_span_tokens) >= 5
                    color = color_dict_pred.get(span_id, "black") if should_color else None
                    for t, l in current_span_tokens:
                        pred_annotated.append(print_colored(f"{t}({l})", color))
                    current_span_tokens = []
                    span_id = None

                # Token con label 'O'
                pred_annotated.append(print_colored(f"{token}({label})", None))

        # Processiamo l'ultimo span se presente
        if current_span_tokens:
            should_color = len(current_span_tokens) >= 5
            color = color_dict_pred.get(span_id, "black") if should_color else None
            for t, l in current_span_tokens:
                pred_annotated.append(print_colored(f"{t}({l})", color))

        print("\nTokenized text with predicted tags:")
        display(HTML(' '.join(pred_annotated)))

##
##
##
##
# Function to display data Relation Classification
def plot_value_counts(df, name):
    """
    Plots a histogram to visualize the occurrences of tags and displays the number of unique labels.

    Parameters:
        df (pandas DataFrame): The DataFrame containing the data.
        name (str): Name of the dataset.

    Returns:
        None
    """
    all_labels = [lab for labels in df['Label'] for lab in labels]

    label_counts = Counter(all_labels)

    counts_df = pd.DataFrame.from_dict(label_counts, orient='index', columns=['count']).reset_index()
    counts_df.columns = ['Label', 'Count']

    counts_df = counts_df.sort_values('Count', ascending=False)

    num_unique_labels = len(counts_df)

    fig = px.bar(counts_df, x='Label', y='Count',
                 title=f'Label Occurrences in {name} Dataset (Total Unique Labels: {num_unique_labels})',
                 labels={'Count': 'Number of Occurrences'},
                 color='Count', color_continuous_scale='Viridis')

    fig.update_layout(xaxis_title='Label',
                      yaxis_title='Number of Occurrences',
                      xaxis_tickangle=-45)

    # Aggiungiamo un'annotazione per il numero di label uniche
    fig.add_annotation(
        x=0.5,
        y=1.05,
        xref="paper",
        yref="paper",
        text=f"Total Unique Labels: {num_unique_labels}",
        showarrow=False,
        font=dict(size=14),
        align="center",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )

    fig.show()


# Functions to display a torch.utils.data.Dataset item
def display_dataset_item(item, max_seq_length=50, max_line_width=80, idx_to_label=None):
    """
    Displays a single item from a PyTorch dataset in a formatted and readable manner.

    Parameters:
    item (dict or tuple): An item from the PyTorch dataset.
    max_seq_length (int): Maximum number of tokens to display for sequences.
    max_line_width (int): Maximum width for wrapped text lines.

    Returns:
    None (prints the formatted item)
    """
    if isinstance(item, dict):
        for key, value in item.items():
            print(f"\n{key.upper()}:")
            if key == 'label':
                _display_label(value, idx_to_label)
            else:
                _display_tensor_or_list(value, max_seq_length, max_line_width)
    elif isinstance(item, tuple):
        for i, value in enumerate(item):
            print(f"\nITEM {i}:")
            _display_tensor_or_list(value, max_seq_length, max_line_width)
    else:
        print("Unsupported item type")

def _display_tensor_or_list(value, max_seq_length, max_line_width):
    if isinstance(value, torch.Tensor):
        if value.dim() == 1:
            _display_sequence(value.tolist(), max_seq_length, max_line_width)
        elif value.dim() == 2:
            _display_2d_tensor(value, max_seq_length)
        else:
            print(f"Tensor shape: {value.shape}")
            print(value)
    elif isinstance(value, list):
        _display_sequence(value, max_seq_length, max_line_width)
    else:
        print(textwrap.fill(str(value), width=max_line_width))

def _display_sequence(seq, max_seq_length, max_line_width):
    if len(seq) > max_seq_length:
        displayed_seq = seq[:max_seq_length] + ['...']
    else:
        displayed_seq = seq
    print(textwrap.fill(str(displayed_seq), width=max_line_width))
    print(f"\nLength: {len(seq)}")

def _display_2d_tensor(tensor, max_seq_length):
    if tensor.shape[0] > max_seq_length:
        displayed_tensor = tensor[:max_seq_length].tolist() + [['...'] * tensor.shape[1]]
    else:
        displayed_tensor = tensor.tolist()
    print(tabulate(displayed_tensor, tablefmt="grid"))
    print(f"Shape: {tensor.shape}")

def _display_label(label_tensor, id_2_label):
    label_id = label_tensor.item()
    label_name = id_2_label.get(label_id, "Unknown")
    print(label_name)

def tokenize_and_encode(sentence, tokenizer, max_length):
    """
    Tokenizes and encodes a sentence using the provided tokenizer.

    Args:
        sentence (str): The input sentence to tokenize and encode.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        max_length (int): The maximum length of the tokenized sequence.

    Returns:
        dict: A dictionary containing the original sentence, tokens, and tokens with their IDs.
    """
    tokens = tokenizer.tokenize(sentence)
    token_ids = tokenizer.encode(sentence, add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True)
    tokens_with_ids = [(token, token_id) for token, token_id in zip(tokenizer.convert_ids_to_tokens(token_ids), token_ids)]

    print(f"Original sentence: \n{sentence}")
    print(f"\nTokens: \n{tokens}")
    print(f"\nTokens and ID: \n{tokens_with_ids}")

def display_relations_table(relations_dict):
    """
    Displays relations in tabular format from the input dictionary using tabulate.

    Parameters:
        relations_dict (dict): Dictionary with keys as tuples representing pairs of components and values as relations.
    """
    rows = []

    for (comp1, comp2), relation in relations_dict.items():
        # Extracting the IDs, types, and texts of components
        comp1_id = comp1[-1]  # Unique ID
        comp1_type = comp1[0]
        comp1_text = comp1[1]
        
        comp2_id = comp2[-1]  # Unique ID 
        comp2_type = comp2[0]
        comp2_text = comp2[1]
        
        # Append a row for each relation if it exists
        if relation[0] == "Relation":
            rows.append([comp1_id, comp1_type, comp2_id, comp2_type, relation[0]])
        else:
            pass

    # Define headers for the table
    headers = ['Component 1 ID', 'Component 1 Type', 'Component 2 ID','Component 2 Type', 'Relation']

    # Print the table using tabulate
    print(tabulate(rows, headers=headers, tablefmt="grid"))



def show_components(tokens, predicted_components, real_components, tokenizer):
    """
    Displays both predicted and real components, decoding the tokenization.

    Parameters:
        tokens (list): List of input tokens
        predicted_components (list): List of tuples (component_type, component_text, start_pos, end_pos, component_id)
        real_components (list): List of tuples (component_type, component_text, start_pos, end_pos, component_id)
        tokenizer: The tokenizer used for encoding/decoding
    """
    print("Real Components:")
    for component in real_components:
        component_type, _, start_pos, end_pos, comp_id = component
        span_tokens = tokens[start_pos:end_pos+1]  # +1 perché end_pos è inclusivo
        token_ids = tokenizer.convert_tokens_to_ids(span_tokens)
        decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"{comp_id}. Type: {component_type}, Text: {decoded_text}")
    
    print("\nPredicted Components:")
    for component in predicted_components:
        component_type, _, start_pos, end_pos, comp_id = component
        span_tokens = tokens[start_pos:end_pos+1]  # +1 perché end_pos è inclusivo
        token_ids = tokenizer.convert_tokens_to_ids(span_tokens)
        decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        print(f"{comp_id}. Type: {component_type}, Text: {decoded_text}")