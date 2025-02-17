import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Dict, Union
from itertools import permutations
from utils.utils_notebooks import convert_ids_to_tags


def print_misclassified_examples(model, dataloader, idx_to_label, max_display):
    """
    Prints misclassified examples from the dataset.

    Parameters:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): The DataLoader for the dataset.
        label_to_idx (dict): Mapping from label names to indices.
        idx_to_label (dict): Mapping from indices to label names.
        tokenizer (AutoTokenizer): The tokenizer used for text processing.
        max_display (int): The maximum number of misclassified examples to display. Default is 5.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    misclassified = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            pred_labels = torch.argmax(logits, dim=1)

            # Identifying misclassified examples
            for i, (pred, true) in enumerate(zip(pred_labels, labels)):
                if pred != true:
                    arg1_text, arg2_text = dataloader.dataset.get_text_pair(batch_idx * dataloader.batch_size + i)
                    arg1_type, arg2_type = dataloader.dataset.get_type_pair(batch_idx * dataloader.batch_size + i)
                    misclassified.append((arg1_text, arg2_text, arg1_type, arg2_type, idx_to_label[true.item()], idx_to_label[pred.item()]))



    # Print the misclassified examples
    print(f"\nTotal misclassified examples: {len(misclassified)} out of {len(dataloader.dataset)}")
    print("\nMisclassified Examples:")
    for count, sample in enumerate(misclassified):
        arg1_text, arg2_text, arg1_type, arg2_type, true_label, pred_label = sample
        if count <= max_display:
          print(f"\nArg1_Text: {arg1_text} \t(Type: {arg1_type})")
          print(f"Arg2_Text: {arg2_text} \t(Type: {arg2_type})")
          print(f"True Label: {true_label}")
          print(f"Predicted Label: {pred_label}\n")
        else:
          break


def get_components(tokens: List[str], tags: List[str], start_end_positions: Union[Dict[str, Tuple[int, int]], List[Tuple[int, int]]], is_predicted: bool = False, discarding_factor: Dict = None) -> List[Tuple[str, str, int, int, str]]:
    """
    Extracts argument components from tokens, tags, and their start-end positions, with unique identifiers.

    Args:
        tokens: List of tokens.
        tags: List of BIO tags.
        start_end_positions: Either a dictionary {'T1': (start, end), ...} for real components
                           or a list of tuples [(start, end), ...] for predicted components.
        is_predicted: Boolean flag to indicate if we're handling predicted components.

    Returns:
        List of tuples containing (component_type, component_text, start_position, end_position, component_id).
    """
    components = []
    if discarding_factor is not None:
        discarding_factor_premise = discarding_factor['discarding_factor_premise']
        discarding_factor_claim = discarding_factor['discarding_factor_claim']
    else:
        discarding_factor_premise = (None, None)
        discarding_factor_claim = (None, None)
    
    # converts start_end_positions into a list of tuples (start, end, id)
    positions_list = []
    if isinstance(start_end_positions, dict) and not is_predicted:
        # for real components
        positions_list = [(start, end, comp_id) for comp_id, (start, end) in start_end_positions.items()]
    else:
        # for predicted components
        positions_list = [(start, end, f"T{i+1}") for i, (start, end) in enumerate(start_end_positions)]
    
    for start_pos, end_pos, comp_id in positions_list:
        current_component = []
        current_tag = None
        
        # collects tokens for the current component
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            if start_pos <= i <= end_pos:
                if not current_component:
                    # tag of the first component's token (B-*)
                    if tag.startswith('B-'):
                        current_tag = tag.split('-')[1]
                    elif tag.startswith('I-'):
                        current_tag = tag.split('-')[1]
                current_component.append(token)
        
        if current_component and current_tag:
            component_text = ' '.join(current_component)
            if discarding_factor is not None:
                # if the component length is less than or equal to the discarding factor, discard it
                if is_predicted and (len(current_component) <= discarding_factor_premise[0] or len(current_component) > discarding_factor_premise[1])  and current_tag == 'Premise':
                    continue
                elif is_predicted and (len(current_component) <= discarding_factor_claim[0] or len(current_component) > discarding_factor_claim[1]) and current_tag == 'Claim':
                    continue
                
            components.append((
                current_tag,
                component_text,
                start_pos,
                end_pos,
                comp_id
            ))
    
    return components

def prepare_inputs_for_relation_classification(tokenizer: AutoTokenizer,
                                            component1: str,
                                            component2: str,
                                            device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepares inputs for relation classification.

    Args:
        tokenizer: Tokenizer to use
        component1: First argument component
        component2: Second argument component
        device: Device to run on

    Returns:
        Tuple of input_ids and attention_mask tensors
    """
    inputs = tokenizer(
        component1, component2,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    token_type_ids = inputs.get("token_type_ids", None)
    return input_ids, attention_mask, token_type_ids

def classify_relations(relation_model, components, tokenizer_rel, idx_to_label, device):
        """
        Classify relations between argument components.

        Args:
            relation_model: The relation classification model
            components: List of tuples containing (component_type, component_text, start_position, end_position)
            tokenizer_rel: The tokenizer for relation classification
            idx_to_label: Mapping from label indices to label names
            device: Device to run on.

        Returns:
            List of predicted relation labels
        """
        relations = {}
        component_pairs = list(permutations(components, 2))

        for comp1, comp2 in component_pairs:
            input_ids, attention_mask, token_type = prepare_inputs_for_relation_classification(
                tokenizer_rel, comp1[1], comp2[1], device
            )

            relation_model.eval()
            with torch.no_grad():
                logits = relation_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type)
                predicted_label = torch.argmax(logits, dim=-1).item()
                predicted_label = convert_ids_to_tags([predicted_label], idx_to_label)

            relations[(comp1, comp2)] = predicted_label

        return relations