from sklearn.metrics import f1_score,accuracy_score
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils_notebooks import set_reproducibility
from transformers import get_linear_schedule_with_warmup, AdamW
from utils.metrics import acc_and_f1, f1_scores
import os
from utils.utils_notebooks import print_annotated_example, convert_ids_to_tags


def train(model, model_name, train_dataloader, val_dataloader, learning_rate, num_epochs, scheduler=False, task_name='seqtag',class_weights=None, save_model=False, models_folder=None, seeds=[42],**kwargs):
    """
    Trains the model on the training set and evaluates it on the validation set.

    Parameters:
        model (nn.Module): The neural network model.
        train_dataloader (DataLoader): The DataLoader for the training set.
        val_dataloader (DataLoader): The DataLoader for the validation set.
        learning_rate (float): The learning rate for the optimizer.
        num_epochs (int): The number of epochs for training.
        scheduler (bool): Whether to use a linear scheduler. Default is False.
        task_name (str): The name of the task. Must be 'seqtag' or 'rel_class'.
        class_weights (dict, optional): The class weights for the loss function.
        save_model (bool): Whether to save the model. Default is False.
        models_folder (str): The folder where to save the model.
        seeds (list): List of random seeds for reproducibility.
        kwargs: Additional keyword arguments.

    Returns:
        tuple: The trained model and the results.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if class_weights and task_name == 'rel_class':
        class_weights_tensor = torch.tensor(list(class_weights.values()),dtype=torch.float32, device=device)
        loss_function = nn.CrossEntropyLoss(weight=class_weights_tensor)
    elif class_weights is None and task_name == 'rel_class':
        class_weights_tensor = None
        loss_function = nn.CrossEntropyLoss()
    elif class_weights and task_name == 'seqtag':
        class_weights_tensor = torch.tensor(list(class_weights.values()),dtype=torch.float32, device=device)
    elif class_weights is None and task_name == 'seqtag':
        class_weights_tensor = None
    else:
        raise ValueError("task_name must be 'seqtag' or 'rel_class'")

    best_val_f1 = 0.0

    eps = kwargs.get('eps', 1e-8)
    weight_decay = kwargs.get('weight_decay', 0.0)
    num_warmup_steps = kwargs.get('num_warmup_steps', 0)

    for seed in seeds:
        print(f"\nTraining with seed {seed}...")
        set_reproducibility(seed)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=eps, weight_decay=weight_decay)
        
        total_steps = len(train_dataloader) * num_epochs
        scheduler =get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps) if scheduler else None

        model.to(device)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            model.train()
            
            train_loss = 0.0
            pred_list, true_list = [], []
            train_iterator = tqdm(train_dataloader, desc="Training", total=len(train_dataloader), leave=False)
            
            for batch in train_iterator:
                if task_name == 'seqtag':
                    input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                    
                    loss, tag_seq = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, class_weights=class_weights_tensor)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if scheduler:
                        scheduler.step()

                    train_loss += loss.item()
                    for pred_tag, true_tag in zip(tag_seq, labels):
                        pred_list.extend(pred_tag)
                        true_list.extend(true_tag.tolist()[:len(pred_tag)])
                elif task_name == 'rel_class':
                    input_ids, attention_mask, token_type_ids, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['token_type_ids'].to(device), batch['label'].to(device)
                    
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
                    loss = loss_function(logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if scheduler:
                        scheduler.step()

                    train_loss += loss.item()
                    pred_list.extend(torch.argmax(logits, dim=1).tolist())
                    true_list.extend(labels.tolist())
                else:
                    raise ValueError("task_name must be 'seqtag' or 'rel_class'")

            avg_train_loss = train_loss / len(train_dataloader)
            print(f"Training\n\ttrain_loss: {avg_train_loss:.4f}")

            # compute the metrics
            if task_name == 'seqtag':
                metrics = acc_and_f1(pred_list, true_list)
            elif task_name == 'rel_class':
                metrics = f1_scores(pred_list, true_list)
            else:
                raise ValueError("task_name must be 'seqtag' or 'rel_class'")

            print(f"\tF1-Score Micro: {metrics['eval_f1_micro']:.4f} | F1-Score Macro: {metrics['eval_f1_macro']:.4f} | Weighted F1-Score: {metrics['f1']:4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | Accuracy: {metrics['acc']:.4f}")

            # VALIDATION PHASE
            if task_name == 'seqtag':
                avg_val_loss, val_metrics = evaluate(model, val_dataloader, device, task_name)
            elif task_name == 'rel_class':
                avg_val_loss, val_metrics = evaluate(model, val_dataloader, device, task_name, loss_function=loss_function)

            if avg_val_loss is not None:
                print(f"Validation\n\tval_loss: {avg_val_loss:.4f}")
                print(f"\tF1-Score Micro: {val_metrics['eval_f1_micro']:.4f} | F1-Score Macro: {val_metrics['eval_f1_macro']:.4f} | Weighted F1-Score: {val_metrics['f1']:.4f} | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | Accuracy: {val_metrics['acc']:.4f}")
            else:
                print("avg_loss None")

            if val_metrics['eval_f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['eval_f1_macro']
                best_epoch = epoch

            print("*" * 100)

        print(f"\nBest Val F1-Score: {best_val_f1:.4f} at epoch {best_epoch + 1}")

        if save_model is True and models_folder is not None:
            try:
                print("Saving model...")
                os.makedirs(os.path.dirname(f'{models_folder}/checkpoint/{model_name}/bi_gru-crf_seed_{seeds[0]}.pt'), exist_ok=True)
                torch.save(model.state_dict(), f'{models_folder}/checkpoint/{model_name}/bi_gru-crf_seed_{seeds[0]}.pt')
                print("Saved!")
            except Exception as e:
                print(f"Error while saving model: {e}")


    print("\nTraining completed.")

    results = {'Training': metrics,
               'Validation': val_metrics}

    return model, results


def evaluate(model, dataloader, device, task_name, loss_function=None, target_names=None, verbose=False, name=None):
    """
    Evaluates the model on the test/validation set.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): The DataLoader for the test/validation set.
        device (str): The device to run the model on (cpu or cuda).
        task_name (str): The name of the task. Must be 'seqtag' or 'rel_class'.
        loss_function (nn.Module, optional): The loss function. Default is None.
        target_names (list, optional): The target names for the classification report. Default is None.
        verbose (bool, optional): If True, print the results. Default is False.
        name (str, optional): The name of the dataset. Default is None.

    Returns:
        tuple: The average loss and the evaluation metrics
    """

    model.eval()
    model.to(device)
    pred_list, true_list = [], []
    val_loss = 0.0
    avg_val_loss = 0.0

    if loss_function is not None:
        loss_function.to(device)
        calculate_loss = True
    else:
        calculate_loss = False

    test_iterator = tqdm(dataloader, desc="Evaluating", total=len(dataloader), leave=False)
    with torch.no_grad():
        for batch in test_iterator:
            if task_name == 'seqtag':
                input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
                
                loss, tag_seq = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += loss.item()

                for pred_tag, true_tag in zip(tag_seq, labels):
                    pred_list.extend(pred_tag)
                    true_list.extend(true_tag.tolist()[:len(pred_tag)])
                    
            elif task_name == 'rel_class':
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                if calculate_loss:
                    loss = loss_function(logits, labels)
                    val_loss += loss.item()

                pred_labels = torch.argmax(logits, dim=1).tolist()
                true_labels = labels.tolist()

                pred_list.extend(pred_labels)
                true_list.extend(true_labels)


    avg_val_loss = val_loss / len(dataloader) if calculate_loss else None


    # Compute the metrics
    if task_name == 'seqtag':
        val_metrics = acc_and_f1(pred_list, true_list)
    elif task_name == 'rel_class':
        val_metrics = f1_scores(pred_list, true_list)
    else:
        raise ValueError("task_name must be 'seqtag' or 'rel_class'")

    if verbose:
        if name is not None:
          print(f"\n{name} Test Results")
          if calculate_loss:
            print(f"\tAverage loss: {avg_val_loss:.4f}")
          print(f"\tF1-Score Micro: {val_metrics['eval_f1_micro']:.4f}")
          print(f"\tF1-Score Macro: {val_metrics['eval_f1_macro']:.4f}")
          print(f"\tWeighted F1-Score: {val_metrics['eval_f1_weighted']:.4f}")
          print(f"\tPrecision: {val_metrics['precision']:.4f}")
          print(f"\tRecall: {val_metrics['recall']:.4f}")
          print(f"\tAccuracy: {val_metrics['acc']:.4f}")
          if task_name == 'seqtag':
            print(f"\tF1 Claim: {val_metrics['f1_claim']:.4f}")
            print(f"\tF1 Evidence {val_metrics['f1_evidence']:.4f}")
          print(f"\tClassification Report:\n{val_metrics['clf_report']}")
        else:
          print(f"\nTest Results")
          if calculate_loss:
            print(f"\tAverage loss: {avg_val_loss:.4f}")
          print(f"\tF1-Score Micro: {val_metrics['eval_f1_micro']:.4f}")
          print(f"\tF1-Score Macro: {val_metrics['eval_f1_macro']:.4f}")
          print(f"\tWeighted F1-Score: {val_metrics['eval_f1_weighted']:.4f}")
          print(f"\tPrecision: {val_metrics['precision']:.4f}")
          print(f"\tRecall: {val_metrics['recall']:.4f}")
          print(f"\tAccuracy: {val_metrics['acc']:.4f}")
          if task_name == 'seqtag':
            print(f"\tF1 Claim: {val_metrics['f1_claim']:.4f}")
            print(f"\tF1 Evidence {val_metrics['f1_evidence']:.4f}")
          print(f"\tClassification Report:\n{val_metrics['clf_report']}")

    return avg_val_loss, val_metrics

def predict(model, df, dataset, sample_idx, idx_to_tag, device='cpu', verbose=True):
    """
    Predicts the labels for a given sample.

    Args:
        model (nn.Module): The trained model.
        df (pd.DataFrame): The DataFrame containing the text data.
        dataset (Dataset): The dataset.
        sample_idx (int): The index of the sample to predict.
        idx_to_tag (dict): The mapping from tag indices to tags.
        device (str): The device to run the model on (cpu or cuda). Default is 'cpu'.
        verbose (bool): If True, print the annotated example. Default is True.

    Returns:
        tuple: The predicted labels and tags.
    """

    model.eval()
    model.to(device)

    data = dataset[sample_idx]
    input_ids = data['input_ids'].unsqueeze(0).to(device)
    attention_mask = data['attention_mask'].unsqueeze(0).to(device)
    labels = data['labels']


    with torch.no_grad():
        _, predicted_tags = model(input_ids=input_ids, attention_mask=attention_mask)

    predicted_labels = convert_ids_to_tags(predicted_tags[0], idx_to_tag)

    if verbose:
        print_annotated_example(df, predicted_labels=predicted_labels,show_special_tokens=False, sample_idx=sample_idx)
    
    return predicted_labels, predicted_tags
