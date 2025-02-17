import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel
from torch.utils.data import Dataset

class RelationDataset(Dataset):
    """
    Custom dataset for relation classification.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the text pairs and labels.
        tokenizer (AutoTokenizer): The tokenizer used to preprocess the text.
        label_to_idx (dict): Mapping from labels to indices.
        max_length (int): The maximum sequence length for tokenization.

    """
    def __init__(self, dataframe, tokenizer, label_to_idx, max_length=128):

        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        text1 = row['Arg1_Text']
        text2 = row['Arg2_Text']

        inputs = self.tokenizer(text1, text2,
                                add_special_tokens=True,
                                truncation=True,
                                padding='max_length',
                                max_length=self.max_length,
                                return_tensors="pt")

        label = self.label_to_idx[row['Label']]
        label = torch.tensor(label, dtype=torch.long)

        token_type_ids = inputs['token_type_ids'].squeeze(0) if 'token_type_ids' in inputs else torch.zeros_like(inputs['input_ids']).squeeze(0)

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'token_type_ids': token_type_ids,
            'label': label
        }

    def get_text_pair(self, idx):
        row = self.dataframe.iloc[idx]
        return row['Arg1_Text'], row['Arg2_Text']

    def get_type_pair(self, idx):
        row = self.dataframe.iloc[idx]
        return row['Arg1_Type'], row['Arg2_Type']

class BERTSentClf(nn.Module):
    """
    SentClf model using BERT variant for relation classification.

    Args:
        bert_model_name (str): The pre-trained SciBERT model name.
        num_labels (int): Number of classification labels.
        dropout (float): Dropout rate for the classifier.
    """
    def __init__(self, bert_model_name, num_labels, dropout):
        super(BERTSentClf, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor): The attention mask.
            arg1_type (torch.Tensor, optional): The type of the first argument.
            arg2_type (torch.Tensor, optional): The type of the second argument.

        Returns:
            logits (torch.Tensor): The logits for each class.
        """
        if token_type_ids is not None:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = bert_outputs.pooler_output

        # Apply dropout to pooled output
        pooled_output = self.dropout(pooled_output)

        # Pass through the classification layer
        logits = self.classifier(pooled_output)

        return logits