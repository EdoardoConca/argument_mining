import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel
from torch.utils.data import Dataset

class ArgumentationDataset(Dataset):
    """
    A PyTorch Dataset class for argumentation detection.

    Args:
        dataframe (DataFrame): The DataFrame containing the input data.
        tokenizer (AutoTokenizer): The pre-trained tokenizer to use.    
    """
    def __init__(self, dataframe, tokenizer):
        self.tokenizer = tokenizer
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        input_ids = row['input_ids']
        attention_mask = row['attention_mask']
        labels = row['encoded_tags']

        # set [CLS] and [SEP] attention to 0
        for i, token_id in enumerate(input_ids):
            if token_id == self.tokenizer.cls_token_id or token_id == self.tokenizer.sep_token_id:
                attention_mask[i] = 0

        # Convert lists to tensors
        input_ids_tensor = torch.tensor(input_ids).long()
        attention_mask_tensor = torch.tensor(attention_mask).long()
        label_tensor = torch.tensor(labels).long()

        return {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'labels': label_tensor
        }


class BertGRUCRF(nn.Module):
    """
    A neural network model that combines BERT, GRU, and CRF layers for sequence labeling tasks.

    Args:
        bert_model_name (str): The name of the pre-trained BERT model to use.
        num_labels (int): The number of labels for the classification task.
        gru_hidden_size (int, optional): The number of hidden units in the GRU layer. Default is 128.
        gru_num_layers (int, optional): The number of layers in the GRU. Default is 1.
        dropout_prob (float, optional): The dropout probability. Default is None.

    Attributes:
        bert (nn.Module): The BERT model.
        gru (nn.GRU): The GRU layer.
        fc (nn.Linear): The fully connected layer.
        dropout (nn.Dropout): The dropout layer.
        crf (CRF): The CRF layer.
    """

    def __init__(self, bert_model_name, num_labels, gru_hidden_size=128, gru_num_layers=1, dropout_prob=None):
        super(BertGRUCRF, self).__init__()
        self.bert_model_name = bert_model_name
        self.num_labels = num_labels
        self.dropout_prob = dropout_prob
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        # BERT backbone
        self.bert = AutoModel.from_pretrained(self.bert_model_name)

        # GRU layer
        self.gru = nn.GRU(input_size=self.bert.config.hidden_size,
                          hidden_size=self.gru_hidden_size,
                          num_layers=self.gru_num_layers,
                          batch_first=True,
                          bidirectional=True)
        # Linear layer
        self.fc = nn.Linear(self.gru_hidden_size * 2, self.num_labels)  # *2 since GRU is bidirectional

        # Dropout layer
        if self.dropout_prob is not None:
            self.dropout = nn.Dropout(dropout_prob)

        # CRF layer
        self.crf = CRF(num_labels,batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, class_weights=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor): The attention mask.
            labels (torch.Tensor, optional): The true labels for the input. Default is None.
            class_weights (torch.Tensor, optional): The class weights for the loss calculation. Default is None.

        Returns:
            loss (torch.Tensor): The loss value if labels are provided, otherwise None.
            output (torch.Tensor): The decoded output from the CRF layer.
        """
        embed = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = embed.last_hidden_state

        gru_output, _ = self.gru(sequence_output) # (batch_size, seq_length, hidden_size * 2)

        emissions = self.fc(gru_output)

        if self.dropout_prob is not None:
            emissions = self.dropout(emissions)


        crf_mask = attention_mask.clone()
        crf_mask[:, 0] = 1      # Includes CLS for the CRF (CRF accepts mask starting with 0s or 1s, in this case 1s)
        crf_mask = crf_mask.bool()

        loss_mask = attention_mask.clone()
        loss_mask = loss_mask.bool()

        loss = None
        if labels is not None:
            tag_seq = self.crf.decode(emissions, crf_mask)
            loss = -1 * self.crf(emissions, labels, mask=crf_mask)
            if class_weights is not None:
                weights = class_weights[labels]
                loss = loss * weights


            loss = loss * loss_mask.float()
            loss = loss.mean()

            return loss, tag_seq
        else:
            tag_seq = self.crf.decode(emissions, crf_mask)
            return loss, tag_seq
        
    

