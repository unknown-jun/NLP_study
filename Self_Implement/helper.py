import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split
from transformers import BertTokenizer

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"# available GPUs : {torch.cuda.device_count()}")
        print(f"GPU name : {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
    return device

class CustomDataset(Dataset):
    
  def __init__(self, input_data:list, target_data:list) -> None:
      self.X = input_data
      self.Y = target_data

  def __len__(self):
      return len(self.X)

  def __getitem__(self, index):
      return self.X[index], self.Y[index]
  
def custom_collate_fn(batch):

  tokenizer_bert= BertTokenizer.from_pretrained("klue/bert-base")
  
  input_list, target_list = [ ], [ ]

  for _input, _label in batch:
    input_list.append(_input)
    target_list.append(_label)
  
  tensorized_input = tokenizer_bert(input_list,
                                    add_special_tokens=True,
                                    truncation=True,
                                    padding='longest',
                                    max_length=512,
                                    return_tensors='pt')
  
  tensorized_label = torch.tensor(target_list)
  return tensorized_input, tensorized_label

class CustomClassifier(nn.Module):

  def __init__(self, hidden_size: int, n_label: int):
    super(CustomClassifier, self).__init__()
    self.bert = BertModel.from_pretrained('klue/bert-base')
    self.hidden_size = hidden_size
    self.n_label = n_label

    dropout_rate = 0.1
    linear_layer_hidden_size = 32

    self.classifier=nn.Sequential(nn.Linear(self.hidden_size, linear_layer_hidden_size),
                                  nn.ReLU(),
                                  nn.Dropout(dropout_rate),
                                  nn.Linear(linear_layer_hidden_size, n_label))

  def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):

    output = self.bert(
        input_ids,
        attention_mask = attention_mask,
        token_type_ids = token_type_ids
    )

    cls_token_last_hidden_states = output[0][:,0,:]
    logits = self.classifier(cls_token_last_hidden_states)

    return logits
