import torch
import torch.nn as nn
from transformers import RobertaModel






class RobertaClassifierWithExtra(nn.Module):  #need nn.Module as parent class

    def __init__(self, num_classes=2, extra_dim=3):
        super().__init__()

        #--- roberta base model ---
        self.roberta = RobertaModel.from_pretrained('roberta-base')

        #--- extra layer specification ---
        self.hidden = nn.Linear(768 + extra_dim, 256) #  776 -> 256
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        #--- classifier specification --- 
        self.classifier = nn.Linear(256, num_classes) # 256 -> 2



        self.num_classes = num_classes  #need this later on..

        

    
    def forward(self, input_ids, attention_mask, extra_features, labels=None):

        #--- Run tokens through roberta ---
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token, shape (batch_size, 768)


        #--- concacentate features from the search engine ---

        x = torch.cat([cls_output, extra_features], dim=1)  # shape (batch_size, 768 + extra_dim)


        #--- extra layers to handle new input ---
        x = self.hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        #--- compute loss when training(when we provide labels) ---
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return logits, loss






