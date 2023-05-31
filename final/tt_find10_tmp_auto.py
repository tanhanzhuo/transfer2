from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import torch.nn as nn

grads = 0
def grad_hook(module, grad_input, grad_output):
    global grads
    grads = grad_output[0].clone().detach()

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-base',num_labels=2)
model.model.shared.register_backward_hook(grad_hook)

x = ['hello,you','hwojea no you']
x_t = tokenizer(x)
logits = model(**x_t).logits
loss_fct = nn.CrossEntropyLoss()
loss = loss_fct(logits, torch.tensor([1,0]))
loss.backward()
print(grads.abs().sum())