from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import torch.nn as nn

grads = 0
def grad_hook(module, grad_input, grad_output):
    global grads
    grads = grad_output[0].clone().detach()

name = 'facebook/bart-base'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name,num_labels=2)
if 'bart' in name:
    model.model.shared.register_backward_hook(grad_hook)
else:
    model.roberta.embeddings.word_embeddings.register_backward_hook(grad_hook)

x = ['hello,you']
x_t = tokenizer(x)
logits = model(input_ids=torch.tensor(x_t['input_ids'])).logits
loss_fct = nn.CrossEntropyLoss()
loss = loss_fct(logits, torch.tensor([1]))
loss.backward()
print(grads.abs().sum())