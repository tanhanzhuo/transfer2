import datasets
import random
data = datasets.load_from_disk('./data_extension/TrainData_line')
data_train = data['train']
print(len(data_train))
print(data_train[random.randint(0,len(data_train))])
print(data_train[random.randint(0,len(data_train))])
print(data_train[random.randint(0,len(data_train))])