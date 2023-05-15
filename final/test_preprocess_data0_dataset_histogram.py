import matplotlib.pyplot as plt
import datasets
for task in 'eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm'.split(','):
    length_all = []
    data_all = datasets.load_from_disk('../finetune/data/' + task + '/token')
    for split in ['train','dev','test']:
        for sample in data_all[split]:
            length_all.append(len(sample['input_ids']))
    plt.figure()
    plt.hist(length_all, bins=10)
    plt.show()