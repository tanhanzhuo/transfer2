import json
tasks = 'eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm'.split(',')
for task in tasks:
    for sp in ['train','dev','test']:
        with open('../finetune/data/'+task+'/'+sp+'_same_500_one20_top100_sp.json','r') as f:
            data_one20 = []
            for line in f:
                tmp = json.loads(line)
                for idx in range(3,100):
                    tmp.pop('text'+str(idx))
                data_one20.append(tmp)
        with open('../finetune/data/' + task + '/' + sp + '_same_500_simcse_top100_sp.json','r') as f:
            data_simcse = []
            for line in f:
                tmp = json.loads(line)
                for idx in range(3,100):
                    tmp.pop('text' + str(idx))
                data_simcse.append(tmp)
        with open('../finetune/data/' + task + '_demo/' + sp + '.json','r') as f:
            data_demo = []
            for line in f:
                tmp = json.loads(line)
                data_demo.append(tmp)
        with open( task + '_sample_' + sp + '.json', 'w', encoding='utf-8') as f:
            for idx in range(len(data_demo)):
                assert data_demo[idx]['text'] == data_one20[idx]['text']
                assert data_demo[idx]['text'] == data_simcse[idx]['text']
                if data_demo[idx]['labels'] == '0' and 'emotion' not in task:
                    continue
                data = {}
                data['text'] = data_demo[idx]['text']
                data['labels'] = data_demo[idx]['labels']
                for idx2 in range(len(data_demo[idx].keys())-2):
                    data['demo'+str(idx2)] = data_demo[idx]['text'+str(idx2)]
                for idx2 in range(len(data_one20[idx].keys()) - 2):
                    data['one20'+str(idx2)] = data_one20[idx]['text'+str(idx2)]
                for idx2 in range(len(data_simcse[idx].keys()) - 2):
                    data['simcse'+str(idx2)] = data_simcse[idx]['text'+str(idx2)]
                data = json.dumps(data,ensure_ascii=False)
                f.write(data+'\n')