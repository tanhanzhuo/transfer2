import json
tasks = 'eval-stance,eval-emotion,eval-irony,eval-offensive,eval-hate,sem21-task7-humor,sem22-task6-sarcasm'.split(',')
for task in tasks:
    for sp in ['train','dev','test']:
        with open('../finetune/data/'+task+'/'+sp+'_same_500_one20_top100_sp.json','r') as f:
            data_one20 = []
            for line in f:
                tmp = json.loads(line)
                for idx in range(100):
                    tmp.pop('text'+str(idx))
                data_one20.append(tmp)
        with open('../finetune/data/' + task + '/' + sp + '_same_500_simcse_top100_sp.json','r') as f:
            data_simcse = []
            for line in f:
                tmp = json.loads(line)
                for idx in range(100):
                    tmp.pop('text' + str(idx))
                data_simcse.append(tmp)
        with open('../finetune/data/' + task + '_demo/' + sp + '.json','r') as f:
            data_demo = []
            for line in f:
                tmp = json.loads(line)
                data_demo.append(tmp)
        with open( task + '_sample_' + sp + '.json', 'w', encoding='utf-8') as f:
            for idx in range(len(data_demo)):
                assert data_demo['text'] == data_one20['text']
                assert data_demo['text'] == data_simcse['text']
                if data_demo['labels'] == '0':
                    continue
                data = {}
                data['text'] = data_demo['text']
                data['labels'] = data_demo['labels']
                for idx2 in range(len(data_demo.keys())-2):
                    data['demo'+str(idx2)] = data_demo['text'+str(idx2)]
                data['one20'] = data_one20['text0']
                data['simcse'] = data_simcse['text0']
                data = json.dumps(data,ensure_ascii=False)
                f.write(data+'\n')