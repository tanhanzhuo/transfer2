import datasets

data_all = []
for task in ['stance/face_masks','stance/fauci','stance/school_closures','stance/stay_at_home_orders']:
    input_dir = '../finetune/data/' + task + '/prob'
    data = datasets.load_from_disk(input_dir)
    data_all.append( data )
    data_train = data['train']
    data_train[0]
