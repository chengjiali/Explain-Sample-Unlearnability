import pickle
from transformers import AutoTokenizer
from data.pretraining import PretrainingDataset


tokenizer = AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')


for split in ['cyber', 'bio']:
    for forget_retain in ['forget', 'retain']:
        hf_args = {'path': 'text', 'data_files': f'data/wmdp/wmdp-corpora/{split}-{forget_retain}-corpus.jsonl', 'split': 'train'}
        dataset = PretrainingDataset(hf_args, None, tokenizer, max_length=512)
        with open(f"/home/jiali/chunked-wmdp-{split}-{forget_retain}.pkl", 'wb') as f:
            pickle.dump(dataset, f)
            # for i in dataset.chunks:
            #     f.write(i)
            #     f.write('\n')

        print('Done')
