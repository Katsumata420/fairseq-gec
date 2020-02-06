import sys
import collections

from subword_nmt import apply_bpe
from tqdm import tqdm

def write_dict(confusion, file_name):
    idx = 0
    with open(file_name, 'w') as o_f:
        for key, values in tqdm(confusion.items()):
            for v in values:
                output = '{} ||| {} ||| {}\n'.format(idx, key, v)
                o_f.write(output)
                idx += 1


def main():
    dict_path = sys.argv[1]
    bpe_code = sys.argv[2]
    output_file = sys.argv[3]
    copy = True
    dict_type = 'morph'

    bpe_model = apply_bpe.BPE(open(bpe_code))

    confusion_set = collections.defaultdict(set)
    confusions = list()
    entry_num = 0

    print('...read dict', file=sys.stderr)
    if dict_type == 'ppdb':
        with open(dict_path) as d_p:
            for line in tqdm(d_p):
                line = line.strip()
                elements = line.split('|||')
                src_words = elements[2].strip().lower()
                src_words = bpe_model.process_line(src_words).split(' ')
                trg_words = elements[3].strip().lower()
                trg_words = bpe_model.process_line(trg_words).split(' ')
                for s in src_words:
                    for t in trg_words:
                        if copy or s != t:
                            confusion_set[s].add(t)
                
    else:
        with open(dict_path) as d_p:
            for line in tqdm(d_p):
                line = line.strip()
                if line == '':
                    for key in confusions:
                        keys = bpe_model.process_line(key).split(' ')
                        for value in confusions:
                            values = bpe_model.process_line(value).split(' ')
                            for k in keys:
                                for v in values:
                                    if copy or k != v:
                                        confusion_set[k].add(v)
                    confusions = list()
                    continue

                words = line.split('\t')
                if len(words) == 1:
                    idx_word = int(line)
                    entry_num += 1
                else:
                    word = words[0]
                    confusions.append(word.lower())

    print('coufusion_matrix size:', len(confusion_set))
    print('total entry size:', entry_num)

    print('...write the dict', file=sys.stderr)
    write_dict(confusion_set, output_file)


if __name__ == '__main__':
    main()
