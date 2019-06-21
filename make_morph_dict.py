import sys
import collections

from tqdm import tqdm

from pymystem3 import Mystem

def main():
    text_file = sys.argv[1]
    output_dict = sys.argv[2]
    morph_dict = collections.defaultdict(set)

    model = Mystem()
    count = 0

    with open(text_file) as t_f:
        for line in t_f:
            count += 1
            if count % 100000 == 0:
                print('now count', count)
            line = line.strip()
            lemmatized_line = model.lemmatize(line) # list
            lemmatized_line = ''.join(lemmatized_line).strip().split()
            line = line.lower().split() # list

            assert len(line) == len(lemmatized_line), (len(line), len(lemmatized_line))
            for org, lemma in zip(line, lemmatized_line):
                morph_dict[lemma].add(org)

    output_file = open(output_dict, 'w')
    for idx, item in enumerate(morph_dict.items()):
        idx += 1
        lemma_key, various_set = item
        output_file.write('{}\n'.format(idx))
        output_file.write('{}\tLemma\n'.format(lemma_key))
        for v_item in various_set:
            output_file.write('{}\tmorph_cand\n'.format(v_item))
        output_file.write('\n')

        """
        print(idx)
        print('{}\tLemma'.format(lemma_key))
        for v_item in various_set:
            print('{}\tmorph_cand'.format(v_item))
        print()
        """

    print('morph entry size:', len(morph_dict), file=sys.stderr)

if __name__ == '__main__':
    main()
