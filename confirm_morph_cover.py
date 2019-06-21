import sys
import collections

def main():
    dict_path = sys.argv[1]
    mono_path = sys.argv[2]

    confusion_set = collections.defaultdict(list)
    confusions = list()
    entry_num = 0

    with open(dict_path) as d_p:
        for line in d_p:
            line = line.strip()
            if line == '':
                for key in confusions:
                    for value in confusions:
                        if key != value:
                            confusion_set[key].append(value)
                confusions = list()
                continue

            words = line.split('\t')
            if len(words) == 1:
                idx_word = int(line)
                entry_num += 1
            else:
                word = words[0]
                confusions.append(word)

    print('coufusion_matrix size:', len(confusion_set))
    print('total entry size:', entry_num)

    count = 0
    total = 0
    with open(mono_path) as mono:
        for line in mono:
            words = line.strip().split()
            for w in words:
                total += 1
                if w in confusion_set:
                    count += 1
    
    print('total tokens:', total)
    print('include tokens:', count)
    print('coverage tokens:', count/total*100)




if __name__ == '__main__':
    main()
