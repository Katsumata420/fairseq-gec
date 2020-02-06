import numpy as np
import re
import random
import shutil
import sys

from fairseq.tokenizer import tokenize_line
from tqdm import tqdm

import collections

class NoiseInjector(object):

    def __init__(self, corpus, morph_dict, is_morph_with=False, 
                 prob_rep_morph=False, dict_form=False, shuffle_sigma=0.5,
                 replace_mean=0.5, replace_std=0.03,
                 delete_mean=0.1, delete_std=0.03,
                 add_mean=0.1, add_std=0.03):
        # READ-ONLY, do not modify
        self.corpus = corpus
        self.shuffle_sigma = shuffle_sigma
        # self.replace_a, self.replace_b = self._solve_ab_given_mean_var(replace_mean, replace_std**2)
        self.replace_a, self.replace_b = self._solve_ab_given_mean_var(replace_mean, replace_std**2)
        self.delete_a, self.delete_b = self._solve_ab_given_mean_var(delete_mean, delete_std**2)
        self.add_a, self.add_b = self._solve_ab_given_mean_var(add_mean, add_std**2)
        
        # for morph dict
        self.is_morph_with = is_morph_with
        self.prob_rep_morph = prob_rep_morph
        # TODO: ここの confusion set を作る処理、キモすぎ
        if dict_form: # if True, dict form is ppdb.
            self.confusion_set = collections.defaultdict(list) 
            with open(morph_dict) as morph:
                for line in morph:
                    line = line.strip()
                    elements = line.split('|||')
                    self.confusion_set[elements[1].strip().lower()].append(elements[2].strip().lower())
            print('fin load the ppdb', file=sys.stderr)
            return None
        # dict form が opencorpora 系の場合はこちら

        if morph_dict == '':
            self.morph_dict = None
            self.confusion_set = None
        else:
            self.morph_dict = morph_dict
            confusions = list()
            self.confusion_set = collections.defaultdict(list)
            with open(morph_dict) as morph:
                for line in morph:
                    line = line.strip()
                    if line == '':
                        for key in confusions:
                            for value in confusions:
                                if key != value:
                                    self.confusion_set[key].append(value)
                        confusions = list()
                        continue

                    words = line.split('\t')
                    if len(words) == 1:
                        idx_word = int(line)
                    else:
                        word = words[0]
                        # TODO: control lower case
                        # confusions.append(word)
                        confusions.append(word.lower())


    @staticmethod
    def _solve_ab_given_mean_var(mean, var):
        a = mean * mean * (1. - mean) / var - mean
        b = (1. - mean) * (mean * (1. - mean) / var - 1.)
        return a, b

    def _shuffle_func(self, tgt):
        if self.shuffle_sigma < 1e-6:
            return tgt

        shuffle_key = [i + np.random.normal(loc=0, scale=self.shuffle_sigma) for i in range(len(tgt))]
        new_idx = np.argsort(shuffle_key)
        res = [tgt[i] for i in new_idx]
            
        return res

    def _replace_func(self, tgt):
        """
        tgt: 一文 (list)
        rnd: 単語に対して確率付与
        confusion set に存在しなかったら replace しない方針
        -> replace による単語誤りは全て morph 変化由来
        TODO: if の分岐おかしくない？
        
        is_morph_with で replace と一緒に morph 置換するのか決定
        """
        replace_ratio = np.random.beta(self.replace_a, self.replace_b)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, p in enumerate(tgt):
            if rnd[i] < replace_ratio: 
                if self.is_morph_with and self.morph_dict is not None:
                    if p in self.confusion_set:
                        rnd_morph_word = random.choice(self.confusion_set[p])
                        ret.append((-1, rnd_morph_word))
                    else:
                        ret.append(p)
                else:
                    # rand_ex: corpus 内の任意の文 
                    # rand_word: ex 内の任意の単語
                    rnd_ex = self.corpus[np.random.randint(len(self.corpus))]
                    rnd_word = rnd_ex[np.random.randint(len(rnd_ex))]
                    ret.append((-1, rnd_word))
            else:
                ret.append(p)
        return ret

    def _replace_morph_func(self, tgt):
        """
        _replace_func との diff
        - 辞書に入っていたら全て置換する点
        - 確率的な処理に置換するのではなく、該当エントリがあれば置換
        """
        ret = []
        replace_ratio = np.random.beta(self.replace_a, self.replace_b)
        if self.prob_rep_morph:
            replace_ratio = 1.00
        rnd = np.random.random(len(tgt))
        count = 0
        for i, p in enumerate(tgt):
            if rnd[i] < replace_ratio:
                if p[1] in self.confusion_set:
                    count += 1
                    rnd_morph_word = random.choice(self.confusion_set[p[1]])
                    ret.append((-1, rnd_morph_word))
                else:
                    ret.append(p)
            else:
                ret.append(p)
        return [ret, count]


    def _delete_func(self, tgt):
        delete_ratio = np.random.beta(self.delete_a, self.delete_b)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, p in enumerate(tgt):
            if rnd[i] < delete_ratio:
                continue
            ret.append(p)
        return ret

    def _add_func(self, tgt):
        add_ratio = np.random.beta(self.add_a, self.add_b)
        ret = []
        rnd = np.random.random(len(tgt))
        for i, p in enumerate(tgt):
            if rnd[i] < add_ratio:
                rnd_ex = self.corpus[np.random.randint(len(self.corpus))]
                rnd_word = rnd_ex[np.random.randint(len(rnd_ex))]
                ret.append((-1, rnd_word))
            ret.append(p)

        return ret

    def _parse(self, pairs):
        align = []
        art = []
        for si in range(len(pairs)):
            ti = pairs[si][0]
            w = pairs[si][1]
            art.append(w)
            if ti >= 0:
                align.append('{}-{}'.format(si, ti))
        return art, align

    def inject_noise(self, tokens):
        # tgt is a vector of integers

        # funcs = [self._add_func, self._shuffle_func, self._replace_func, self._delete_func]
        # TODO: replace の扱い
        funcs = [self._add_func, self._shuffle_func, self._delete_func]
        if not self.is_morph_with:
            funcs.append(self._replace_morph_func)
        np.random.shuffle(funcs)
        
        pairs = [(i, w) for (i, w) in enumerate(tokens)]
        count = 0
        for f in funcs:
            pairs = f(pairs)
            if f == self._replace_morph_func:
                count += pairs[-1] 
                pairs = pairs[0]
            art, align = self._parse(pairs)

        return self._parse(pairs), count


def save_file(filename, contents):
    with open(filename, 'w') as ofile:
        for content in contents:
            ofile.write(' '.join(content) + '\n')

# make noise from filename
def noise(filename, ofile_suffix, morph_dict=None, ppdb_format=False, rep_prob=0.2):
    is_morph_with_replace = False # ここ True の場合は確率置換に入ってくる 
    prob_rep_morph = False

    lines = open(filename).readlines()
    tgts = [tokenize_line(line.strip()) for line in lines]
    noise_injector = NoiseInjector(tgts, morph_dict, 
                                   is_morph_with=is_morph_with_replace,
                                   prob_rep_morph=prob_rep_morph, 
                                   dict_form=ppdb_format,
                                   replace_mean=rep_prob)
    
    srcs = []
    aligns = []
    replace_count = 0
    for tgt in tqdm(tgts):
        # tgt は1文単位
        (src, align), count = noise_injector.inject_noise(tgt)
        replace_count += count
        srcs.append(src)
        aligns.append(align)
    
    print('replace count is :',replace_count)
    save_file('{}.src'.format(ofile_suffix), srcs)
    save_file('{}.tgt'.format(ofile_suffix), tgts)
    save_file('{}.forward'.format(ofile_suffix), aligns)

import argparse

parser=argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', type=int, default=10)
parser.add_argument('-s', '--seed', type=int, default=2468)
parser.add_argument('-d', '--data', type=str, default='./data/train_1b.tgt')
parser.add_argument('--morph_dict', type=str, default='', 
                    help='if exist morph dict, use it in replaced noise.')
parser.add_argument('--use_phrase', action='store_true', 
                    help='if true, your dict is written in paraphrase format.')
parser.add_argument('--rep_prob', type=float, default=0.2,
                    help='replace probability; default is 0.2')

args = parser.parse_args()
np.random.seed(args.seed)
random.seed(args.seed)
if __name__ == '__main__':
    """
    変更箇所は2つ, is_morph_with を False にしたり
    _replace_func を入れたり
    """
    print("epoch={}, seed={}".format(args.epoch, args.seed))

    filename = args.data
    ofile_suffix = './data_art_morph/train_1b_{}'.format(args.epoch)

    noise(filename, ofile_suffix, args.morph_dict, args.use_phrase, args.rep_prob)

