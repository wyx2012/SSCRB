import numpy as np
import collections
import math

import torch



#%%定义处理数据方法
coden_dict = {'GCU': 0, 'GCC': 0, 'GCA': 0, 'GCG': 0,                             # alanine<A>
              'UGU': 1, 'UGC': 1,                                                 # systeine<C>
              'GAU': 2, 'GAC': 2,                                                 # aspartic acid<D>
              'GAA': 3, 'GAG': 3,                                                 # glutamic acid<E>
              'UUU': 4, 'UUC': 4,                                                 # phenylanaline<F>
              'GGU': 5, 'GGC': 5, 'GGA': 5, 'GGG': 5,                             # glycine<G>
              'CAU': 6, 'CAC': 6,                                                 # histidine<H>
              'AUU': 7, 'AUC': 7, 'AUA': 7,                                       # isoleucine<I>
              'AAA': 8, 'AAG': 8,                                                 # lycine<K>
              'UUA': 9, 'UUG': 9, 'CUU': 9, 'CUC': 9, 'CUA': 9, 'CUG': 9,         # leucine<L>
              'AUG': 10,                                                          # methionine<M>
              'AAU': 11, 'AAC': 11,                                               # asparagine<N>
              'CCU': 12, 'CCC': 12, 'CCA': 12, 'CCG': 12,                         # proline<P>
              'CAA': 13, 'CAG': 13,                                               # glutamine<Q>
              'CGU': 14, 'CGC': 14, 'CGA': 14, 'CGG': 14, 'AGA': 14, 'AGG': 14,   # arginine<R>
              'UCU': 15, 'UCC': 15, 'UCA': 15, 'UCG': 15, 'AGU': 15, 'AGC': 15,   # serine<S>
              'ACU': 16, 'ACC': 16, 'ACA': 16, 'ACG': 16,                         # threonine<T>
              'GUU': 17, 'GUC': 17, 'GUA': 17, 'GUG': 17,                         # valine<V>
              'UGG': 18,                                                          # tryptophan<W>
              'UAU': 19, 'UAC': 19,                                               # tyrosine(Y)
              'UAA': 20, 'UAG': 20, 'UGA': 20,                                    # STOP code
              }
def get_1_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**1
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        nucle_com.append(ch0)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return  word_index

def get_2_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**2
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        nucle_com.append(ch0 + ch1)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return  word_index

def get_3_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**3
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n//base
        ch1=chars[n%base]
        n=n//base
        ch2=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2)
    word_index = dict((w, i) for i, w in enumerate(nucle_com))
    return  word_index

def coden(seq,kmer,tris):
    coden_dict = tris
    freq_dict = frequency(seq,kmer,coden_dict)
    vectors = np.zeros((101, len(coden_dict.keys())))
    for i in range(len(seq) - int(kmer) + 1):
        value = freq_dict[coden_dict[seq[i:i+kmer].replace('T', 'U')]]
        vectors[i][coden_dict[seq[i:i+kmer].replace('T', 'U')]] = value/100
    return vectors

def codenKNF(seq):
    vectors = np.zeros((len(seq), 21))
    for i in range(len(seq)):
        if i < len(seq)-2:
            vectors[i][coden_dict[seq[i:i+3].replace('T', 'U')]] = 1
    return vectors.tolist()

def frequency(seq,kmer,coden_dict):
    Value = []
    k = kmer
    coden_dict = coden_dict
    for i in range(len(seq) - int(k) + 1):
        kmer = seq[i:i+k]
        kmer_value = coden_dict[kmer.replace('T', 'U')]
        Value.append(kmer_value)
    freq_dict = dict(collections.Counter(Value))
    return freq_dict

def creatmat(data):
    mat = np.zeros([len(data),len(data)])
    for i in range(len(data)):
        for j in range(len(data)):
            coefficient = 0
            for add in range(30):
                if i - add >= 0 and j + add <len(data):
                    score = paired(data[i - add],data[j + add])
                    if score == 0:
                        break
                    else:
                        coefficient = coefficient + score * Gaussian(add)
                else:
                    break
            if coefficient > 0:
                for add in range(1,30):
                    if i + add < len(data) and j - add >= 0:
                        score = paired(data[i + add],data[j - add])
                        if score == 0:
                            break
                        else:
                            coefficient = coefficient + score * Gaussian(add)
                    else:
                        break
            mat[[i],[j]] = coefficient
    return mat


def Gaussian(x):
    return math.exp(-0.5*(x*x))

def paired(x,y):
    if x == 'A' and y == 'U':
        return 2
    elif x == 'G' and y == 'C':
        return 3
    elif x == "G"and y == 'U':
        return 0.8
    elif x == 'U' and y == 'A':
        return 2
    elif x == 'C' and y == 'G':
        return 3
    elif x == "U"and y == 'G':
        return 0.8
    else:
        return 0
#%%处理Kmer，输出为101×84
def dealwithdata1(protein):
    tris1 = get_1_trids()
    tris2 = get_2_trids()
    tris3 = get_3_trids()
    dataX = []
    with open(r'./Datasets/circRNA-RBP/'+protein+'/positive') as f:
        for line in f:
            if '>' not in line:
                kmer1 = coden(line.strip(),1,tris1)
                kmer2 = coden(line.strip(),2,tris2)
                kmer3 = coden(line.strip(),3,tris3)
                Kmer = np.hstack((kmer1,kmer2,kmer3))
                dataX.append(Kmer.tolist())
    with open(r'./Datasets/circRNA-RBP/'+protein+'/negative') as f:
        for line in f:
            if '>' not in line:
                kmer1 = coden(line.strip(),1,tris1)
                kmer2 = coden(line.strip(),2,tris2)
                kmer3 = coden(line.strip(),3,tris3)
                Kmer = np.hstack((kmer1,kmer2,kmer3))
                dataX.append(Kmer.tolist())

    dataX = np.array(dataX)
    return dataX
#%%处理结构特征，输出为101×101
def creatmat(protein):
    #data_pair = np.arange(10201).reshape(1,101,101)
    pair_data = []
    with open(r'./Datasets/circRNA-RBP/'+protein+'/positive') as f:
        num = 0
        for data in f:
            if '>' not in data:
                data = data[:-1]
                mat = np.zeros([len(data),len(data)])
                for i in range(len(data)):
                    for j in range(len(data)):
                        coefficient = 0
                        for add in range(30):
                            if i - add >= 0 and j + add <len(data):
                                score = paired(data[i - add].replace('T', 'U'),data[j + add].replace('T', 'U'))
                                if score == 0:
                                    break
                                else:
                                    coefficient = coefficient + score * Gaussian(add)
                            else:
                                break
                        if coefficient > 0:
                            for add in range(1,30):
                                if i + add < len(data) and j - add >= 0:
                                    score = paired(data[i + add],data[j - add])
                                    if score == 0:
                                        break
                                    else:
                                        coefficient = coefficient + score * Gaussian(add)
                                else:
                                    break
                        mat[[i],[j]] = coefficient
                if len(pair_data)==0:
                    pair_data = torch.from_numpy(mat).unsqueeze(0)
                else:
                    matt = torch.from_numpy(mat).unsqueeze(0)
                    pair_data = torch.cat((pair_data,matt),0)
                num=num+1
                print(num)
    with open(r'./Datasets/circRNA-RBP/'+protein+'/negative') as f:
        for data in f:
            if '>' not in data:
                data = data[:-1]
                mat = np.zeros([len(data), len(data)])
                for i in range(len(data)):
                    for j in range(len(data)):
                        coefficient = 0
                        for add in range(30):
                            if i - add >= 0 and j + add < len(data):
                                score = paired(data[i - add], data[j + add])
                                if score == 0:
                                    break
                                else:
                                    coefficient = coefficient + score * Gaussian(add)
                            else:
                                break
                        if coefficient > 0:
                            for add in range(1, 30):
                                if i + add < len(data) and j - add >= 0:
                                    score = paired(data[i + add], data[j - add])
                                    if score == 0:
                                        break
                                    else:
                                        coefficient = coefficient + score * Gaussian(add)
                                else:
                                    break
                        mat[[i], [j]] = coefficient

                matt = torch.from_numpy(mat).unsqueeze(0)
                pair_data = torch.cat((pair_data, matt), 0)
                num = num + 1
                print(num)
    return pair_data
#%%处理KNF特征，输出为99×21
def dealwithdataKNF(protein):
    dataXKNF = []
    dataYKNF = []
    with open(r'./Datasets/circRNA-RBP/'+protein+'/positive') as f:
            for line in f:
                if '>' not in line:
                    dataXKNF.append(codenKNF(line.strip()))
                    dataYKNF.append(1)
    with open(r'./Datasets/circRNA-RBP/'+protein+'/negative') as f:
            for line in f:
                if '>' not in line:
                    dataXKNF.append(codenKNF(line.strip()))
                    dataYKNF.append(0)
    #indexes = np.random.choice(len(dataYKNF), len(dataYKNF), replace=False)
    # dataX = np.array(dataXKNF)[indexes]
    # dataY = np.array(dataYKNF)[indexes]
    dataX = np.array(dataXKNF)
    dataY = np.array(dataYKNF)
    return dataX,dataY
#%%读取主函数
def all_data(protein):
    KNF = dealwithdataKNF(protein)
    Kmer = dealwithdata1(protein)
    knf = KNF[0]
    Y = KNF[1]
    return Kmer,knf,Y





