import numpy as np
import pandas as pd
import os
import math


def main():
    df = get_dataset()
    c = pd.DataFrame([], dtype=int)
    cum_probabilities = pd.DataFrame([], dtype=float)
    for i in df['fold'].unique():
        train = df.loc[df['fold'] != i]
        test = df.loc[df['fold'] == i]
        probs = train_bayes(train.drop('fold', axis=1))
        prediction = test.apply(func=(lambda row: predict(row.tolist()[:-2], probs)), axis=1)
        c = c.add(confusion_matrix(test['label'], prediction), fill_value=0)
        cum_probabilities=cum_probabilities.add(probs, fill_value=0)

    probabilities = cum_probabilities / (df['fold'].unique().size)
    print (probabilities)
    print(c)

    samp_positive=generate_samples(probabilities, 5, 'positive')
    samp_negative = generate_samples(probabilities, 5, 'negative')

    print("Sample for positive review")
    print(samp_positive)
    print ("Sample for negative review")
    print(samp_negative)


def generate_samples(probabilities, number, type):
    p = probabilities.loc[type]
    smpls = np.random.random_sample((number,p.size)) # we have 'p.size' words and 'number' samples
    result=np.empty((0, p.size), dtype=str)
    for r in smpls:
        s=[None if c >= p.iloc[i] else p.index[i] for i,c in enumerate(r)]
        result=np.append(result, [s], axis=0)
    return result


def get_dataset():
    labels = ['awful', 'bad', 'boring', 'dull', 'effective', 'enjoyable', 'great', 'hilarious', 'label']
    dictionary = labels[0:-1]
    res = np.empty((0, 9), dtype=int)
    fold = np.empty((1, 0), dtype=int)

    for polarity in ['pos', 'neg']:
        d = "txt_sentoken/" + polarity + "/"
        all_files = os.listdir(d)
        arr = np.empty((0, 8), dtype=int)
        for f in all_files:
            fold = np.append(fold, [[(int(f[2:5]) // 100) + 1]])
            fh = open(d + f, 'r')
            comment = fh.read().lower().replace("\n", "")
            fh.close()
            counts = pd.value_counts(comment.split(" "))
            counts = counts[counts.index.isin(dictionary)].index.tolist()
            arr = np.append(arr, [[(1 if i in counts else 0) for i in dictionary]], axis=0)

        # adding the classification column
        aux = (np.zeros((arr.shape[0], arr.shape[1] + 1)) if polarity == 'neg' else np.ones(
            (arr.shape[0], arr.shape[1] + 1)));
        aux[:, :-1] = arr
        res = np.append(res, aux, axis=0)

    df = pd.DataFrame(res, columns=labels, dtype=int)
    df['label'] = df['label'].map({0: 'negative', 1: 'positive'})
    df['fold'] = pd.Series(fold)

    return df


def train_bayes(trainset, y_label='label'):
    aux = trainset.groupby(y_label)
    probab = (aux.sum() + 1) / (aux.count() + 1)  # laplace rule
    return probab


def likelihood(test_doc, word_probs):
    res = {'positive': 0, 'negative': 0}
    for (i, w) in enumerate(test_doc):
        if w == 1:
            res['positive'] += math.log(word_probs.loc['positive'].iloc[i])
            res['negative'] += math.log(word_probs.loc['negative'].iloc[i])
        else:
            res['positive'] += math.log(1 - word_probs.loc['positive'].iloc[i])
            res['negative'] += math.log(1 - word_probs.loc['negative'].iloc[i])
    return res


def predict(test_doc, word_probs):
    l = likelihood(test_doc, word_probs)
    return max(l.keys(), key=(lambda k: l[k]))


def confusion_matrix(y_actu, y_pred):
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    return df_confusion


if __name__ == '__main__':
    main()
