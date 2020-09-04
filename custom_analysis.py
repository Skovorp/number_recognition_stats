import datetime
import os

import numpy as np
import pandas as pd


def mod_lev(numb, pred, weights, abc, sub_arr, del_arr, ins_arr):
    """Finds Levenshtein distance and writes edit sequence to file
        Function first calculates array of substrings distances encouraging
                                                            deleting of ?
        Then moving backwards finds used  for the final distance

    Parameters:
        numb (string): Actual number
        pred (string): Prediction of number by algorithm
        weights (5 elem list): List of prices for operations
        abc (array): Array of already seen letters and numbers
        sub_arr (np 2d array): Contains substitution stats
        del_arr (array): Contains deletion stats
        ins_arr (array): Contains insertion stats
    Returns:
        Levenshtein distance between numb and pred
    """

    # costs of operations, when modifying pred
    del_price = weights[0]  # default 1
    ins_price = weights[1]  # default 1
    sub_price = weights[2]  # default 1

    # costs of operations with ?
    # no ? in number, so no insertions of ?
    del_price_q = weights[3]  # default 0.9
    sub_price_q = weights[4]  # default

    dist = np.full((len(pred) + 2, len(numb) + 2), np.inf)

    dist[1, 1] = 0
    # dist[_, 1] depends on del_price_q, so it's defined iteratively
    for i, _ in enumerate(numb):
        dist[1, i + 2] = (i + 1) * ins_price

    for i, letter_pred in enumerate(pred):
        if letter_pred == '?':
            for j, letter_numb in enumerate('*' + numb):
                del_cost = dist[i + 1, j + 1] + del_price_q
                ins_cost = dist[i + 2, j] + ins_price
                sub_cost = dist[i + 1, j] if letter_pred == letter_numb else \
                    dist[i + 1, j] + sub_price_q

                dist[i + 2, j + 1] = min(sub_cost, ins_cost, del_cost)

        else:
            for j, letter_numb in enumerate('*' + numb):
                del_cost = dist[i + 1, j + 1] + del_price
                ins_cost = dist[i + 2, j] + ins_price
                sub_cost = dist[i + 1, j] if letter_pred == letter_numb else \
                    dist[i + 1, j] + sub_price

                dist[i + 2, j + 1] = min(sub_cost, ins_cost, del_cost)

    # creating edit sequence
    i, j = len(pred) + 1, len(numb) + 1

    while (i > 1) or (j > 1):
        if pred[i - 2] == '?':
            if (dist[i, j] == (dist[i - 1, j - 1] + sub_price_q)) and (
                    numb[j - 2] != pred[i - 2]):

                wrong = pred[i - 2]
                right = numb[j - 2]

                wrong_index, sub_arr = index_and_arrays(wrong, abc, sub_arr,
                                                        ins_arr, del_arr)
                right_index, sub_arr = index_and_arrays(right, abc, sub_arr,
                                                        ins_arr, del_arr)

                sub_arr[right_index][wrong_index] += 1

                i -= 1
                j -= 1
            elif dist[i, j] == (dist[i, j - 1] + ins_price):

                let = numb[j - 2]

                di_index, sub_arr = index_and_arrays(let, abc, sub_arr,
                                                     ins_arr,
                                                     del_arr)
                ins_arr[di_index] += 1

                j -= 1
            elif dist[i, j] == (dist[i - 1, j] + del_price_q):

                let = pred[i - 2]

                di_index, sub_arr = index_and_arrays(let, abc, sub_arr,
                                                     ins_arr,
                                                     del_arr)
                del_arr[di_index] += 1

                i -= 1
            else:
                # characters match - no action needed
                i -= 1
                j -= 1
        else:
            if dist[i, j] == (dist[i - 1, j - 1] + sub_price) and (
                    numb[j - 2] != pred[i - 2]):

                wrong = pred[i - 2]
                right = numb[j - 2]

                wrong_index, sub_arr = index_and_arrays(wrong, abc, sub_arr,
                                                        ins_arr, del_arr)
                right_index, sub_arr = index_and_arrays(right, abc, sub_arr,
                                                        ins_arr, del_arr)

                sub_arr[right_index][wrong_index] += 1

                i -= 1
                j -= 1
            elif dist[i, j] == (dist[i, j - 1] + ins_price):

                let = numb[j - 2]

                di_index, sub_arr = index_and_arrays(let, abc, sub_arr,
                                                     ins_arr,
                                                     del_arr)
                ins_arr[di_index] += 1

                j -= 1
            elif dist[i, j] == (dist[i - 1, j] + del_price):

                let = pred[i - 2]

                di_index, sub_arr = index_and_arrays(let, abc, sub_arr,
                                                     ins_arr,
                                                     del_arr)
                del_arr[di_index] += 1

                i -= 1
            else:
                # characters match - no action needed
                i -= 1
                j -= 1
    return dist[-1, -1], sub_arr


def type_detect(s, types):
    """
    Defines type of a car number

    Parameters:
        s (string): Number for type definition
        types (array): Array with all type patterns as regular expressions
            default types are А123ВС77(7), АВ123477(7)
    Returns:
        -2 if number can be several types
        -1 if none of all types are possible
        i if type is types[i]"""

    match_pos = -1

    for type_i, type_pattern in enumerate(types):
        if type_pattern.match(s):
            if type_pattern.match(s).span() == (0, len(s)):
                if match_pos != -1:
                    return -2
                else:
                    match_pos = type_i

    return match_pos


def index_and_arrays(ch, abc, sub_arr, ins_arr, del_arr):
    """Adds new characters to abc and arrays of operations, calculates indexes
    indexes for the same character are the same for all arrays"""
    sub_arr_changed = np.copy(sub_arr)
    if ch not in abc:
        # np.resize method doesn't works when arr is referenced
        sub_arr_changed = np.vstack(
            (sub_arr_changed, [0] * len(sub_arr_changed)))  # + row
        sub_arr_changed = np.hstack((sub_arr_changed, np.transpose(
            [[0] * (np.shape(sub_arr_changed)[0])])))  # + col
        ins_arr.append(0)
        del_arr.append(0)

        abc.append(ch)
        return len(abc) - 1, np.copy(sub_arr_changed)
    else:
        return abc.index(ch), np.copy(sub_arr_changed)


def arrays_from_logs(sub_arr, del_arr, ins_arr, abc,
                     numbers_column=pd.DataFrame([])):
    """Prints 3 arrays with stats for each type of operation from logs"""

    np.set_printoptions(linewidth=200)

    # adds sum of cols
    sub_arr = np.vstack((sub_arr, [0] * len(sub_arr)))

    for i, _ in enumerate(abc):
        sub_arr[-1, i] = np.sum(sub_arr[:, i])

    df = pd.DataFrame(sub_arr, columns=abc, index=abc + ["sums"])
    df = df.append(pd.DataFrame([ins_arr], columns=abc, index=['Inserts']))
    df = df.append(pd.DataFrame([del_arr], columns=abc, index=['Deletes']))

    if not numbers_column.empty:
        counts = count_symbols(numbers_column, abc)
        df = df.div(counts)
        df['?'] = "no stat"

    # ads sum of rows
    df['sums'] = df.sum(axis=1)

    return df


def count_symbols(df, abc):
    """Returns an array of symbol amounts in gt in abc order"""
    counts = [0] * len(abc)
    df = df.to_numpy()
    for numb in df:
        for _, let in enumerate(numb):
            if let in abc:
                counts[abc.index(let)] += 1

    return counts


def create_folder(path):
    """Creates folder 'path/number_tests/current time' for running tests"""

    path = os.path.join(path,
                        "number_tests",
                        datetime.datetime.now().isoformat(sep="_").replace(':',
                                                                           '.'))

    if not os.path.exists(path):
        os.makedirs(path)

    return path
