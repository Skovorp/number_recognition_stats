from textwrap import dedent
import custom_analysis as func
import pandas as pd
import argparse
import re
import numpy as np

from os import path as pt


def create_nones_stats(data):

    with open(pt.join(path, "Nones_stats.txt"), "a") as file_nones:
        print(
            f"\nNumber - detection -: "
            f"{len(data[(data['number'] == 'None') & (data['prediction'] == 'None')])}\n"
            f"Number + detection -: "
            f"{len(data[(data['number'] != 'None') & (data['prediction'] == 'None')])}\n"
            f"Number - detection +: "
            f"{len(data[(data['number'] == 'None') & (data['prediction'] != 'None')])}\n"
            f"Number + detection +: "
            f"{len(data[(data['number'] != 'None') & (data['prediction'] != 'None')])}",
            file=file_nones)
        if len(data[(data['number'] != 'None') & (
                data['prediction'] == 'None')]) != 0:
            print("\nNumber wasn't detected: \n",
                  data[(data['number'] != 'None') & (
                          data['prediction'] == 'None')], file=file_nones)
        if len(data[(data['number'] == 'None') & (
                data['prediction'] != 'None')]) > 0:
            print("\nNumber falsely detected:\n",
                  data[(data['number'] == 'None') & (
                          data['prediction'] != 'None')], file=file_nones)

        if len(data[data.isnull().any(axis=1)]) != 0:
            print("\nNaNs from merging files:\n",
                  data[data.isnull().any(axis=1)], file=file_nones)
        else:
            print("No nans from merging", file=file_nones)

        # deleting all rows with Nones
        data = data[
            (data['number'] != 'None') & (data['prediction'] != 'None')]
        data = data.dropna()
        data = data.reset_index(drop=True)

        return data


def create_ids_stats(data):

    abc = []  # already seen chars
    sub_arr = np.zeros((0, 0))  # 2d np.array of substitutions
    del_arr = []  # list of deletions
    ins_arr = []  # list of insertions"""

    data['lev_dist'] = -1.0

    for i in range(data.shape[0]):
        data.at[i, 'lev_dist'], sub_arr = func.mod_lev(data.at[i, 'number'],
                                                       data.at[
                                                           i, 'prediction'],
                                                       weights_lev_dist, abc,
                                                       sub_arr, del_arr,
                                                       ins_arr)

    arrays_stat_norm = func.arrays_from_logs(sub_arr, del_arr, ins_arr,
                                                 abc, data['number'])
    arrays_stat_norm.to_excel(pt.join(path, "ids_stat_normal.xlsx"))

    arrays_stat = func.arrays_from_logs(sub_arr, del_arr, ins_arr, abc)
    arrays_stat.to_excel(pt.join(path, "ids_stat.xlsx"))
    return data


def create_all_mistakes_file(data):
    with open(pt.join(path, "All_mistakes.txt"), "a") as file_mistakes:
        print(f"All numbers: {len(data['number'])}\n"
              f"Numbers with mistakes: "
              f"{len(data[data['number'] != data['prediction']])}\n"
              f"% of mistakes: "
              f"{round(len(data[data['number'] != data['prediction']]) / len(data['number']) * 100, 2)}\n",
              file=file_mistakes)
        print(data[data['number'] != data['prediction']].sort_values(
            'lev_dist', ascending=False).to_string(index=False),
              file=file_mistakes)


def create_type_stats(data, types):

    numb_pattern = "[0-9?]"
    let_pattern = "[A-Z?]"
    re_types = []
    for i, type_pattern in enumerate(types):
        re_type_pattern = ""
        for j, let in enumerate(type_pattern):
            if let.isdigit():
                re_type_pattern += numb_pattern
            elif let.isalpha():
                re_type_pattern += let_pattern
            elif let == "!":
                re_type_pattern += "?"
            else:
                raise Exception("Unknown symbol in type pattern")
        re_types.append(re.compile(re_type_pattern))

    with open(pt.join(path, "Type_stats.txt"), "a") as file_type:
        data['num_type'] = data.apply(
            lambda x: func.type_detect(x['number'], re_types),
            axis=1)
        data['pred_type'] = data.apply(
            lambda x: func.type_detect(x['prediction'], re_types),
            axis=1)

        for i, type_name in enumerate(types):
            print(
                f"Numbers of type {i} ({type_name})]:"
                f" {len(data[data['num_type'] == i])}", file=file_type)

        print("Mistakes:", file=file_type)

        if len(data[data['pred_type'] == -2]) == 0:
            print("\nTo many ? for type identification: 0\n", file=file_type)
        else:
            print(
                f"\nTo many ? for type identification: "
                f"{len(data[data['pred_type'] == -2])}\n"
                f"{data[data['pred_type'] == -2].to_string(line_width=500, index=False)}",
                file=file_type)

        if len(data[data['pred_type'] == -1]) == 0:
            print("\nImpossible type: 0", file=file_type)
        else:
            print(f"\nImpossible type: {len(data[data['pred_type'] == -1])}\n"
                  f"{data[data['pred_type'] == -1].to_string(line_width=500, header=False, index=False)}",
                  file=file_type)

        if len(data[(data['pred_type'] != data['num_type']) & (
                data['pred_type'] >= 1)]) == 0:
            print("\nWrong type: 0", file=file_type)
        else:
            print(
                f"\nWrong type: {len(data[(data['pred_type'] != data['num_type']) & (data['pred_type'] >= 1)])}\n"
                f"{data[(data['pred_type'] != data['num_type']) & (data['pred_type'] >= 1)].to_string(line_width=500, header=False, index=False)}",
                file=file_type)
    return data


def create_merged_data_file(data):
    data.to_csv(pt.join(path, "Merged_data.csv"))


if __name__ == "__main__":
    formatter_class = argparse.RawDescriptionHelpFormatter,

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent("""\
    Creates analytics for number detection:
       Writes table with inserts, deletes, substitutions (ids) to ids.xlsx 
            and ids_normal.xlsx (normalized by letter freq in gt)
       Writes stats of number detection to Nones_stats.txt
       Writes stats of type detection to Type_stats.txt
       Writes all numbers with mistakes to All_mistakes.txt
       Writes all ids operations made on all numbers to ids_log.txt
       Writes merged gt and results to Merged_data.csv"""))

    parser.add_argument("-g", "--gt_path", metavar='', required=True, type=str,
                        help='Path to gt.csv')
    parser.add_argument('-r', '--results_path', metavar='', required=True,
                        type=str,
                        help='Path to results.csv')
    parser.add_argument('-o', '--out_path', metavar='', type=str,
                        default='Folder of results.csv',
                        help='Path to an output directory')

    parser.add_argument('-dp', '--del_price', metavar='', type=float,
                        default=1,
                        help='Deletion price in Levenstein distance')
    parser.add_argument('-ip', '--ins_price', metavar='', type=float,
                        default=1,
                        help='Insertion price in Levenstein distance')
    parser.add_argument('-sp', '--sub_price', metavar='', type=float,
                        default=1,
                        help='Substitution price in Levenstein distance')
    parser.add_argument('-dpq', '--del_price_q', metavar='', type=float,
                        default=0.9,
                        help='? deletion price in Levenstein distance')
    parser.add_argument('-spq', '--sub_price_q', metavar='', type=float,
                        default=1,
                        help='? substitution price in Levenstein distance')
    parser.add_argument('-typs', '--type_patterns', metavar='', nargs='*',
                        type=str,
                        default=["A123BC777!", "AB1234777!"],
                        help='List of type patterns. For each type write a '
                             'number of maximum length with A-Z or 0-9 '
                             'at places for letters and numbers respectively.'
                             'After optional characters place !')

    args = parser.parse_args()

    path = args.out_path
    gt_path = args.gt_path
    results_path = args.results_path
    types_str = args.type_patterns

    weights_lev_dist = [args.del_price, args.ins_price, args.sub_price,
                        args.del_price_q, args.sub_price_q]

    if path == 'Folder of results.csv':
        path = pt.abspath(pt.join(results_path, pt.pardir))

    path = func.create_folder(path)

    print("files at", path)

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    df1 = pd.read_csv(gt_path, header=None)
    df2 = pd.read_csv(results_path, header=None)
    
    df1.drop(df1.columns.difference([0, 1]), axis=1, inplace=True)
    df2.drop(df2.columns.difference([0, 1]), axis=1, inplace=True)           
                
    df1.columns=['file', 'number']
    df2.columns=['file', 'prediction']
    
    data = df1.merge(df2, on='file', how='outer')

    data = create_nones_stats(data)
    data = create_ids_stats(data)
    create_all_mistakes_file(data)
    data = create_type_stats(data, types_str)
    create_merged_data_file(data)
