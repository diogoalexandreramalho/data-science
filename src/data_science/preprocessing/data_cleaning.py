import json
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_science.datasets import DATASETS

"""register_matplotlib_converters()
data = pd.read_csv('Data/covtype.csv', sep=',', decimal='.')
data = data.groupby('Cover_Type').apply(lambda s: s.sample(1000))
print(data)"""


# Creates a dic with a list of columns names associated to the titles given in the csv file
def general_dic(write_file):
    dic = DATASETS["PD"].feature_groups

    if write_file:
        with open("dic.txt", "w") as file:
            file.write(json.dumps(dic))

    return dic


# receives a group of columns and divides them according to their initial letters
def group_dic(group_data, key_len, write_file):

    initials = []
    group_dic = {}

    for col in group_data:
        if col[:key_len] not in initials:
            initials += [col[:key_len]]
            group_dic[col[:key_len]] = [col]
        else:
            group_dic[col[:key_len]] += [col]

    if write_file:
        with open("dic.txt", "w") as file:
            file.write(json.dumps(group_dic))

    return group_dic


# get the data associated to a set of variables
def get_data_from_dic(data, dic, group_name):
    lst = dic[group_name]
    return data[lst]


# get group of data associated based on a regular expression
def get_data_by_expression(group_data, reg_expression):
    group_lst = []

    for var in group_data:
        x = re.findall(reg_expression, var)
        if x:
            group_lst += [var]

    return group_data[group_lst]


# correlate a group of variables
def group_correlation(group_data):
    plt.figure(figsize=[14, 7])
    corr_mtx = group_data.corr()
    sns.heatmap(
        corr_mtx,
        xticklabels=corr_mtx.columns,
        yticklabels=corr_mtx.columns,
        annot=True,
        cmap="Blues",
    )
    plt.title("Correlation analysis")
    plt.show()


# calculates new variable from the mean of a group of variables
def add_variable_from_mean(data, new_var, vars_lst, delete):
    data.insert(0, new_var, 0)

    for i in range(756):
        num = 0
        for col in vars_lst:
            num += data[col][i]

        num /= len(vars_lst)

        data.loc[i, new_var] = num

    if delete:
        for var in vars_lst:
            del data[var]

    return data


def delete_columns(data, vars_lst):
    for var in vars_lst:
        del data[var]
    return data


# Produz todas as variaveis que tem o mesmo sufixo
def produce_allvariables(s, n):
    lst = []
    for i in range(1, n):
        v = s + str(i)
        lst.append(v)
    return lst


##############################
####### DISTRIBUTIONS ########
##############################

dic = general_dic(False)


def baseline_features(data, dic, ratio, correlations, pickles, write):
    if not write:
        if ratio == 0.97:
            new_bf_data = pd.read_pickle("Pickles/Baseline/baseline_97.pkl")
        elif ratio == 0.95:
            new_bf_data = pd.read_pickle("Pickles/Baseline/baseline_95.pkl")
        elif ratio == 0.9:
            new_bf_data = pd.read_pickle("Pickles/Baseline/baseline_90.pkl")
        elif ratio == 0.85:
            new_bf_data = pd.read_pickle("Pickles/Baseline/baseline_85.pkl")
        elif ratio == 0.8:
            new_bf_data = pd.read_pickle("Pickles/Baseline/baseline_80.pkl")

    else:
        bf_data = get_data_from_dic(data, dic, "Baseline Features")

        # Shimmer
        if ratio > 0.89:
            if pickles[0]:
                shimmer = get_data_by_expression(bf_data, "^.*Shimmer")
                lst = list(shimmer.columns)
                lst.remove("apq11Shimmer")
                shimmer = add_variable_from_mean(shimmer, "shimmer_master", lst, True)
                shimmer.to_pickle("Pickles/Baseline/shimmer_+89.pkl")
            else:
                shimmer = pd.read_pickle("Pickles/Baseline/shimmer_+89.pkl")
        else:
            if pickles[0]:
                shimmer = get_data_by_expression(bf_data, "^.*Shimmer")
                shimmer = add_variable_from_mean(
                    shimmer, "shimmer_master", list(shimmer.columns), True
                )
                shimmer.to_pickle("Pickles/Baseline/shimmer_-89.pkl")
            else:
                shimmer = pd.read_pickle("Pickles/Baseline/shimmer_-89.pkl")

        # Jitter
        if pickles[1]:
            jitter = get_data_by_expression(bf_data, "^.*Jitter")
            jitter = add_variable_from_mean(jitter, "jitter_master", list(jitter.columns), True)
            jitter.to_pickle("Pickles/Baseline/jitter.pkl")
        else:
            jitter = pd.read_pickle("Pickles/Baseline/jitter.pkl")

        # Harmonicity
        if ratio > 0.82:
            harmonicity = get_data_by_expression(bf_data, "Harmonicity$")
        else:
            harmonicity = get_data_by_expression(bf_data, "Harmonicity$")
            del harmonicity["meanHarmToNoiseHarmonicity"]

        # Pulses
        pulses = get_data_by_expression(bf_data, "^.*Pulses")
        del pulses["numPeriodsPulses"]

        new_bf_data = data[["PPE", "DFA", "RPDE"]]
        new_bf_data = pd.concat(
            [new_bf_data, pulses, jitter, shimmer, harmonicity], axis=1, sort=False
        )

        if ratio == 0.97:
            new_bf_data.to_pickle("Pickles/Baseline/baseline_97.pkl")
        elif ratio == 0.95:
            new_bf_data.to_pickle("Pickles/Baseline/baseline_95.pkl")
        elif ratio == 0.9:
            new_bf_data.to_pickle("Pickles/Baseline/baseline_90.pkl")
        elif ratio == 0.85:
            new_bf_data.to_pickle("Pickles/Baseline/baseline_85.pkl")
        elif ratio == 0.8:
            new_bf_data.to_pickle("Pickles/Baseline/baseline_80.pkl")

        if correlations:
            new_bf_data = get_data_from_dic(data, dic, "Baseline Features")

            shimmer = get_data_by_expression(new_bf_data, "Shimmer$")
            group_correlation(shimmer)

            jitter = get_data_by_expression(new_bf_data, "Jitter$")
            group_correlation(jitter)

            harmonicity = get_data_by_expression(new_bf_data, "Harmonicity$")
            group_correlation(harmonicity)

            pulses = get_data_by_expression(new_bf_data, "Pulses$")
            group_correlation(pulses)

    return new_bf_data


def bandwidth_parameters(data, dic, correlation, gender_data):
    bp_data = get_data_from_dic(data, dic, "Bandwidth Parameters")
    if correlation:
        group_correlation(bp_data)
    return bp_data


def formant_frequencies(data, dic, correlation):
    ff_data = get_data_from_dic(data, dic, "Formant Frequencies")
    if correlation:
        group_correlation(ff_data)
    return ff_data


def intensity_parameters(data, dic, ratio, correlations, write):
    if not write:
        if ratio == 0.97 or ratio == 0.95:
            ip_data = pd.read_pickle("Pickles/Intensity/intensity_+91.pkl")
        elif ratio == 0.9 or ratio == 0.85 or ratio == 0.8:
            ip_data = pd.read_pickle("Pickles/Intensity/intensity_-91.pkl")
    else:
        if not correlations:
            ip_data = get_data_from_dic(data, dic, "Intensity Parameters")

            if ratio > 0.91:
                del ip_data["maxIntensity"]
            else:
                ip_data = delete_columns(ip_data, ["maxIntensity", "minIntensity"])

            if ratio == 0.97 or ratio == 0.95:
                ip_data.to_pickle("Pickles/Intensity/intensity_+91.pkl")
            elif ratio == 0.9 or ratio == 0.85 or ratio == 0.8:
                ip_data.to_pickle("Pickles/Intensity/intensity_-91.pkl")

        else:
            ip_data = get_data_from_dic(data, dic, "Intensity Parameters")
            group_correlation(ip_data)

    return ip_data


def mfcc(data, dic, ratio, correlations, pickles, write):
    mfcc_data = get_data_from_dic(data, dic, "MFCC ")

    if not write:
        if ratio == 0.97:
            new_mfcc_data = pd.read_pickle("Pickles/MFCC/mfcc_97.pkl")
        elif ratio == 0.95:
            new_mfcc_data = pd.read_pickle("Pickles/MFCC/mfcc_95.pkl")
        elif ratio == 0.9:
            new_mfcc_data = pd.read_pickle("Pickles/MFCC/mfcc_90.pkl")
        elif ratio == 0.85:
            new_mfcc_data = pd.read_pickle("Pickles/MFCC/mfcc_85.pkl")
        elif ratio == 0.8:
            new_mfcc_data = pd.read_pickle("Pickles/MFCC/mfcc_80.pkl")
    else:
        # mean_coef
        if pickles[0]:
            mean_coef = get_data_by_expression(mfcc_data, "^mean_MFCC_.*")
            mean_coef = pd.concat([mfcc_data["mean_Log_energy"], mean_coef], axis=1, sort=False)
            mean_coef.to_pickle("Pickles/MFCC/mean_coef.pkl")
        else:
            mean_coef = pd.read_pickle("Pickles/MFCC/mean_coef.pkl")

        # mean_deltas
        if ratio <= 0.86:
            if pickles[1]:
                mean_delta = get_data_by_expression(mfcc_data, "^mean_(...|....)_delta$")
                mean_delta_delta = get_data_by_expression(
                    mfcc_data, "^mean_(...|....)_delta_delta$"
                )
                mean_deltas = pd.concat(
                    [
                        mfcc_data["mean_delta_log_energy"],
                        mean_delta,
                        mfcc_data["mean_delta_delta_log_energy"],
                        mean_delta_delta,
                    ],
                    axis=1,
                    sort=False,
                )
                lst = ["mean_delta_log_energy", "mean_0th_delta"]
                mean_deltas = add_variable_from_mean(mean_deltas, "mean_delta_master", lst, True)
                mean_deltas.to_pickle("Pickles/MFCC/mean_deltas_-86.pkl")
            else:
                mean_deltas = pd.read_pickle("Pickles/MFCC/mean_deltas_-86.pkl")
        else:
            if pickles[1]:
                mean_delta = get_data_by_expression(mfcc_data, "^mean_(...|....)_delta$")
                mean_delta_delta = get_data_by_expression(
                    mfcc_data, "^mean_(...|....)_delta_delta$"
                )
                mean_deltas = pd.concat(
                    [
                        mfcc_data["mean_delta_log_energy"],
                        mean_delta,
                        mfcc_data["mean_delta_delta_log_energy"],
                        mean_delta_delta,
                    ],
                    axis=1,
                    sort=False,
                )
                mean_deltas.to_pickle("Pickles/MFCC/mean_deltas_+86.pkl")
            else:
                mean_deltas = pd.read_pickle("Pickles/MFCC/mean_deltas_+86.pkl")

        # std_coef
        if pickles[2]:
            std_coef = get_data_by_expression(mfcc_data, "^std_MFCC_.*")
            std_coef = pd.concat([mfcc_data["std_Log_energy"], std_coef], axis=1, sort=False)
            if ratio <= 0.88:
                lst = ["std_Log_energy", "std_MFCC_0th_coef"]
                std_coef = add_variable_from_mean(std_coef, "std_MFCC_master", lst, True)
                std_coef.to_pickle("Pickles/MFCC/std_coef_-88.pkl")
            else:
                std_coef.to_pickle("Pickles/MFCC/std_coef_+88.pkl")
        else:
            if ratio <= 0.88:
                std_coef = pd.read_pickle("Pickles/MFCC/std_coef_-88.pkl")
            else:
                std_coef = pd.read_pickle("Pickles/MFCC/std_coef_+88.pkl")

        # std_deltas
        if pickles[3]:
            std_delta = get_data_by_expression(mfcc_data, "^std_(...|....)_delta$")
            std_deltas = pd.concat(
                [
                    mfcc_data["std_delta_log_energy"],
                    std_delta,
                    mfcc_data["std_delta_delta_log_energy"],
                ],
                axis=1,
                sort=False,
            )

            if ratio > 0.95:
                std_delta_delta = get_data_by_expression(mfcc_data, "^std_(...|....)_delta_delta$")
                std_deltas = pd.concat([std_deltas, std_delta_delta], axis=1, sort=False)
                std_deltas.to_pickle("Pickles/MFCC/std_deltas_+95.pkl")
            if ratio <= 0.95:
                lst = ["std_6th_delta_delta", "std_8th_delta_delta", "std_11th_delta_delta"]
                std_deltas = pd.concat([std_deltas, mfcc_data[lst]], axis=1, sort=False)
                std_deltas.to_pickle("Pickles/MFCC/std_deltas_=95.pkl")
            if ratio <= 0.94:
                lst = ["std_6th_delta_delta", "std_8th_delta_delta"]
                std_deltas = delete_columns(std_deltas, lst)
                std_deltas.to_pickle("Pickles/MFCC/std_deltas_=94.pkl")
            if ratio <= 0.93:
                lst = ["std_11th_delta_delta"]
                std_deltas = delete_columns(std_deltas, lst)
                std_deltas.to_pickle("Pickles/MFCC/std_deltas_=93.pkl")
            if ratio <= 0.92:
                lst = ["std_delta_delta_log_energy"]
                std_deltas = delete_columns(std_deltas, lst)
                std_deltas.to_pickle("Pickles/MFCC/std_deltas_-92.pkl")

        else:
            if ratio > 0.95:
                std_deltas = pd.read_pickle("Pickles/MFCC/std_deltas_+95.pkl")
            if ratio == 0.95:
                std_deltas = pd.read_pickle("Pickles/MFCC/std_deltas_=95.pkl")
            if ratio == 0.94:
                std_deltas = pd.read_pickle("Pickles/MFCC/std_deltas_=94.pkl")
            if ratio == 0.93:
                std_deltas = pd.read_pickle("Pickles/MFCC/std_deltas_=93.pkl")
            if ratio <= 0.92:
                std_deltas = pd.read_pickle("Pickles/MFCC/std_deltas_-92.pkl")

    new_mfcc_data = pd.concat([mean_coef, mean_deltas, std_coef, std_deltas], axis=1, sort=False)

    if ratio == 0.97:
        new_mfcc_data.to_pickle("Pickles/MFCC/mfcc_97.pkl")
    elif ratio == 0.95:
        new_mfcc_data.to_pickle("Pickles/MFCC/mfcc_95.pkl")
    elif ratio == 0.9:
        new_mfcc_data.to_pickle("Pickles/MFCC/mfcc_90.pkl")
    elif ratio == 0.85:
        new_mfcc_data.to_pickle("Pickles/MFCC/mfcc_85.pkl")
    elif ratio == 0.8:
        new_mfcc_data.to_pickle("Pickles/MFCC/mfcc_80.pkl")

    if correlations:
        # std
        std = get_data_by_expression(mfcc_data, "^std_.*")
        group_correlation(std)

        # mean
        mean = get_data_by_expression(mfcc_data, "^mean_.*")
        group_correlation(mean)

        # mean_MFCC
        mean_MFCC = get_data_by_expression(mfcc_data, "^mean_MFCC_.*")
        mean_MFCC = pd.concat([mean_MFCC, mfcc_data["mean_Log_energy"]], axis=1, sort=False)
        mean_MFCC = add_variable_from_mean(
            mean_MFCC, "mean_MFCC_master", list(mean_MFCC.columns), False
        )
        group_correlation(mean_MFCC)

        # mean_deltas
        mean_delta = get_data_by_expression(mfcc_data, "^mean_.*delta$")
        mean_delta = pd.concat(
            [
                mean_delta,
                mfcc_data["mean_delta_log_energy"],
                mfcc_data["mean_delta_delta_log_energy"],
            ],
            axis=1,
            sort=False,
        )
        lst = ["mean_delta_log_energy", "mean_0th_delta"]
        mean_delta = add_variable_from_mean(mean_delta, "mean_delta_master", lst, False)
        group_correlation(mean_delta)

        # std_MFCC
        std_MFCC = get_data_by_expression(mfcc_data, "^std_MFCC_.*")
        std_MFCC = pd.concat([std_MFCC, mfcc_data["std_Log_energy"]], axis=1, sort=False)
        lst = ["std_Log_energy", "std_MFCC_0th_coef"]
        std_MFCC = add_variable_from_mean(std_MFCC, "std_MFCC_master", lst, False)
        group_correlation(std_MFCC)

        # std_deltas
        std_delta = get_data_by_expression(mfcc_data, "^std_.*delta$")
        std_delta = pd.concat(
            [std_delta, mfcc_data["std_delta_log_energy"], mfcc_data["std_delta_delta_log_energy"]],
            axis=1,
            sort=False,
        )
        std_delta = add_variable_from_mean(
            std_delta, "std_delta_master", list(std_delta.columns), False
        )
        group_correlation(std_delta)

        # std_MFCC and mean_MFCC
        mean_std_MFCC = pd.concat([mean_MFCC, std_MFCC], axis=1, sort=False)
        group_correlation(mean_std_MFCC)

        # std_deltas and mean_deltas
        mean_std_deltas = pd.concat([mean_delta, std_delta], axis=1, sort=False)
        group_correlation(mean_std_deltas)

    return new_mfcc_data


def vocal_fold(data, dic, ratio, correlations, pickles, write):
    vf_data = get_data_from_dic(data, dic, "Vocal Fold")

    if not write:
        if ratio == 0.97:
            new_vf_data = pd.read_pickle("Pickles/Vocal/vocal_97.pkl")
        elif ratio == 0.95:
            new_vf_data = pd.read_pickle("Pickles/Vocal/vocal_95.pkl")
        elif ratio == 0.9:
            new_vf_data = pd.read_pickle("Pickles/Vocal/vocal_90.pkl")
        elif ratio == 0.85:
            new_vf_data = pd.read_pickle("Pickles/Vocal/vocal_85.pkl")
        elif ratio == 0.8:
            new_vf_data = pd.read_pickle("Pickles/Vocal/vocal_80.pkl")

    else:
        # GQ
        if pickles[0]:
            gq = get_data_by_expression(vf_data, "^GQ")
            gq.to_pickle("Pickles/Vocal/GQ.pkl")
        else:
            gq = pd.read_pickle("Pickles/Vocal/GQ.pkl")

        # GNE
        if pickles[1]:
            gne = get_data_by_expression(vf_data, "^GNE")
            if ratio <= 0.91:
                lst = ["GNE_SNR_TKEO", "GNE_NSR_TKEO"]
                gne = add_variable_from_mean(gne, "GNE_master", lst, True)
                gne.to_pickle("Pickles/Vocal/GNE_-91.pkl")
            else:
                gne.to_pickle("Pickles/Vocal/GNE_+91.pkl")
        else:
            if ratio <= 0.91:
                gne = pd.read_pickle("Pickles/Vocal/GNE_-91.pkl")
            else:
                gne = pd.read_pickle("Pickles/Vocal/GNE_+91.pkl")

        # VFER
        if pickles[2]:
            vfer = get_data_by_expression(vf_data, "^VFER")
            if ratio <= 0.98:
                vfer = delete_columns(vfer, ["VFER_entropy"])
                vfer.to_pickle("Pickles/Vocal/VFER_-98.pkl")
            else:
                vfer.to_pickle("Pickles/Vocal/VFER_+98.pkl")
        else:
            if ratio <= 0.98:
                vfer = pd.read_pickle("Pickles/Vocal/VFER_-98.pkl")
            else:
                vfer = pd.read_pickle("Pickles/Vocal/VFER_+98.pkl")

        # IMF
        if pickles[3]:
            imf = get_data_by_expression(vf_data, "^IMF")
            if ratio > 0.98:
                imf.to_pickle("Pickles/Vocal/IMF_+98.pkl")
            if ratio <= 0.98:
                imf = delete_columns(imf, ["IMF_NSR_entropy"])
                imf.to_pickle("Pickles/Vocal/IMF_-98.pkl")
            if ratio <= 0.8:
                lst = ["IMF_SNR_entropy", "IMF_SNR_SEO"]
                imf = add_variable_from_mean(imf, "IMF_SNR_master", lst, True)
                imf.to_pickle("Pickles/Vocal/IMF_-80.pkl")
        else:
            if ratio > 0.98:
                imf = pd.read_pickle("Pickles/Vocal/IMF_+98.pkl")
            if ratio <= 0.98 and ratio > 0.8:
                imf = pd.read_pickle("Pickles/Vocal/IMF_-98.pkl")
            if ratio <= 0.8:
                imf = pd.read_pickle("Pickles/Vocal/IMF_-80.pkl")

        new_vf_data = pd.concat([gq, gne, vfer, imf], axis=1, sort=False)

        if ratio == 0.97:
            new_vf_data.to_pickle("Pickles/Vocal/vocal_97.pkl")
        elif ratio == 0.95:
            new_vf_data.to_pickle("Pickles/Vocal/vocal_95.pkl")
        elif ratio == 0.9:
            new_vf_data.to_pickle("Pickles/Vocal/vocal_90.pkl")
        elif ratio == 0.85:
            new_vf_data.to_pickle("Pickles/Vocal/vocal_85.pkl")
        elif ratio == 0.8:
            new_vf_data.to_pickle("Pickles/Vocal/vocal_80.pkl")

    if correlations:
        # GQ
        gq = get_data_by_expression(vf_data, "^GQ")
        group_correlation(gq)

        # GNE
        gne = get_data_by_expression(vf_data, "^GNE")
        lst = ["GNE_SNR_TKEO", "GNE_NSR_TKEO"]
        gne = add_variable_from_mean(gne, "GNE_master", lst, False)
        group_correlation(gne)

        # VFER
        vfer = get_data_by_expression(vf_data, "^VFER")
        group_correlation(vfer)

        # IMF
        imf = get_data_by_expression(vf_data, "^IMF")
        lst = ["IMF_SNR_entropy", "IMF_SNR_SEO"]
        imf = add_variable_from_mean(imf, "IMF_master", lst, False)
        group_correlation(imf)

    return new_vf_data


def wavelet_features(data, dic, ratio, correlations, pickles, write):
    wf_data = get_data_from_dic(data, dic, "Wavelet Features")

    if not write:
        if ratio == 0.97:
            new_wf_data = pd.read_pickle("Pickles/Wavelet/wavelet_97.pkl")
        elif ratio == 0.95:
            new_wf_data = pd.read_pickle("Pickles/Wavelet/wavelet_95.pkl")
        elif ratio == 0.9:
            new_wf_data = pd.read_pickle("Pickles/Wavelet/wavelet_90.pkl")
        elif ratio == 0.85:
            new_wf_data = pd.read_pickle("Pickles/Wavelet/wavelet_85.pkl")
        elif ratio == 0.8:
            new_wf_data = pd.read_pickle("Pickles/Wavelet/wavelet_80.pkl")
    else:
        # Ea
        if pickles[0]:
            ea = get_data_by_expression(wf_data, "^Ea")
            if ratio <= 0.82:
                ea = delete_columns(ea, ["Ea2"])
                ea.to_pickle("Pickles/Wavelet/Ea/ea_-82.pkl")
            else:
                ea.to_pickle("Pickles/Wavelet/Ea/ea_+82.pkl")
        else:
            if ratio <= 0.82:
                ea = pd.read_pickle("Pickles/Wavelet/Ea/ea_-82.pkl")
            else:
                ea = pd.read_pickle("Pickles/Wavelet/Ea/ea_+82.pkl")

        # Ed
        if pickles[1]:
            ed = get_data_by_expression(wf_data, "^Ed")
            if ratio > 0.95:
                ed.to_pickle("Pickles/Wavelet/Ed/ed_+95.pkl")
            if ratio <= 0.95:
                ed = delete_columns(ed, ["Ed2_1_coef", "Ed2_2_coef"])
                ed.to_pickle("Pickles/Wavelet/Ed/ed_-95.pkl")
            if ratio <= 0.93:
                ed = delete_columns(ed, ["Ed2_3_coef", "Ed2_5_coef", "Ed_1_coef"])
                ed.to_pickle("Pickles/Wavelet/Ed/ed_-93.pkl")
            if ratio <= 0.89:
                ed = delete_columns(ed, ["Ed_3_coef"])
                ed.to_pickle("Pickles/Wavelet/Ed/ed_-89.pkl")
            if ratio <= 0.87:
                ed = delete_columns(ed, ["Ed2_9_coef"])
                ed.to_pickle("Pickles/Wavelet/Ed/ed_-87.pkl")
            if ratio <= 0.86:
                ed = delete_columns(ed, ["Ed_5_coef", "Ed2_4_coef"])
                ed.to_pickle("Pickles/Wavelet/Ed/ed_-86.pkl")
            if ratio <= 0.85:
                ed = delete_columns(ed, ["Ed_8_coef"])
                ed.to_pickle("Pickles/Wavelet/Ed/ed_-85.pkl")
            if ratio <= 0.84:
                ed = delete_columns(ed, ["Ed_9_coef"])
                ed.to_pickle("Pickles/Wavelet/Ed/ed_-84.pkl")
            if ratio <= 0.82:
                ed = delete_columns(ed, ["Ed_10_coef"])
                ed.to_pickle("Pickles/Wavelet/Ed/ed_-82.pkl")
            if ratio <= 0.81:
                ed = delete_columns(ed, ["Ed_6_coef"])
                ed.to_pickle("Pickles/Wavelet/Ed/ed_-81.pkl")
        else:
            if ratio > 0.95:
                ed = pd.read_pickle("Pickles/Wavelet/Ed/ed_+95.pkl")
            elif ratio in [0.94, 0.95]:
                ed = pd.read_pickle("Pickles/Wavelet/Ed/ed_-95.pkl")
            elif ratio > 0.89 and ratio <= 0.93:
                ed = pd.read_pickle("Pickles/Wavelet/Ed/ed_-93.pkl")
            elif ratio in [0.88, 0.89]:
                ed = pd.read_pickle("Pickles/Wavelet/Ed/ed_-89.pkl")
            elif ratio == 0.87:
                ed = pd.read_pickle("Pickles/Wavelet/Ed/ed_-87.pkl")
            elif ratio == 0.86:
                ed = pd.read_pickle("Pickles/Wavelet/Ed/ed_-86.pkl")
            elif ratio == 0.85:
                ed = pd.read_pickle("Pickles/Wavelet/Ed/ed_-85.pkl")
            elif ratio in [0.84, 0.83]:
                ed = pd.read_pickle("Pickles/Wavelet/Ed/ed_-84.pkl")
            elif ratio == 0.82:
                ed = pd.read_pickle("Pickles/Wavelet/Ed/ed_-82.pkl")
            elif ratio <= 0.81:
                ed = pd.read_pickle("Pickles/Wavelet/Ed/ed_-81.pkl")

        # det_entropy_shannon
        if pickles[2]:
            det_entropy_shannon = get_data_by_expression(wf_data, "^det_entropy_shannon")
            if ratio > 0.91:
                det_entropy_shannon.to_pickle(
                    "Pickles/Wavelet/det_entropy/shannon/det_entropy_shannon_+91.pkl"
                )
            if ratio <= 0.91:
                det_entropy_shannon = delete_columns(
                    det_entropy_shannon, ["det_entropy_shannon_2_coef"]
                )
                det_entropy_shannon.to_pickle(
                    "Pickles/Wavelet/det_entropy/shannon/det_entropy_shannon_-91.pkl"
                )
            if ratio <= 0.86:
                det_entropy_shannon = delete_columns(
                    det_entropy_shannon, ["det_entropy_shannon_10_coef"]
                )
                det_entropy_shannon.to_pickle(
                    "Pickles/Wavelet/det_entropy/shannon/det_entropy_shannon_-86.pkl"
                )
        else:
            if ratio > 0.91:
                det_entropy_shannon = pd.read_pickle(
                    "Pickles/Wavelet/det_entropy/shannon/det_entropy_shannon_+91.pkl"
                )
            elif ratio <= 0.91 and ratio > 0.86:
                det_entropy_shannon = pd.read_pickle(
                    "Pickles/Wavelet/det_entropy/shannon/det_entropy_shannon_-91.pkl"
                )
            elif ratio <= 0.86:
                det_entropy_shannon = pd.read_pickle(
                    "Pickles/Wavelet/det_entropy/shannon/det_entropy_shannon_-86.pkl"
                )

        # det_entropy_log
        if pickles[3]:
            det_entropy_log = get_data_by_expression(wf_data, "^det_entropy_log")
            if ratio > 0.96:
                det_entropy_log.to_pickle("Pickles/Wavelet/det_entropy/log/det_entropy_log_+96.pkl")
            if ratio <= 0.96:
                det_entropy_log = delete_columns(det_entropy_log, ["det_entropy_log_9_coef"])
                det_entropy_log.to_pickle("Pickles/Wavelet/det_entropy/log/det_entropy_log_-96.pkl")
            if ratio <= 0.94:
                det_entropy_log = delete_columns(det_entropy_log, ["det_entropy_log_2_coef"])
                det_entropy_log.to_pickle("Pickles/Wavelet/det_entropy/log/det_entropy_log_-94.pkl")
            if ratio <= 0.93:
                det_entropy_log = delete_columns(det_entropy_log, ["det_entropy_log_7_coef"])
                det_entropy_log.to_pickle("Pickles/Wavelet/det_entropy/log/det_entropy_log_-93.pkl")
            if ratio <= 0.91:
                det_entropy_log = delete_columns(det_entropy_log, ["det_entropy_log_5_coef"])
                det_entropy_log.to_pickle("Pickles/Wavelet/det_entropy/log/det_entropy_log_-91.pkl")
            if ratio <= 0.86:
                det_entropy_log = delete_columns(det_entropy_log, ["det_entropy_log_4_coef"])
                det_entropy_log.to_pickle("Pickles/Wavelet/det_entropy/log/det_entropy_log_-86.pkl")
            if ratio <= 0.85:
                det_entropy_log = delete_columns(det_entropy_log, ["det_entropy_log_8_coef"])
                det_entropy_log.to_pickle("Pickles/Wavelet/det_entropy/log/det_entropy_log_-85.pkl")
        else:
            if ratio > 0.96:
                det_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/det_entropy/log/det_entropy_log_+96.pkl"
                )
            elif ratio in [0.95, 0.96]:
                det_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/det_entropy/log/det_entropy_log_-96.pkl"
                )
            elif ratio == 0.94:
                det_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/det_entropy/log/det_entropy_log_-94.pkl"
                )
            elif ratio in [0.92, 0.93]:
                det_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/det_entropy/log/det_entropy_log_-93.pkl"
                )
            elif ratio <= 0.91 and ratio > 0.86:
                det_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/det_entropy/log/det_entropy_log_-91.pkl"
                )
            elif ratio == 0.86:
                det_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/det_entropy/log/det_entropy_log_-86.pkl"
                )
            elif ratio <= 0.85:
                det_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/det_entropy/log/det_entropy_log_-85.pkl"
                )

        # det_TKEO
        if pickles[4]:
            det_TKEO = get_data_by_expression(wf_data, "^det_TKEO")
            det_TKEO = delete_columns(
                det_TKEO, ["det_TKEO_std_7_coef", "det_TKEO_std_8_coef", "det_TKEO_std_10_coef"]
            )
            if ratio > 0.98:
                det_TKEO.to_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_+98.pkl")
            if ratio <= 0.98:
                det_TKEO = delete_columns(det_TKEO, ["det_TKEO_std_9_coef"])
                det_TKEO.to_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_-98.pkl")
            if ratio <= 0.97:
                det_TKEO = delete_columns(det_TKEO, ["det_TKEO_std_2_coef"])
                det_TKEO.to_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_-97.pkl")
            if ratio <= 0.96:
                det_TKEO = delete_columns(det_TKEO, ["det_TKEO_std_4_coef", "det_TKEO_std_5_coef"])
                det_TKEO.to_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_-96.pkl")
            if ratio <= 0.94:
                det_TKEO = delete_columns(det_TKEO, ["det_TKEO_std_1_coef", "det_TKEO_std_6_coef"])
                det_TKEO.to_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_-94.pkl")
            if ratio <= 0.92:
                det_TKEO = delete_columns(det_TKEO, ["det_TKEO_mean_2_coef"])
                det_TKEO.to_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_-92.pkl")
            if ratio <= 0.91:
                det_TKEO = delete_columns(det_TKEO, ["det_TKEO_std_3_coef"])
                det_TKEO.to_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_-91.pkl")
            if ratio <= 0.83:
                det_TKEO = delete_columns(det_TKEO, ["det_TKEO_mean_10_coef"])
                det_TKEO.to_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_-83.pkl")
        else:
            if ratio > 0.98:
                det_TKEO = pd.read_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_+98.pkl")
            elif ratio == 0.98:
                det_TKEO = pd.read_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_-98.pkl")
            elif ratio == 0.97:
                det_TKEO = pd.read_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_-97.pkl")
            elif ratio in [0.95, 0.96]:
                det_TKEO = pd.read_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_-96.pkl")
            elif ratio in [0.93, 0.94]:
                det_TKEO = pd.read_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_-94.pkl")
            elif ratio == 0.92:
                det_TKEO = pd.read_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_-92.pkl")
            elif ratio <= 0.91 and ratio > 0.83:
                det_TKEO = pd.read_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_-91.pkl")
            elif ratio <= 0.83:
                det_TKEO = pd.read_pickle("Pickles/Wavelet/det_TKEO/det_TKEO_-83.pkl")

        # app_entropy_shannon
        if pickles[5]:
            app_entropy_shannon = get_data_by_expression(wf_data, "^app_entropy_shannon_4_coef")
            app_entropy_shannon.to_pickle("Pickles/Wavelet/app_entropy/app_entropy_shannon.pkl")
        else:
            app_entropy_shannon = pd.read_pickle(
                "Pickles/Wavelet/app_entropy/app_entropy_shannon.pkl"
            )

        # app_entropy_log
        if pickles[6]:
            app_entropy_log_1 = get_data_by_expression(wf_data, "^app_entropy_log_3_coef")
            app_entropy_log_2 = get_data_by_expression(wf_data, "^app_entropy_log_7_coef")
            app_entropy_log = pd.concat([app_entropy_log_1, app_entropy_log_2], axis=1, sort=False)
            if ratio > 0.88:
                app_entropy_log.to_pickle("Pickles/Wavelet/app_entropy/app_entropy_log_+88.pkl")
            else:
                app_entropy_log = delete_columns(app_entropy_log, ["app_entropy_log_3_coef"])
                app_entropy_log.to_pickle("Pickles/Wavelet/app_entropy/app_entropy_log_-88.pkl")
        else:
            if ratio > 0.88:
                app_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/app_entropy/app_entropy_log_+88.pkl"
                )
            else:
                app_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/app_entropy/app_entropy_log_-88.pkl"
                )

        # app_det_TKEO
        if pickles[7]:
            app_det_TKEO_1 = get_data_by_expression(wf_data, "^app_det_TKEO_mean_1_coef")
            app_det_TKEO_2 = get_data_by_expression(wf_data, "^app_det_TKEO_mean_2_coef")
            app_det_TKEO_3 = get_data_by_expression(wf_data, "^app_det_TKEO_mean_3_coef")
            app_det_TKEO_6 = get_data_by_expression(wf_data, "^app_det_TKEO_mean_6_coef")
            app_det_TKEO = pd.concat(
                [app_det_TKEO_1, app_det_TKEO_2, app_det_TKEO_3, app_det_TKEO_6], axis=1, sort=False
            )

            if ratio > 0.96:
                app_det_TKEO.to_pickle("Pickles/Wavelet/app_det_TKEO/app_det_TKEO_+96.pkl")
            if ratio <= 0.96:
                app_det_TKEO = delete_columns(app_det_TKEO, ["app_det_TKEO_mean_3_coef"])
                app_det_TKEO.to_pickle("Pickles/Wavelet/app_det_TKEO/app_det_TKEO_-96.pkl")
            if ratio <= 0.92:
                app_det_TKEO = delete_columns(app_det_TKEO, ["app_det_TKEO_mean_2_coef"])
                app_det_TKEO.to_pickle("Pickles/Wavelet/app_det_TKEO/app_det_TKEO_-92.pkl")
        else:
            if ratio > 0.96:
                app_det_TKEO = pd.read_pickle("Pickles/Wavelet/app_det_TKEO/app_det_TKEO_+96.pkl")
            elif ratio <= 0.96 and ratio > 0.92:
                app_det_TKEO = pd.read_pickle("Pickles/Wavelet/app_det_TKEO/app_det_TKEO_-96.pkl")
            elif ratio <= 0.92:
                app_det_TKEO = pd.read_pickle("Pickles/Wavelet/app_det_TKEO/app_det_TKEO_-92.pkl")

        # app_TKEO_std
        if pickles[8]:
            app_TKEO_std_1 = get_data_by_expression(wf_data, "^app_TKEO_std_1_coef")
            app_TKEO_std_2 = get_data_by_expression(wf_data, "^app_TKEO_std_2_coef")
            app_TKEO_std_3 = get_data_by_expression(wf_data, "^app_TKEO_std_3_coef")
            app_TKEO_std_6 = get_data_by_expression(wf_data, "^app_TKEO_std_6_coef")
            app_TKEO_std = pd.concat(
                [app_TKEO_std_1, app_TKEO_std_2, app_TKEO_std_3, app_TKEO_std_6], axis=1, sort=False
            )

            if ratio > 0.97:
                app_TKEO_std.to_pickle("Pickles/Wavelet/app_TKEO_std/app_TKEO_std_+97.pkl")
            if ratio <= 0.97:
                app_TKEO_std = delete_columns(app_TKEO_std, ["app_TKEO_std_3_coef"])
                app_TKEO_std.to_pickle("Pickles/Wavelet/app_TKEO_std/app_TKEO_std_-97.pkl")
            if ratio <= 0.94:
                app_TKEO_std = delete_columns(app_TKEO_std, ["app_TKEO_std_2_coef"])
                app_TKEO_std.to_pickle("Pickles/Wavelet/app_TKEO_std/app_TKEO_std_-94.pkl")
        else:
            if ratio > 0.97:
                app_TKEO_std = pd.read_pickle("Pickles/Wavelet/app_TKEO_std/app_TKEO_std_+97.pkl")
            elif ratio <= 0.97 and ratio > 0.94:
                app_TKEO_std = pd.read_pickle("Pickles/Wavelet/app_TKEO_std/app_TKEO_std_-97.pkl")
            elif ratio <= 0.94:
                app_TKEO_std = pd.read_pickle("Pickles/Wavelet/app_TKEO_std/app_TKEO_std_-94.pkl")

        # det_LT_entropy_shannon
        if pickles[9]:
            det_LT_entropy_shannon = get_data_by_expression(wf_data, "^det_LT_entropy_shannon")
            det_LT_entropy_shannon.to_pickle(
                "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_shannon.pkl"
            )
        else:
            det_LT_entropy_shannon = pd.read_pickle(
                "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_shannon.pkl"
            )

        # det_LT_entropy_log
        if pickles[10]:
            det_LT_entropy_log = get_data_by_expression(wf_data, "^det_LT_entropy_log")

            if ratio > 0.96:
                det_LT_entropy_log.to_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_+96.pkl"
                )
            if ratio <= 0.96:
                det_LT_entropy_log = delete_columns(
                    det_LT_entropy_log, ["det_LT_entropy_log_9_coef"]
                )
                det_LT_entropy_log.to_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_-96.pkl"
                )
            if ratio <= 0.94:
                det_LT_entropy_log = delete_columns(
                    det_LT_entropy_log, ["det_LT_entropy_log_2_coef"]
                )
                det_LT_entropy_log.to_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_-94.pkl"
                )
            if ratio <= 0.93:
                det_LT_entropy_log = delete_columns(
                    det_LT_entropy_log, ["det_LT_entropy_log_7_coef"]
                )
                det_LT_entropy_log.to_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_-93.pkl"
                )
            if ratio <= 0.91:
                det_LT_entropy_log = delete_columns(
                    det_LT_entropy_log, ["det_LT_entropy_log_5_coef"]
                )
                det_LT_entropy_log.to_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_-91.pkl"
                )
            if ratio <= 0.87:
                det_LT_entropy_log = delete_columns(
                    det_LT_entropy_log, ["det_LT_entropy_log_4_coef"]
                )
                det_LT_entropy_log.to_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_-87.pkl"
                )
            if ratio <= 0.85:
                det_LT_entropy_log = delete_columns(
                    det_LT_entropy_log, ["det_LT_entropy_log_8_coef"]
                )
                det_LT_entropy_log.to_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_-85.pkl"
                )
            if ratio <= 0.8:
                det_LT_entropy_log = delete_columns(
                    det_LT_entropy_log, ["det_LT_entropy_log_3_coef"]
                )
                det_LT_entropy_log.to_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_-80.pkl"
                )

        else:
            if ratio > 0.96:
                det_LT_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_+96.pkl"
                )
            if ratio in [0.95, 0.96]:
                det_LT_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_-96.pkl"
                )
            if ratio == 0.94:
                det_LT_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_-94.pkl"
                )
            if ratio in [0.93, 0.92]:
                det_LT_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_-93.pkl"
                )
            if ratio <= 0.91 and ratio > 0.87:
                det_LT_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_-91.pkl"
                )
            if ratio in [0.86, 0.87]:
                det_LT_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_-87.pkl"
                )
            if ratio <= 0.85 and ratio > 0.8:
                det_LT_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_-85.pkl"
                )
            if ratio <= 0.8:
                det_LT_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/det_LT_entropy/det_LT_entropy_log_-80.pkl"
                )

        # det_LT_TKEO
        if pickles[11]:
            det_LT_TKEO = get_data_by_expression(wf_data, "^det_LT_TKEO")
            det_LT_TKEO = delete_columns(
                det_LT_TKEO,
                [
                    "det_LT_TKEO_std_7_coef",
                    "det_LT_TKEO_std_8_coef",
                    "det_LT_TKEO_std_9_coef",
                    "det_LT_TKEO_std_10_coef",
                ],
            )

            if ratio > 0.97:
                det_LT_TKEO.to_pickle("Pickles/Wavelet/det_LT_TKEO/det_LT_TKEO_+97.pkl")
            if ratio <= 0.97:
                det_LT_TKEO = delete_columns(det_LT_TKEO, ["det_LT_TKEO_std_2_coef"])
                det_LT_TKEO.to_pickle("Pickles/Wavelet/det_LT_TKEO/det_LT_TKEO_-97.pkl")
            if ratio <= 0.95:
                det_LT_TKEO = delete_columns(
                    det_LT_TKEO, ["det_LT_TKEO_std_5_coef", "det_LT_TKEO_std_6_coef"]
                )
                det_LT_TKEO.to_pickle("Pickles/Wavelet/det_LT_TKEO/det_LT_TKEO_-95.pkl")
            if ratio <= 0.94:
                det_LT_TKEO = delete_columns(
                    det_LT_TKEO, ["det_LT_TKEO_std_1_coef", "det_LT_TKEO_std_4_coef"]
                )
                det_LT_TKEO.to_pickle("Pickles/Wavelet/det_LT_TKEO/det_LT_TKEO_-94.pkl")
            if ratio <= 0.91:
                det_LT_TKEO = delete_columns(det_LT_TKEO, ["det_LT_TKEO_std_3_coef"])
                det_LT_TKEO.to_pickle("Pickles/Wavelet/det_LT_TKEO/det_LT_TKEO_-91.pkl")
            if ratio <= 0.9:
                det_LT_TKEO = delete_columns(det_LT_TKEO, ["det_LT_TKEO_mean_2_coef"])
                det_LT_TKEO.to_pickle("Pickles/Wavelet/det_LT_TKEO/det_LT_TKEO_-90.pkl")
        else:
            if ratio > 0.97:
                det_LT_TKEO = pd.read_pickle("Pickles/Wavelet/det_LT_TKEO/det_LT_TKEO_+97.pkl")
            elif ratio in [0.96, 0.97]:
                det_LT_TKEO = pd.read_pickle("Pickles/Wavelet/det_LT_TKEO/det_LT_TKEO_-97.pkl")
            elif ratio == 0.95:
                det_LT_TKEO = pd.read_pickle("Pickles/Wavelet/det_LT_TKEO/det_LT_TKEO_-95.pkl")
            elif ratio <= 0.94 and ratio > 0.91:
                det_LT_TKEO = pd.read_pickle("Pickles/Wavelet/det_LT_TKEO/det_LT_TKEO_-94.pkl")
            elif ratio == 0.91:
                det_LT_TKEO = pd.read_pickle("Pickles/Wavelet/det_LT_TKEO/det_LT_TKEO_-91.pkl")
            elif ratio <= 0.9:
                det_LT_TKEO = pd.read_pickle("Pickles/Wavelet/det_LT_TKEO/det_LT_TKEO_-90.pkl")

        # app_LT_entropy_shannon
        if pickles[12]:
            app_LT_entropy_shannon_3 = get_data_by_expression(
                wf_data, "^app_LT_entropy_shannon_3_coef"
            )
            app_LT_entropy_shannon_6 = get_data_by_expression(
                wf_data, "^app_LT_entropy_shannon_6_coef"
            )
            app_LT_entropy_shannon = pd.concat(
                [app_LT_entropy_shannon_3, app_LT_entropy_shannon_6], axis=1, sort=False
            )
            app_LT_entropy_shannon.to_pickle(
                "Pickles/Wavelet/app_LT_entropy/app_LT_entropy_shannon.pkl"
            )
        else:
            app_LT_entropy_shannon = pd.read_pickle(
                "Pickles/Wavelet/app_LT_entropy/app_LT_entropy_shannon.pkl"
            )

        # app_LT_entropy_log
        if pickles[13]:
            app_LT_entropy_log_1 = get_data_by_expression(wf_data, "^app_LT_entropy_log_1_coef")
            app_LT_entropy_log_6 = get_data_by_expression(wf_data, "^app_LT_entropy_log_6_coef")
            app_LT_entropy_log_10 = get_data_by_expression(wf_data, "^app_LT_entropy_log_10_coef")
            app_LT_entropy_log = pd.concat(
                [app_LT_entropy_log_1, app_LT_entropy_log_6, app_LT_entropy_log_10],
                axis=1,
                sort=False,
            )

            if ratio > 0.88:
                app_LT_entropy_log.to_pickle(
                    "Pickles/Wavelet/app_LT_entropy/app_LT_entropy_log_+88.pkl"
                )
            else:
                app_LT_entropy_log = delete_columns(
                    app_LT_entropy_log, ["app_LT_entropy_log_6_coef"]
                )
                app_LT_entropy_log.to_pickle(
                    "Pickles/Wavelet/app_LT_entropy/app_LT_entropy_log_-88.pkl"
                )
        else:
            if ratio > 0.88:
                app_LT_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/app_LT_entropy/app_LT_entropy_log_+88.pkl"
                )
            else:
                app_LT_entropy_log = pd.read_pickle(
                    "Pickles/Wavelet/app_LT_entropy/app_LT_entropy_log_-88.pkl"
                )

        # app_LT_TKEO
        if pickles[14]:
            app_LT_TKEO_1 = get_data_by_expression(wf_data, "^app_LT_TKEO_mean_1_coef")
            app_LT_TKEO_3 = get_data_by_expression(wf_data, "^app_LT_TKEO_mean_3_coef")
            app_LT_TKEO_5 = get_data_by_expression(wf_data, "^app_LT_TKEO_mean_6_coef")
            app_LT_TKEO_6 = get_data_by_expression(wf_data, "^app_LT_TKEO_std_1_coef")
            app_LT_TKEO_8 = get_data_by_expression(wf_data, "^app_LT_TKEO_std_3_coef")
            app_LT_TKEO = pd.concat(
                [app_LT_TKEO_1, app_LT_TKEO_3, app_LT_TKEO_5, app_LT_TKEO_6, app_LT_TKEO_8],
                axis=1,
                sort=False,
            )

            if ratio > 0.97:
                app_LT_TKEO.to_pickle("Pickles/Wavelet/app_LT_TKEO/app_LT_TKEO_+97.pkl")
            if ratio <= 0.97:
                app_LT_TKEO = delete_columns(app_LT_TKEO, ["app_LT_TKEO_std_3_coef"])
                app_LT_TKEO.to_pickle("Pickles/Wavelet/app_LT_TKEO/app_LT_TKEO_-97.pkl")
            if ratio <= 0.93:
                app_LT_TKEO = delete_columns(app_LT_TKEO, ["app_LT_TKEO_mean_3_coef"])
                app_LT_TKEO.to_pickle("Pickles/Wavelet/app_LT_TKEO/app_LT_TKEO_-93.pkl")
            if ratio <= 0.88:
                app_LT_TKEO = delete_columns(app_LT_TKEO, ["app_LT_TKEO_std_1_coef"])
                app_LT_TKEO.to_pickle("Pickles/Wavelet/app_LT_TKEO/app_LT_TKEO_-88.pkl")
        else:
            if ratio > 0.97:
                app_LT_TKEO = pd.read_pickle("Pickles/Wavelet/app_LT_TKEO/app_LT_TKEO_+97.pkl")
            elif ratio <= 0.97 and ratio > 0.93:
                app_LT_TKEO = pd.read_pickle("Pickles/Wavelet/app_LT_TKEO/app_LT_TKEO_-97.pkl")
            elif ratio <= 0.93 and ratio > 0.88:
                app_LT_TKEO = pd.read_pickle("Pickles/Wavelet/app_LT_TKEO/app_LT_TKEO_-93.pkl")
            elif ratio <= 0.88:
                app_LT_TKEO = pd.read_pickle("Pickles/Wavelet/app_LT_TKEO/app_LT_TKEO_-88.pkl")

        new_wf_data = pd.concat(
            [
                ea,
                ed,
                det_entropy_shannon,
                det_entropy_log,
                det_TKEO,
                app_entropy_shannon,
                app_entropy_log,
                app_det_TKEO,
                app_TKEO_std,
                det_LT_entropy_shannon,
                det_LT_entropy_log,
                det_LT_TKEO,
                app_LT_entropy_shannon,
                app_LT_entropy_log,
                app_LT_TKEO,
            ],
            axis=1,
            sort=False,
        )

        if ratio == 0.97:
            new_wf_data.to_pickle("Pickles/Wavelet/wavelet_97.pkl")
        elif ratio == 0.95:
            new_wf_data.to_pickle("Pickles/Wavelet/wavelet_95.pkl")
        elif ratio == 0.9:
            new_wf_data.to_pickle("Pickles/Wavelet/wavelet_90.pkl")
        elif ratio == 0.85:
            new_wf_data.to_pickle("Pickles/Wavelet/wavelet_85.pkl")
        elif ratio == 0.8:
            new_wf_data.to_pickle("Pickles/Wavelet/wavelet_80.pkl")

    if correlations:
        # Ea
        ea = get_data_by_expression(wf_data, "^Ea")
        group_correlation(ea)

        # Ed
        ed = get_data_by_expression(wf_data, "^Ed")
        ed = add_variable_from_mean(ed, "ed_master", list(ed.columns), False)
        group_correlation(ed)

        # det_entropy
        det_entropy = get_data_by_expression(wf_data, "^det_entropy")
        group_correlation(det_entropy)

        # det_entropy_shannon
        det_entropy_shannon = get_data_by_expression(wf_data, "^det_entropy_shannon")
        group_correlation(det_entropy_shannon)

        # det_entropy_log
        det_entropy_log = get_data_by_expression(wf_data, "^det_entropy_log")
        group_correlation(det_entropy_log)

        # det_TKEO
        det_TKEO = get_data_by_expression(wf_data, "^det_TKEO")
        group_correlation(det_TKEO)

        # det_TKEO_mean
        det_TKEO_mean = get_data_by_expression(wf_data, "^det_TKEO_mean")
        group_correlation(det_TKEO_mean)

        # det_TKEO_std
        det_TKEO_std = get_data_by_expression(wf_data, "^det_TKEO_std")
        group_correlation(det_TKEO_std)

        # app_entropy
        app_entropy = get_data_by_expression(wf_data, "^app_entropy")
        group_correlation(app_entropy)

        # app_entropy_shannon
        app_entropy_shannon = get_data_by_expression(wf_data, "^app_entropy_shannon")
        group_correlation(app_entropy_shannon)

        # app_entropy_log
        app_entropy_log = get_data_by_expression(wf_data, "^app_entropy_log")
        group_correlation(app_entropy_log)

        # app_det_TKEO
        app_det_TKEO = get_data_by_expression(wf_data, "^app_det_TKEO")
        group_correlation(app_det_TKEO)

        # app_TKEO
        app_TKEO = get_data_by_expression(wf_data, "^app_TKEO")
        group_correlation(app_TKEO)

        # det_LT_entropy
        det_LT_entropy = get_data_by_expression(wf_data, "^det_LT_entropy")
        group_correlation(det_LT_entropy)

        # det_LT_entropy_shannon
        det_LT_entropy_shannon = get_data_by_expression(wf_data, "^det_LT_entropy_shannon")
        group_correlation(det_LT_entropy_shannon)

        # det_LT_entropy_log
        det_LT_entropy_log = get_data_by_expression(wf_data, "^det_LT_entropy_log")
        group_correlation(det_LT_entropy_log)

        # det_LT_TKEO
        det_LT_TKEO = get_data_by_expression(wf_data, "^det_LT_TKEO")
        group_correlation(det_LT_TKEO)

        # det_LT_TKEO_mean
        det_LT_TKEO_mean = get_data_by_expression(wf_data, "^det_LT_TKEO_mean")
        group_correlation(det_LT_TKEO_mean)

        # det_LT_TKEO_std
        det_LT_TKEO_std = get_data_by_expression(wf_data, "^det_LT_TKEO_std")
        group_correlation(det_LT_TKEO_std)

        # app_LT_entropy
        app_LT_entropy = get_data_by_expression(wf_data, "^app_LT_entropy")
        group_correlation(app_LT_entropy)

        # app_LT_entropy_shannon
        app_LT_entropy_shannon = get_data_by_expression(wf_data, "^app_LT_entropy_shannon")
        group_correlation(app_LT_entropy_shannon)

        # app_LT_entropy_log
        app_LT_entropy_log = get_data_by_expression(wf_data, "^app_LT_entropy_log")
        group_correlation(app_LT_entropy_log)

        # app_LT_TKEO
        app_LT_TKEO = get_data_by_expression(wf_data, "^app_LT_TKEO")
        group_correlation(app_LT_TKEO)

        # app_LT_TKEO_mean
        app_LT_TKEO_mean = get_data_by_expression(wf_data, "^app_LT_TKEO_mean")
        group_correlation(app_LT_TKEO_mean)

        # app_LT_TKEO_std
        app_LT_TKEO_std = get_data_by_expression(wf_data, "^app_LT_TKEO_std")
        group_correlation(app_LT_TKEO_std)

    return new_wf_data


def tqwt_features(data, dic):

    tqwt_data = get_data_from_dic(data, dic, "TQWT Features")

    """
	# groups of data
	tqwt_energy = get_data_by_expression(tqwt_data, "^tqwt_energy.*")
	tqwt_entropy_shannon = get_data_by_expression(tqwt_data, "^tqwt_entropy_shannon.*")
	tqwt_entropy_log = get_data_by_expression(tqwt_data, "^tqwt_entropy_log.*")
	tqwt_TKEO_mean = get_data_by_expression(tqwt_data, "^tqwt_TKEO_mean.*")
	tqwt_TKEO_std = get_data_by_expression(tqwt_data, "^tqwt_TKEO_std.*")
	tqwt_medianValue = get_data_by_expression(tqwt_data, "^tqwt_medianValue.*")
	tqwt_meanValue = get_data_by_expression(tqwt_data, "^tqwt_meanValue.*")
	tqwt_stdValue = get_data_by_expression(tqwt_data, "^tqwt_stdValue.*")
	tqwt_minValue = get_data_by_expression(tqwt_data, "^tqwt_minValue.*")
	tqwt_maxValue = get_data_by_expression(tqwt_data, "^tqwt_maxValue.*")
	tqwt_skewnessValue = get_data_by_expression(tqwt_data, "^tqwt_skewnessValue.*")
	tqwt_kurtosisValue = get_data_by_expression(tqwt_data, "^tqwt_kurtosisValue.*")


	# produce list of variables for a group
	tqwt_energy_lst = produce_allvariables("tqwt_energy_dec_",37)
	tqwt_entropy_shannon_lst = produce_allvariables("tqwt_entropy_shannon_dec_",37)
	tqwt_entropy_log_lst = produce_allvariables("tqwt_entropy_log_dec_",37)
	tqwt_TKEO_mean_lst = produce_allvariables("tqwt_TKEO_mean_dec_",37)
	tqwt_TKEO_std_lst = produce_allvariables("tqwt_TKEO_std_dec_",37)
	tqwt_medianValue_lst = produce_allvariables("tqwt_medianValue_dec_",37)
	tqwt_meanValue_lst = produce_allvariables("tqwt_meanValue_dec_",37)
	tqwt_stdValue_lst = produce_allvariables("tqwt_stdValue_dec_",37)
	tqwt_minValue_lst = produce_allvariables("tqwt_minValue_dec_",37)
	tqwt_maxValue_lst = produce_allvariables("tqwt_maxValue_dec_",37)
	tqwt_skewnessValue_lst = produce_allvariables("tqwt_skewnessValue_dec_",37)
	tqwt_kurtosisValue_lst = produce_allvariables("tqwt_kurtosisValue_dec_",37)



	# add new variable in a dataset that represents a group ######
	data = add_variable_from_mean(data, "tqwt_energy",tqwt_energy_lst, 1)
	data = add_variable_from_mean(data, "tqwt_entropy_shannon",tqwt_entropy_shannon_lst, 1)
	data = add_variable_from_mean(data, "tqwt_entropy_log",tqwt_entropy_log_lst, 1)
	data = add_variable_from_mean(data, "tqwt_TKEO_mean",tqwt_TKEO_mean_lst, 1)
	data = add_variable_from_mean(data, "tqwt_TKEO_std",tqwt_TKEO_std_lst, 1)
	data = add_variable_from_mean(data, "tqwt_medianValue",tqwt_medianValue_lst, 1)
	data = add_variable_from_mean(data, "tqwt_meanValue",tqwt_meanValue_lst, 1)
	data = add_variable_from_mean(data, "tqwt_stdValue",tqwt_stdValue_lst, 1)
	data = add_variable_from_mean(data, "tqwt_minValue",tqwt_minValue_lst, 1)
	data = add_variable_from_mean(data, "tqwt_maxValue",tqwt_maxValue_lst, 1)
	data = add_variable_from_mean(data, "tqwt_skewnessValue",tqwt_skewnessValue_lst, 1)
	data = add_variable_from_mean(data, "tqwt_kurtosisValue",tqwt_kurtosisValue_lst, 1)


	# add new variable in a dataset that represents a group ######
	print("1/12")
	tqwt_energy = add_variable_from_mean(tqwt_energy, "tqwt_energy",tqwt_energy_lst, 0)
	print("2/12")
	tqwt_entropy_shannon = add_variable_from_mean(tqwt_entropy_shannon, "tqwt_entropy_shannon",tqwt_entropy_shannon_lst, 0)
	print("3/12")
	tqwt_entropy_log = add_variable_from_mean(tqwt_entropy_log, "tqwt_entropy_log",tqwt_entropy_log_lst, 0)
	print("4/12")
	tqwt_TKEO_mean = add_variable_from_mean(tqwt_TKEO_mean, "tqwt_TKEO_mean",tqwt_TKEO_mean_lst, 0)
	print("5/12")
	tqwt_TKEO_std = add_variable_from_mean(tqwt_TKEO_std, "tqwt_TKEO_std",tqwt_TKEO_std_lst, 0)
	print("6/12")
	tqwt_medianValue = add_variable_from_mean(tqwt_medianValue, "tqwt_medianValue",tqwt_medianValue_lst, 0)
	print("7/12")
	tqwt_meanValue = add_variable_from_mean(tqwt_meanValue, "tqwt_meanValue",tqwt_meanValue_lst, 0)
	print("8/12")
	tqwt_stdValue = add_variable_from_mean(tqwt_stdValue, "tqwt_stdValue",tqwt_stdValue_lst, 0)
	print("9/12")
	tqwt_minValue = add_variable_from_mean(tqwt_minValue, "tqwt_minValue",tqwt_minValue_lst, 0)
	print("10/12")
	tqwt_maxValue = add_variable_from_mean(tqwt_maxValue, "tqwt_maxValue",tqwt_maxValue_lst, 0)
	print("11/12")
	tqwt_skewnessValue = add_variable_from_mean(tqwt_skewnessValue, "tqwt_skewnessValue",tqwt_skewnessValue_lst, 0)
	print("12/12")
	tqwt_kurtosisValue = add_variable_from_mean(tqwt_kurtosisValue, "tqwt_kurtosisValue",tqwt_kurtosisValue_lst, 0)



	# heatmaps for different groups
	group_correlation(tqwt_energy)
	group_correlation(tqwt_entropy_shannon)
	group_correlation(tqwt_entropy_log)
	group_correlation(tqwt_TKEO_mean)
	group_correlation(tqwt_TKEO_std)
	group_correlation(tqwt_medianValue)
	group_correlation(tqwt_meanValue)
	group_correlation(tqwt_stdValue)
	group_correlation(tqwt_minValue)
	group_correlation(tqwt_maxValue)
	group_correlation(tqwt_skewnessValue)
	group_correlation(tqwt_kurtosisValue)

	"""
    return tqwt_data


# dic = general_dic(False)


# sum = 0

# bf_data = baseline_features(data, dic, 0.8, [0,0,0,0], [0,0,1,1], False)
# ip_data = intensity_parameters(data, dic, 0.90, [0,0,0], True)
# ff_data = formant_frequencies(data, dic, False)
# bp_data = bandwidth_parameters(data, dic, False)
# vf_data = vocal_fold(data, dic)
# mfcc_data = mfcc(data, dic)
# wf_data = wavelet_features(data, dic, False)
# tqwt_data = tqwt_features(data, dic)


# new_data = data[['id','gender']]
# new_data = pd.concat([new_data, bf_data, ip_data, ff_data, bp_data, vf_data, mfcc_data, wf_data, tqwt_data], axis=1, sort=False)
