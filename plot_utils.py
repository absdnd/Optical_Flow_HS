import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14

def plot_error_dict(error_dict, output_filename): 
    fig, ax = plt.subplots(figsize = (10, 7))
    plt.subplots_adjust(left=0.17, bottom=0.15)
    for reg_parm, reg_dict in error_dict.items():
        plot_list = []
        x_val = [*reg_dict.keys()]
        y_val = []
        for _, v in reg_dict.items(): 
            if type(v) == 'list': 
                y_val += [sum(v)]
            else:
                y_val += [v]
        
        print(x_val, y_val)
        ax.plot(x_val, y_val,
             label=f"reg_param = {reg_parm}",
             marker = '^')
        
    ax.set_ylabel('MSE', fontsize = 20)
    ax.set_xlabel('Regularization Parameter', fontsize = 20)
    ax.tick_params(axis='x', which='major', labelsize=18)
    ax.tick_params(axis='y', which='major', labelsize=18)
    ax.legend(loc = 'lower right')
    plt.savefig(output_filename)

def plot_final_error_vs_reg_param(error_dict, output_filename): 
    fig, ax = plt.subplots(figsize = (10, 7))
    plt.subplots_adjust(left=0.17, bottom=0.15)
    x_label_list = [*error_dict.keys()]
    y_label_list = []
    colors = ['green', 'red', 'red', 'red']
    y_ticks = [2.5e-4, 5e-4, 7.5e-4, 1e-3, 1.25e-3, 1.5e-3, 1.75e-3, 2e-3]
    for reg_param, reg_dict in error_dict.items():
        plot_list = []
        x_val = [*reg_dict.keys()]
        y_val = []
        for _, v in reg_dict.items(): 
            if type(v) == 'list': 
                y_val += [sum(v)]
            else:
                y_val += [v]

        y_label_list.append(max(y_val))
    ax.bar(x_label_list, y_label_list, width = 1.0, color = colors)
    ax.set_ylabel('MSE', fontsize = 20)
    ax.set_xlabel('Regularization Parameter', fontsize = 20)
    ax.set_xticks(x_label_list, labels = x_label_list)
    ax.tick_params(axis='x', which='major', labelsize=18)
    ax.tick_params(axis='y', which='major', labelsize=18)
    # ax.set_yticks(y_ticks, labels = y_ticks, fontsize = 18)
    ax.legend(loc = 'upper right')
    plt.savefig(output_filename)

