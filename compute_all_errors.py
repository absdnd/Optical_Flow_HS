'''
Compute all the errors and the final plots included in the report 
'''
import numpy as np
from utils import (
    meanSquareError, 
)
from plot_utils import (
    plot_error_dict,
    plot_final_error_vs_reg_param,
)
import cv2
import numpy as np
import os

BASE_RESULT_PATH = "results/other-pred-flow"
BASE_GT_PATH = "data/other-gt-flow/other-gt-flow"

 
reg_param_list = [1.0, 5, 10, 15]
iteration_list = [10, 20, 30, 40, 50]

# Normalize an image to have values between -1, 1 using min-max normalization
def normalize_img(img):
    # img_norm = (img - np.mean(img)) / np.std(img)
    img -= np.min(img)
    img /= np.max(img)
    # img = img * 2 - 1
    return img


def read_gt_flow(gt_path, filename = "flow10.flo", use_gt = True):
    mask = None
    if use_gt:
        full_gt_path = gt_path + "/" + filename
        gt_flow = cv2.readOpticalFlow(full_gt_path)
        mask =  gt_flow < 1e-9
        gt_flow[~mask] = 0.0
        gt_flow = normalize_img(gt_flow)
    return gt_flow, mask

def average_error_across_images(error_dict):
    ret_dict = {}
    for reg_param, reg_dict in error_dict.items():
        ret_dict[reg_param] = {}
        for iteration, img_dict in reg_dict.items():
            avg_val = np.mean(list(img_dict.values()))
            ret_dict[reg_param][iteration] = avg_val
    return ret_dict

def compute_error_dict(reg_param_list, iteration_list):
    error_dict = {}
    for reg_param in reg_param_list:
        error_dict[reg_param] = {}
        reg_path = BASE_RESULT_PATH + "/" + f"reg_param_{reg_param}"
        
        for iteration in iteration_list:
            error_dict[reg_param][iteration] = {}
            for img_folder in os.listdir(reg_path): 
                full_reg_path = reg_path + "/" + img_folder
                gt_path = BASE_GT_PATH + "/" + img_folder

                if not os.path.exists(gt_path): 
                    continue

                gt_flow, mask  = read_gt_flow(gt_path)
                # visualize_gt_flow = flow_uv_to_colors(gt_flow)

                full_pred_path = full_reg_path + "/" + f"optical_flow_{iteration}.npy"

                pred_flow = np.load(full_pred_path)
                pred_flow = normalize_img(pred_flow)
                
                error = meanSquareError(gt_flow=gt_flow, computed_flow=pred_flow, masks = mask)
                error_dict[reg_param][iteration][img_folder] = error

    return error_dict 

if __name__ == "__main__":
    error_dict = compute_error_dict(reg_param_list=reg_param_list, iteration_list=iteration_list)
    avg_error_dict = average_error_across_images(error_dict)
    output_iteration_plot_filename = BASE_RESULT_PATH + "/error_plot_with_iterations.png"
    output_bar_plot_filename = BASE_RESULT_PATH + "/error_bar_plot.png"
    plot_error_dict(avg_error_dict, output_filename=output_iteration_plot_filename)
    plot_final_error_vs_reg_param(avg_error_dict, output_filename=output_bar_plot_filename)
