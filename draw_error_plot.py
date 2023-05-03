import matplotlib.pyplot as plt
import numpy as np
import os
from utils import readFlowFile
from utils import flow_uv_to_colors, meanSquareError
import cv2


def compute_gt_flow(source_img, target_img):
    gt_flow = cv2.calcOpticalFlowFarneback(
        source_img, 
        target_img, 
        None, 
        pyr_scale=0.5,
        levels = 3,
        winsize = 15,
        iterations = 3,
        poly_n = 5, 
        poly_sigma = 1.2, 
        flags = 0
    )
    return gt_flow

def compute_error_across_images(
    img_folder, 
    pred_folder, 
    frame1_name = "frame10.png",
    frame2_name = "frame11.png",
    iteration = 300
):
    img_list = os.listdir(img_folder)
    error_list = []
    for img_name in img_list:
        frame1 = cv2.imread(os.path.join(img_folder, img_name, frame1_name), cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(os.path.join(img_folder, img_name, frame2_name), cv2.IMREAD_GRAYSCALE)

        gt_flow = compute_gt_flow(frame1, frame2)
        pred_flow = np.load(os.path.join(pred_folder, img_name, f"optical_flow_{iteration}.npy"))
        error = meanSquareError(gt_flow=gt_flow, computed_flow=pred_flow)
        error_list.append(error)

    return error_list

def plot_error_across_iterations(
    error_dict
): 
    fig, ax = plt.subplots()
    for reg_parm, reg_dict in error_dict.items():
        plot_list = []
        x_val = [*reg_dict.keys()]
        y_val = [sum(v) for _, v in reg_dict.items()]
        print(x_val, y_val)
        ax.plot(x_val, y_val, label=f"reg_param = {reg_parm}")

    ax.legend()
    plt.savefig("results/error_plot.png")

    
if __name__ == "__main__": 
    file_path = 'data/other-gt-flow/other-gt-flow/Dimetrodon/flow10.flo'
    compute_path = 'results/other-pred-flow/reg_param_0.15/Dimetrodon/computed_optical_flow.npy'
    flow_img = cv2.readOpticalFlow(file_path)
    img_folder = "data/other-gray-twoframes/other-data-gray/"
    pred_folder = "results/other-pred-flow/"

    # iteration_list = [50, 100, 150, 200, 250, 300]
    # alpha_list = [1, 0.05, 0.1, 0.15]
    alpha_list = [5, 10, 15]
    iteration_list = [10, 20, 30, 40, 50]


    error_dict = {}
    
    for alpha in alpha_list:
        alpha_pred_folder = f"results/other-pred-flow/reg_param_{alpha}/"
        error_dict[alpha] = {}
        for iteration in iteration_list:
            error_dict[alpha][iteration] = compute_error_across_images(img_folder, alpha_pred_folder, iteration=iteration)

    plot_error_across_iterations(error_dict)
    