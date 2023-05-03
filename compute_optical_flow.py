import cv2
import numpy as np
import os
from argparse import ArgumentParser
from utils import (
    discrete_optical_flow, 
    write_optical_flow, 
    compute_all_errors, 
    write_all_optical_flow
)
import argparse

''' 
Experimental requirements 
- Using 50 iterations of optical flow to compute the error 
'''

def compute_gt_flow(source_img, target_img, gt_path, use_gt = False):
    if not use_gt: 
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
    else: 
        gt_flow = cv2.readOpticalFlow(os.path.join(gt_path, "flow10.flo"))
    return gt_flow

def log(message):
    print(message)

def compute_all_optical_flow(
    data_path = "",
    gt_path = "", 
    output_path="", 
    source_img_name = "frame10.png",
    target_img_name = "frame11.png",
    gt_image_name = "flow10.flo",
    alpha = 15, 
    delta = 0.1, 
    num_iterations = 300,
    save_optical_flow = True,
    iteration_list = [], 
    use_gt = False
):

    gt_folders = os.listdir(gt_path)
    img_folders = os.listdir(data_path)
    error_list = []
    
    for img_folder in img_folders: 
        log("Computing Optical Flow for folder:.... " + img_folder)     
        
        source_img_path = os.path.join(data_path, img_folder, source_img_name)
        target_img_path = os.path.join(data_path, img_folder, target_img_name)
        
        source_img = cv2.imread(source_img_path, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)

        # results = computeHS(source_img, target_img, alpha=alpha,  delta=delta, max_iter = num_iterations, iteration_list = iteration_list)
        results = discrete_optical_flow(source_img, target_img, alpha=alpha,  delta=delta, max_iter = num_iterations, iteration_list = iteration_list)
        result_key  = max([*results.keys()])

        [u, v] = results[result_key]
        computed_optical_flow = np.stack((u, v), -1)
        img_gt_path = os.path.join(gt_path, img_folder)
        computed_gt_flow = compute_gt_flow(source_img, target_img, img_gt_path, use_gt = use_gt)

        if save_optical_flow:
            write_all_optical_flow(
                results, 
                img_folder, 
                output_path = output_path
            )

            write_optical_flow(
                computed_gt_flow, 
                img_folder, 
                output_path = output_path
            )

        error_dict = compute_all_errors(results, computed_gt_flow)
        log ("Mean Squared Error for folder " + img_folder + " is " + str(error_dict[result_key]))
        error_list.append(error_dict)
    return error_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, default='default', help='Name of the run')
    

    data_path  = "data/other-gray-twoframes/other-data-gray"
    gt_path = "data/other-gt-flow"
    output_path = "results/other-pred-flow"

    iteration_list = [0, 50, 100, 150, 200, 250, 300]
    ''' 
    TODO: Pass in all the variable parameters as arguments
    - This can help with doing comparisons towards the end 
    '''

    error_list = compute_all_optical_flow(
        data_path = data_path,
        gt_path = gt_path,
        output_path = output_path, 
        alpha = 0.15, 
        delta = 0.1, 
        iteration_list = iteration_list, 
        num_iterations=20
    )
    np.save(output_path, 'error_list.npy')

''' 
24th April, 2023 - Goal finalize the experimental setup

- Fix the number of iterations to 500 
- Log values across the iterations [100, 200, 300, 400, 500]


'''