import cv2
from scipy.ndimage import convolve as filter2
from matplotlib import pyplot as plt
import numpy as np
import os

''' 
Read optical flow from the flow-file-path 
'''
def readOpticalFlow(flow_file_path):
    flow_image = cv2.readOpticalFlow(flow_file_path)
    return flow_image
    
import numpy as np


def get_derivatives(img1, img2):
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    fx = filter2(img1,x_kernel) + filter2(img2,x_kernel)
    fy = filter2(img1, y_kernel) + filter2(img2, y_kernel)
    ft = filter2(img1, -t_kernel) + filter2(img2, t_kernel)

    return [fx,fy, ft]

def readOpticalFlow(flow_file_path):
    flow_image = cv2.readOpticalFlow(flow_file_path)
    return flow_image




def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)

    return flow_image



def show_image(name, image):
    if image is None:
        return

    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_optical_flow(op_flow):
    u = op_flow[..., 0]
    v = op_flow[..., 1]


    rgb = flow_uv_to_colors(u, v, convert_to_bgr=True)
    return rgb

def compute_all_errors(results, computed_gt_flow):
    errors_dict = {}
    for key in results.keys():
        [u,v] = results[key]
        pred_flow = np.stack((u, v), axis = -1)
        errors_dict[key] = meanSquareError(pred_flow, computed_gt_flow)
    return errors_dict

''' 
Optical flow computation between two images 
- Alpha, delta are parameters for the algorithm
- max_iter is the maximum number of iterations
- iteration_list is a list of iterations to save the intermediate results
'''

# Blur the image to remove high frequency noise
def blur_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def compute_image_derivatives(img1, img2):
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    Ex = filter2(img1,x_kernel) + filter2(img2,x_kernel)
    Ey = filter2(img1, y_kernel) + filter2(img2, y_kernel)
    Et = filter2(img1, -t_kernel) + filter2(img2, t_kernel)

    return [Ex, Ey, Et]


''' 
Optical flow computation between two images 
- Alpha is the tradeoff parameter between brightness constancy and smoothness
- max_iter is the maximum number of iterations
- iteration_list is a list of iterations to save the intermediate results
'''
def discrete_optical_flow(img1, img2, alpha, max_iter = 300, iteration_list = []):
    results = {}
    img1 = blur_image(img1.astype(float))
    img2 = blur_image(img2.astype(float))

    assert img1.shape[0] == img2.shape[0]
    assert img1.shape[1] == img2.shape[1]
    
    u, v = np.zeros((img1.shape[0], img1.shape[1])), np.zeros((img2.shape[0], img2.shape[1]))

    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                            [1 / 6, 0, 1 / 6],
                            [1 / 12, 1 / 6, 1 / 12]], np.float32)

    Ex, Ey, Et = compute_image_derivatives(img1, img2) 
    iter_counter = 0
    for iter in range(max_iter): 
        u_avg = filter2(u, avg_kernel)
        v_avg = filter2(v, avg_kernel)
        
        bright_constancy = Ex * u_avg + Ey * v_avg + Et

        update_term = bright_constancy / (4* alpha ** 2 + Ex ** 2 + Ey ** 2)

        u = u_avg - Ex * update_term
        v = v_avg - Ey * update_term

        results[iter] = [u, v]
    return results


''' 
Write all optical flow from the results dictionary
- Writes all iterations of optical flow to a folder
'''
def write_all_optical_flow(
    results, 
    img_folder, 
    output_path,
    save_npy = True
): 
    for k, v in results.items(): 
        op_flow = np.stack((v[0], v[1]), axis = -1)
        write_optical_flow(op_flow, img_folder, output_path, img_name = f"optical_flow_{k}.png", save_npy = save_npy)


def write_optical_flow(
    optical_flow,
    img_folder, 
    output_path, 
    img_name = "optical_flow.png", 
    save_npy = True
): 
    
    u, v = optical_flow[..., 0], optical_flow[..., 1]
    vis_optical_flow = flow_uv_to_colors(u, v, convert_to_bgr=True)

    output_vis_path = os.path.join(output_path, img_folder)
    os.makedirs(output_vis_path, exist_ok=True)
    cv2.imwrite(os.path.join(output_vis_path, img_name), vis_optical_flow)

    if save_npy: 
        npy_name = img_name.replace(".png", ".npy")
        np.save(os.path.join(output_vis_path, npy_name), optical_flow)

''' 
Compute the masked optical flow between two images: 
- computed_flow: the computed optical flow
- gt_flow: the ground truth optical flow
- masks: Masking out uncertain regions from the ground truth optical flow
'''
def meanSquareError(computed_flow, gt_flow, masks = None):
    total_samples = computed_flow.shape[0] * computed_flow.shape[1]
    diff = computed_flow - gt_flow
    
    if masks is not None:
        diff = diff*masks
        total_samples = masks.sum()
    else: 
        total_samples = computed_flow.shape[0] * computed_flow.shape[1]

    return np.linalg.norm(computed_flow - gt_flow)/total_samples
