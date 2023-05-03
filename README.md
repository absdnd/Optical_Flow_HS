# Optical_Flow_HS
Computing the optical flow using the horn and schunk method composed of the brightness constancy and smoothness constraint. This implementation is was done for the course ECE 6560's final project 

## Dependencies 
- numpy == 1.24.3 
- opencv-python == 4.7.0 
- scipy == 1.10.1 

## Dataset 
We utilize the middlebury optical flow dataset for this implementation. Please download the dataset from here - https://vision.middlebury.edu/flow/data/. We utilize the images with available gt-flow under **Ground Truth Flow**. Please create a `data/` folder in the root folder and download `other-gray-two-frames` and  `other-gt-flow` to this folder. 

## Running the Code 

### Step-1: Compute optical flow 
This step computes the optical flow between two images and saves all the results in a folder called `other-preds-flow/` in the same folder. This saves the optical flow in the npy format in that folder. This includes the predicted optical flow as an image using function `def flow_uv_to_color()` which converts the `[u, v]` vector to a visualizable image (using the implementation from Middlebury dataset). 

To run this execute 

 `python compute_optical_flow.py` 

### Step-2: Computing all Errors 
This step computes the errors of the images saved in the previous step and draws to plots - i) The best error plot across iterations and ii) the evolution of computed error across iterations. The error is implemented by computing the difference with the ground truth optical flow. 

To execute this, please use - 

`python compute_all_errors.py`

## References
- Middlebury Optical flow dataset (https://vision.middlebury.edu/flow/data/)
- Horn and Schunk Original paper (https://www.cmor-faculty.rice.edu/~zhang/caam699/opt-flow/horn81.pdf)
- Kris Kanti's optical flow lectures (https://www.cs.cmu.edu/~16385/s17/Slides/14.3_OF__HornSchunck.pdf) 
- Reference Code on Optical Flow (https://github.com/lmiz100/Optical-flow-Horn-Schunck-method)
