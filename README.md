# HDU_CS method for GAZE 2021 competition on the EVE dataset

* Authors: Jun Bao, Buyu Liu, Jun Yu

Our full report for this method can be found in paper 'The Story in Your Eyes: An Individual-difference-aware Model for Cross-person Gaze Estimation', https://arxiv.org/abs/2106.14183.

This code is developed on the repository https://github.com/swook/EVE, which is written by Seonwook Park, Emre Aksan, Xucong Zhang, and Otmar Hilliges as the baseline method for EVE dataset. For more information about the baseline method and EVE dataset please see their ECCV 2020 paper 'Towards End-to-end Video-based Eye Tracking' and accompanying project page: https://ait.ethz.ch/projects/2020/EVE/.


## Environment Setup
The setup is similar to that of the baseline method.

We have tested this code-base in the following environments:
* Ubuntu 18.04
* Python 3.6 
* PyTorch 1.5.1

Clone this repository somewhere with:

    git clone git@github.com:bjj9/EVE_SCPT
    cd EVE_SCPT/

Then from the base directory of this repository, install all dependencies with:

    pip install -r requirements.txt

You will also need to setup **ffmpeg** for video decoding. On Linux, we recommend running 'sudo apt install ffmpeg' in terminal.


## Usage

### Directories specification

1. The EVE dataset should be put in directory 'EVE_SCPT/inputs/datasets/eve_dataset'. An example sample directory is 'EVE_SCPT/inputs/datasets/eve_dataset/test01/step008_image_MIT-i2263021117'

2. The JSON file we are using is 'EVE_SCPT/src/configs/inference_eye_net_10Hz_without_pupil_valid_center_calibrated_st.json'.

3. The model parameters for eye_net and PT module is stored in 'EVE_SCPT/src/models/trained_model_params'.

4. The person-specific memory for offline evaluation will be stored in 'EVE_SCPT/memories' after running online evaluation.

5. The evaluation results will be stored in 'EVE_SCPT/outputs/eval_resultseval_codalab_online_inference_eye_net_10Hz_without_pupil_sc_pt_nsejjeif' with suffix 'online' or 'offline'


### Running evaluation
For online and offline evaluation results run:

    cd EVE_SCPT/
    python eval_codalab_offline.py

The result 'for_codalab_xxxxxx_xxxxxx_online.pkl.gz' should score 2.14 degrees on Gaze direction error.

The result 'for_codalab_xxxxxx_xxxxxx_offline.pkl.gz' should score 1.95 degrees on Gaze direction error.


## Citation
If you find this post helpful, please cite:

    @misc{bao2021story,
          title={The Story in Your Eyes: An Individual-difference-aware Model for Cross-person Gaze Estimation}, 
          author={Jun Bao and Buyu Liu and Jun Yu},
          year={2021},
          eprint={2106.14183},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }

For GAZE 2021 competition on EVE dataset, please cite:

    @inproceedings{Park2020ECCV,
      author    = {Seonwook Park and Emre Aksan and Xucong Zhang and Otmar Hilliges},
      title     = {Towards End-to-end Video-based Eye-Tracking},
      year      = {2020},
      booktitle = {European Conference on Computer Vision (ECCV)}
    }

