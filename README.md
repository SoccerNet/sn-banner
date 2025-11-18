# Banner replacement

<hr>

## Weights
Download the keypoints and line detection model weights for No-Bells-Just-Whistles camera calibration model, and the mask2former semantic segmentation model weights. Place them in the specified directories.

| Model | Link | Destination |
| :-- | :-: | :-- |
|Keypoints| [**SV_kp**](https://github.com/mguti97/No-Bells-Just-Whistles/releases/download/v1.0.0/SV_kp) | camera_calibration/No-Bells-Just-Whistles/SV_kp |
|Lines| [**SV_lines**](https://github.com/mguti97/No-Bells-Just-Whistles/releases/download/v1.0.0/SV_lines) | camera_calibration/No-Bells-Just-Whistles/SV_lines |
|challenge_mask2former| [**best_mIoU_iter_10935.pth**](https://huggingface.co/datasets/SoccerNet/BannerReplacement/resolve/main/best_mIoU_iter_10935.pth) | semantic_segmentation/models/challenge_mask2former/best_mIoU_iter_10935.pth |


<hr>

## Installation

Though [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [No-Bells-Just-Whistles](https://github.com/mguti97/No-Bells-Just-Whistles) (or NBJW) could share the same conda environment, the installation of mmcv, a requirement for mmsegmentation, is not straightforward for pytorch versions more recent than 2.1.2, and NBJW has shown improved accuracy with more recent pytorch versions. Therefore, this project works with two independent conda environments: one for MMsegmentation (``mmseg``) and another for the rest of the project (``banner-replacement``).

The environnement `banner-replacement` is the default one, except for the files `train.py`, `test.py` and `inference.py`, in the directory `semantic_segmentation`, which must be run in the `mmseg` environment.

### mmseg

The conda environment for MMsegmentation can be created by running the following command:
```shell
conda create -n mmseg python=3.11 -y
conda activate mmseg
```
Install pytorch. The package mmcv, a requirement for mmsegmentation, depends heavily on the pytorch version installed. The recommended version is 2.1.2, with CUDA 11.8 or 12.1. If you want to install another pytorch version, see below. The installation commands for each cuda version and using either pip or conda can be found [here](https://pytorch.org/get-started/previous-versions/#v212). For example, to install the recommended pytorch version with pip and CUDA 12.1:
```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

If the installation of mmcv fails after the installation of pytorch with pip, try using conda instead, and vice versa.

Then, install [mim](https://github.com/open-mmlab/mim) v0.3.9, a package management tool for the OpenMMLab projects, which makes it easy to install mmcv:
```shell
pip install openmim==0.3.9
```

Install mmengine v0.10.3, mmdet v3.3.0 and mmcv v2.1.0:
```shell
mim install mmengine==0.10.3 mmdet==3.3.0 mmcv==2.1.0
```

If you want to use another version of pytorch and cuda, alternatives to install mmcv v2.1.0 are:
- Let mim check your system and install, if possible, a pre-built package (faster, no compilation needed, list of systems with pre-built packages available [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip)) else it will try to built the package.
- Install with pip a pre-built package. The list of type of system, CUDA version, pytorch version and mmcv version (which in our case must be 2.1.0) is available [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip).
- Install from [source](https://mmcv.readthedocs.io/en/latest/get_started/build.html)

More information on the [MMCV official installation guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip).

Installing mmengine will probably install numpy>2, which is not compatible withg mmsegmentation 1.2.2. Check using `pip list` to see the installed version. Therefore, downgrade numpy and packages requiring numpy>2:
```shell
pip install numpy==1.26.4 opencv-python==4.10.0.84
```

Finally, install the rest of the requirements:
```shell
pip install regex==2024.7.24 ftfy==6.2.0 mmsegmentation==1.2.2
```

### banner-replacement

The conda environment for the rest of the project can be created by running the following commands:
```shell
conda create -n banner-replacement python=3.11 -y
conda activate banner-replacement
```

Then, install pytorch from the [official website](https://pytorch.org/get-started/locally/) or a [previous version](https://pytorch.org/get-started/previous-versions/). No-Bells-Just-Whistles, the camera calibration model, has shown improved accuracy with more recent versions. The recommended and most recent pytorch version tested is 2.4.0.

Finally, install the rest of the requirements:
```shell
pip install scikit-learn==1.5.1 opencv-python==4.10.0.84 tqdm==4.66.4 matplotlib==3.8.4 PyYAML==6.0.1 pandas==2.2.2
```


<hr>

## Inference

Don't forget to activate `banner-replacement` conda environment before running the following commands.

On video:
``` shell
python inference.py path/to/video.mp4 path/to/logo.png
```

On image sequence from SN-GameState, or any directory with N images named "%06d.jpg" and beginning with "000001.jpg":
``` shell
python inference.py path/to/SoccerNetGS/test/SNGS-142/img1/ path/to/logo.png --sequence
```

Optional arguments:
- ```--n_workers```: Number of workers to speed up some steps of the pipeline. Default is 1. Attention it may use a lot of memory.
- ```--tta```: Whether to use test-time augmentation. Default is False. A GPU with at least 24GB of memory is required.
- ```--speed```: The speed of the banner translation animation. Default is 1.75.
- ```--keep_work_dir```: Whether to keep the temporary working directory.

### Exemple

First, activate the `banner-replacement` conda environment:
``` shell
conda activate banner-replacement
```
In the Github repository, create a directory `SoccerNetGS` and download the challenge set of SoccerNet-GameState by following the instructions [here](https://github.com/SoccerNet/sn-gamestate?tab=readme-ov-file#manual-downloading-of-soccernet-gamestate). Then, run the following command to execute the pipeline on the first video of the challenge set, with the ULiège logo, 6 workers and no test-time augmentation:
``` shell
python inference.py SoccerNetGS/challenge/SNGS-001/img1/ University_of_Liège_logo_resized.png --sequence --n_workers 6
```
A working directory named `work_dir_` followed by a random string will be created in the current directory and progression bars will be displayed for each step of the pipeline, except for the semantic segmentation step. To see the progression of the semantic segmentation step, check the log file in `work_dir_xxxxx/logs/date_stamp/date_stamp.log`.
The resulting video, named `output_SNGS-001.mp4`, will be created in the current directory and should look like [this](https://huggingface.co/datasets/SoccerNet/BannerReplacement/resolve/main/output_SNGS-001-non-tta.mp4).

To use test-time augmentation, add the `--tta` argument:
``` shell
python inference.py SoccerNetGS/challenge/SNGS-001/img1/ University_of_Liège_logo_resized.png --sequence --n_workers 6 --tta
```
The resulting video should look like [this](https://huggingface.co/datasets/SoccerNet/BannerReplacement/resolve/main/output_SNGS-001-tta.mp4).


<hr>