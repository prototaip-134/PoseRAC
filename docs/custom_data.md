# Pose Estimation and Landmark Extraction Project

This repository provides scripts and resources for extracting pose landmarks from video data and annotating periodic movements. The core functionalities include frame extraction, landmark annotation, and salient pose identification. 

## Repository Structure

- `/annotation`: Contains annotation CSV files (`pose_train.csv`, `test.csv`, `valid.csv`).
- `/video`: Contains video files organized into subdirectories for testing, training, and validation:
  - `/video/test`
  - `/video/train`
  - `/video/valid`

## Annotation Format

Each annotation CSV contains the following columns:

- **Filename**: Name of the video file.
- **L1, L2, ...**: Salient pose landmarks for the movements. Odd-numbered landmarks (`L1`, `L3`, ...) indicate the start, and even-numbered landmarks (`L2`, `L4`, ...) indicate the end of half a periodic movement. For `n` repetitions of an action, there will be `2n` salient poses.
![image](https://github.com/prototaip-134/PoseRAC/assets/67076071/23f5b27f-94f5-462b-98fa-ba106f5c7b12)
- **Count** *(test only)*: Number of repetitions in the video (for test annotations only).
![image](https://github.com/prototaip-134/PoseRAC/assets/67076071/11377f39-8645-4b33-95e9-eacba7edaadb)

## Scripts

### `pretrain.py`

- **Purpose**: Extracts salient pose frames and landmarks for training data.
- **Output**: 
  - Salient frames stored in `/extracted/train/[action]/[salientX]/[sample_folder]/0.jpg`.
  - Landmark annotations saved in `/annotation_pose/train.csv`.

### `pretest.py`

- **Purpose**: Extracts salient pose frames and landmarks for testing data.
- **Output**: 
  - Salient frames stored in `/test_poses` as `.npy` files.

### Other Scripts

Other scripts are provided for various tasks and can be used as described in the README. Ensure that `pretest.py` is executed before performing inference.

## Setup and Usage

1. **Extract Salient Poses**:
   - Run `pretrain.py` to process the training videos and extract frames and landmarks.
   - Run `pretest.py` to process the test videos and extract frames and landmarks.

2. **Annotations**:
   - Training annotations will be stored in `/annotation_pose/train.csv`.
   - Test annotations will be saved in `.npy` format under `/test_poses`.

## Important Notes

- Ensure that the `pretest.py` script is executed before inferencing.
- Follow the steps in the README for setting up the environment and dependencies.

---

For detailed usage instructions and examples, refer to the [README](../README.md) file.
