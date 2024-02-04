# EYESAY-AI

## Introduction
On-Device Eye Tracking System Using MobileNetV3 and TensorFlow

## Dataset
The EyeSay model leverages the extensive MIT GazeCapture dataset, accessible upon registration at the dataset's website. Our selection criteria ensured the inclusion of frames with valid face and eye detections, leading to a refined dataset from 1,241 participants, totaling 501,735 frames. Detailed explanations of the dataset's structure and contents are provided in the official GazeCapture repository.

[project website](https://gazecapture.csail.mit.edu/download.php)

## The Network
The EyeSay model reproduces the neural network architecture proposed by Google, utilizing TensorFlow. We have adapted the architecture to be compatible with MobileNetV3, optimizing it for on-device performance without compromising accuracy.

### Training
The model was trained using data formatted as TF Records, with a focus on minimizing Mean Squared Error (MSE) and evaluating performance based on Mean Euclidean Distance (MED). Post-training quantization was applied to mitigate potential accuracy drops when converting to TensorFlow Lite format.

## Implementation Details

### Architecture
The EyeSay model employs MobileNetV3, chosen for its efficiency and effectiveness in on-device applications. The network takes cropped images of the face, left eye, and right eye as inputs, producing x, y coordinates of the gaze as output.

### On-Device Deployment
The trained TensorFlow model was converted to TensorFlow Lite for integration into mobile applications. This conversion facilitates the deployment of the eye tracking system directly on smartphones, enabling real-time gaze tracking without the need for external hardware.



### References
```
1. Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

2. Accelerating eye movement research via accurate and affordable smartphone eye tracking
Valliappan, N., Dai, N., Steinberg, E., He, J., Rogers, K., Ramachandran, V., Xu, P., Shojaeizadeh, M., Guo, L., Kohlhoff, K. and Navalpakkam, V.
Nature communications, 2020

3. Somnath, S. (2022). Gaze Tracker [Software]. Available from https://github.com/s0mnaths/Gaze-Tracker.
```
