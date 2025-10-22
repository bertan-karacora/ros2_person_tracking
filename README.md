# ROS 2 Person Tracking

ROS 2 package for Person Tracking using multiple sensors.

Note: This is a public, stripped-down version of a private repository. It may depend on other repositories which might not have a public version. Some paths, configurations, dependencies, have been removed or altered, so the code may not run out of the box.

This person tracking system is actually designed as a full 3D multi-object tracking system. Similar to SORT and its variants, it follows the widely-used tracking-by-detection paradigm - currently the most prominent real-time tracking strategy in state-of-the-art literature. This approach simplifies the tracking process into two main steps: detection and track-to-detection association. This implementation is inspired by multiple state-of-the-art tracking methods. Most importantly, to supplement the typical tracking capabilities of SORT, DeepSORT, etc., which do tracking in 2D images, we do 3D MOT using multiple different sensors with as much flexibility and independence as possible. We decided to separate the detection pipelines of the sensors and only fuse afterwards on detection-level in the track association and Kalman filtering steps, allowing for modular and standalone implementations of the detection stage.

## ReID integration

This system does not perform ReID (Re-Identification). Instead, ReID is handled as an optional part of the detection process.

If a person becomes occluded or temporarily leaves the camera’s field of view, their track is dropped and a new one is created when they reappear. This is **intentional behavior**, designed to clearly separate responsibilities:

- **Tracking**: Assigns a track ID while the person is visible—no assumptions about identity upon reappearance.
- **ReID**: Handles associating new tracks with previously lost ones, enabling identity persistence.

This separation allows the tracking module to operate continuously and independently.

## Sensor Inputs

Tracking relies on detections from both the camera (either Orbbec or IDS) and the 2D LiDARs. Note that it is very eas to add additional sensors as most of the implementation and all the representation of tracks, detections, etc., are already abstracted from the sensors. A sensor_id attribute is used to keep track of the sensor each detection originated from. This ID is also tracked and a slight cost reduction factor is subtracted if a detection originated from a different sensor than the last one used to update a track. This mitigates inaccurate calibration.

### 2D LiDAR-based Person Detection

- Based on DR-SPAAM, a state-of-the-art deep learning model for detecting people in BEV 2D point clouds.
- Key strengths:
  - Full 360° coverage
  - High efficiency
- Custom PyTorch implementation of pre/post-processing allows partial GPU acceleration, significantly reducing inference latency of original work with only limited increase in memory.
- Helps maintain tracks when the camera loses the target due to limited field of view and if the robot navigation makes it turn away.

### Camera-based Person Detection

- More computationally expensive, but more accurate (slightly less so with IDS camera).
- Detects people and computes ReID embeddings for both face and appearance.
- As the robot moves, detections are transformed into a global reference frame for consistency (motion compensation).

## Track Association

The core component of the pipeline is the association of existing tracks with new detections. This is handled using the Hungarian algorithm to solve the linear assignment problem efficiently.

### Cost Computation

The cost for associating a track \( t_i \) and a detection \( d_j \) is based on a convex combination of spatial and semantic similarity:

- **Spatial similarity** from BEV Euclidean distance with cutoff distance $d_{max}$:

$$
    s_{sp} = 1 - \frac{\max(\|p(t_i) - p(d_j)\|_2, d_{max})}{d_{max}}
$$

- **Semantic similarity** as a nonlinear remapping (not going into details here) of the cosine similarity of the features:

$$
    s_{se} = g(f(t_i)^\intercal f(d_j))
$$

- **Final cost**:

$$
    c_{ij} = w_{se} \cdot (1 - s_{se}) + (1 - w_{se}) \cdot (1 - s_{sp})
$$

## Setup

```bash
git clone https://git.ais.uni-bonn.de/athome/ros2_person_tracking.git
cd ros2_person_tracking
git submodule update --init --recursive
```

## Installation

### Build container

```bash
container/build.sh
```

## Usage

### Run in container

```bash
container/run.sh
```

You may provide any command with arguments directly, e.g.:

```bash
container/run.sh -a scripts/start_all.sh name_config:=constant_acceleration
```

Note that the person tracking module has a lot of parameters that are not exposed as ROS launch parameters yet. In `person_tracker.py`, there are parameters:

- `num_matches_to_confirm_track=10` for the number of detection associated to a track to start publishin it.
- `num_steps_to_keep_track_alive=50` for the number of steps to keep a track alive. This controls how fast a person is forgotten after their track is lost.

and in `track.py`:

- `momentum_ema=0.99, use_adaptive_ema=False` for the momentum of the Exponential Moving Average.

Check the `constant_velocity.yaml` config file to adjust parameters like

- `max_distance: 1.3` for the spatial association cost.
- `threshold: 0.85` for the semantic (ReID) association cost.
- `weight_reid: 0.3` for the weight of the semantic cost in the full cost used for track-detection association.
- `bias_fusion_incentive: 0.2` for the incentive to reduce the spatial cost by if the current detection comes from a different sensor than the last one used for updating the track. This helps mitigating wrong calibration or inaccuracies in detections coming from different sensors.

## Links

- [MCTrack](https://arxiv.org/abs/2409.16149)
- [SimpleTrack](https://arxiv.org/pdf/2111.09621)
- [ByteTrackV2](https://arxiv.org/pdf/2303.15334)
- [Hybrid-SORT](https://arxiv.org/pdf/2308.00783)
- [GHOST](https://arxiv.org/pdf/2206.04656)
- [FastMOT](https://github.com/GeekAlexis/FastMOT)
- [BoxMOT](https://github.com/mikel-brostrom/boxmot/tree/master)
- [YOLO Tracking](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/trackers)
- [Out-of-Sequence State Estimation](https://ieeexplore.ieee.org/abstract/document/1292140)
- [Handling Out-of-Sequence Measurements](https://de.mathworks.com/help/fusion/ug/handle-out-of-sequence-measurements-with-filter-retrodiction.html)
- [Stone Soup Tracking Framework](https://stonesoup.readthedocs.io/en/latest/design.html)
- [Crowdbot Project](https://github.com/VisualComputingInstitute/CROWDBOT_perception)
- [Spencer Multi-Modal Detection and Tracking Framework](https://github.com/spencer-project/spencer_people_tracking)
- [2D LiDAR Detection and Tracking](https://github.com/spencer-project/spencer_people_tracking)
- [OSNet weights, MSMT+CUHK3+DUKE](https://drive.google.com/file/d/1nIrszJVYSHf3Ej8-j6DTFdWz8EnO42PB/view?usp=sharing)
- [OSNet weights, MSMT](https://drive.google.com/file/d/1nIrszJVYSHf3Ej8-j6DTFdWz8EnO42PB/view?usp=sharing)

## TODO

- Fix: IDs from associator incorrectly used when retrodiction enabled. Retrodiction currently disabled.
- Fix: Wrong usage of retrocorrected output, need to insert prediction into saves at the time of the measurement, and update only current state
- Fix: Filters for size and orientation currently unfinished and not used
- Add second association stage
- Optimize hyperparameters
- Consider face features
