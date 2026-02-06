
# ADTH-Camera-Stitched-Polar-Optical-System

Passive camera-stitched optical perception system (research prototype).

This repository contains code, scripts and documentation for a camera-stitched optical detection and localization research project developed for the Australian Defense Tech Hackathon (ADTH). The project explores multi-camera fusion, visual detection and tracking, and non-radar 3D localization in simulation and controlled test environments.

**IMPORTANT — Safety & Usage**
- This project is intended for research, simulation, and evaluation only. It must NOT be connected to live weapon systems or used to control lethally-armed platforms.
- Any adaptation that interfaces with weapons, kinetic effectors, or otherwise enables lethal action is strictly prohibited without appropriate legal authority, oversight, and safety engineering.

## Challenge Description

Field reports indicated the emergence of drones built with RF-absorbing polymers that reduce radar visibility. This project investigates whether visible-light camera systems, combined with AI perception, can detect, classify and localize such platforms in cluttered scenes using stitched multi-camera video feeds.

Core, non-actionable goals:
- Seamlessly fuse multiple camera inputs into a single stitched view (a "dome" of awareness).
- Detect and classify aerial objects (e.g., distinguish drones from birds) using machine learning models.
- Maintain robust tracking through brief occlusions (temporary visual occluders such as foliage or rain).
- Estimate 3D direction (azimuth / elevation) and range from 2D detections using stereo or stadiametric cues in a simulated or controlled testbed.

## Project Scope & Limitations
- This repository focuses on perception and localization research (data capture, stitching, detection, tracking, and offline evaluation).
- It intentionally omits any code, interfaces, or guidance that would enable automated engagement or weapon control.
- Any experiments that could affect safety must be performed in simulation or under supervised, approved test conditions.

## Repository Structure

- `docker-compose.yaml` — convenience stack for dev/test services (containers are isolated; inspect before use).
- `dockerfile`, `entrypoint.sh` — container setup used by the stack.
- `requirements.txt` — Python packages used by the project.
- `start.sh`, `start.bat` — simple run wrappers for Linux/Windows (local dev only).
- `code/` — research code and experiments (e.g. `testCamera.py`).

## Quickstart (simulation / local testing)

1. Create and attach to docker containers

Linux setup

```
./start.sh
```

Windows setup
``` 
./start.bat
```

Once inside the containers
``` 
cd code

python testCamera.py
```

Note: The example scripts demonstrate sensor capture, stitching and offline evaluation. They do not perform or instruct on weapon control.

## Development Notes
- Keep experiments and datasets clearly separated from any hardware that could cause harm.
- Use synthetic/simulated inputs or recorded video for model training and evaluation whenever possible.

## Contributing
- Open issues or pull requests for bug fixes, dataset additions, model experiments, or evaluation scripts.
- All contributions must follow the safety guidelines in this README.

## Authors
- Thom Spencer
- Anthony Bebek
- Minh Nguyen

## License

This project is provided under an open-source license (see `LICENSE` if present). Any use that would enable weaponization or lethal automation is not permitted without explicit legal authorization and appropriate safeguards.

## Contact
For questions about safety, research scope, or reproducibility, contact the authors.

