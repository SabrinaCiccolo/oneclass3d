# oneclass3d

One-class 3D anomaly detection on **MVTec 3D-AD**: fit on normal scans only, flag unseen
defects. A frozen per-patch descriptor feeds a memory bank of normal patches, and **six
anomaly scorers** are compared on equal footing, each on **raw** and on **whitened**
features. No training, no GPU: everything from the raw point clouds to the figures runs
on a CPU in this repository.
