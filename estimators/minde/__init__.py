"""Internal building blocks for the MINDE diffusion-based MI estimator.

Used by ``estimators.MINDE``; the pieces are imported directly from their
modules (absolute ``estimators.minde.<module>`` paths), so nothing is
re-exported here:

- ``diffusion``     : the VP-SDE forward/reverse diffusion process.
- ``UnetMLP``       : U-Net-style MLP score network.
- ``diff_utils``    : concat/deconcat, marginalization, conditioning, EMA.
- ``info_measures`` : joint/conditional MI read-offs from the score model.
- ``importance``    : importance-weighted normalizing constant.
"""
