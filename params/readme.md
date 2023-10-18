Pretrained parameters can be conveniently used with `x_xy.subpkgs.ml.load(...)`.

For example
```python
from x_xy.subpkgs import ml

# note that `version` is `None`, since their is only a single set of parameters
# in the folder `rr_rr_unknwon`
params = ml.load(pretrained="rr_rr_unknown", version=None)

# multiple versions exist for e.g. `rr_rr_rr_known`
params = ml.load(pretrained="rr_rr_rr_known", version=0)
```
