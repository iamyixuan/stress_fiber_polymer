## Data Description

The zip file `dataset.zip` in `data` folder contains 5321 2D microstructure slices and the corresponding z-normal stress distribution. 

Each slice is store in a `numpy` array with the filename being the corresponding coordinates values.

## Load data:

Use `numpy.load`

Example:

```python
import numpy as np
expl_point = np.load(unzipped_folder_path + '2_1_30', 'r', True)
print(expl_point.shape)
>>> (2, 32, 32)
```
