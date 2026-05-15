# Pointcloud-Localizer

A self-contained ICP-based 3-D point cloud registration and localisation toolkit.
**The entire ICP loop is hand-rolled** — no `open3d.registration`, PCL, or any
other ICP library function is called.

---

## What it does

| Task | Module |
|---|---|
| Load `.ply` / `.pcd` files | `loader.py` |
| Generate synthetic test pairs with ground-truth | `synthetic.py` |
| Voxel downsampling | `preprocess.py` |
| **ICP registration** | `icp.py` |
| Evaluate accuracy, plot results | `evaluate.py` |
| Command-line interface | `cli.py` |

---

## Setup

### Step 1 — Clone this project
```bash
git clone https://github.com/RishikaIITJ/pointcloud-localizer-
cd pointcloud-localizer
```

### Step 2 — Create a virtual environment (recommended)
```bash
python3 -m venv myenv
source myenv/bin/activate 
```

### Step 3 — Install dependencies
```bash
pip install numpy scipy matplotlib open3d pytest
```

### Step 4 - Download the standford bunny dataset
``` bash
https://github.com/alecjacobson/common-3d-test-models/blob/master/data/stanford-bunny.zip
```
---

## Running the project

### Quick demo
#### Open cli.py to change the mode - 'register' or 'sweep'
```bash
python -m pointcloud_localizer.cli
```

### Run tests
```bash
pytest tests/test_icp.py -v    
```
---
