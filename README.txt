# Color adoption

We want to change the colors in one image to exactly match the color values in a donor image. Every color must be adopted (and can be adopted multiple times).

## Install

```
pip install -r requirements.txt
```

## Color transfer followed by closest allowable color

```
parallel python colortransferlibscript.py --alg {} lukeman.jpg 18.png lukeman-from-18-{} ::: GLO PDF CCS MKL GPC NST DPT EB3 CAM RHG
parallel python color-transfer-closest.py lukeman-from-18-{}.png 18.png lukeman-from-18-{}-closest.png ::: GLO PDF CCS MKL GPC NST DPT EB3 CAM RHG
ql lukeman-from-18-*-closest.png
```

GPC may be a winner

## Bipartite experiment

### Many k:

```
parallel python color-transfer-bipartite-matching.py -k '{}' source.png donor.png source-from-donor-k'{}'.png ::: 1 5 10 100 1000
```
