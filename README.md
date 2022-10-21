# Tougong

> 《偷功》 is a song in old HongKong film composed by 胡伟立

Mimnal Dem version

<img src="https://github.com/mrzhuzhe/taichi_dem/blob/main/gifs/v01.gif" height="270px">

PBF version

<img src="https://github.com/mrzhuzhe/taichi_dem/blob/main/gifs/pbf-v01.gif" height="270px">


## Run 
```
python3 dem-3d.py  # Yellow dem cyclone

python3 pbf.py # Blue position based fluid cyclone

python3 dem-3d-snode.py  # YelloW dem cyclone , for comparing, implement by taichi dynamic snode features 
```

## Installation
Make sure your `pip` is up-to-date:

```bash
$ pip3 install pip --upgrade
```

Assume you have a Python 3 environment, to install Taichi:

```bash
$ pip3 install -U taichi
```

To run the demo:

```bash
$ python dem.py
```


## References 

1. Position based fluid is implement based on 

    Ten minutes physics https://matthias-research.github.io/pages/tenMinutePhysics/
    
    https://github.com/matthias-research/pages/blob/master/challenges/fluid2d.html

2. Taichi structured data organization Document 

    Snodes https://docs.taichi-lang.org/docs/internal#data-structure-organization
    
    Dynamic snode https://docs.taichi-lang.org/docs/sparse#dynamic-snode