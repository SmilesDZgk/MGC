## Code for MGC

This repo contains an example implementation of our paper (under reviewing for TNNLS) "Handling Over-smoothing and Over-squashing in Graph Convolution with Maximization Operation" . Please note that the repository is not yet finalized and will be further enhanced with complete code and settings upon possible acceptance.


### Dependencies 
our code is based on Python 3.8. The required packages can be found in `requirements.txt`


### Usage

For training  the model on the Cora dataset:

```bash
python3 main.py --n-layers 128  --n-epochs 200 --lr 0.1   --dataset cora   --model MGC   --weight-decay 0.0003  --dropout1 0.4  --n-hid 32  --dropout2 0.4 --alpha 0.05
```


For large-scale dataset, such as MAG240M dataset, where 120M paper nodes with 1.2B citations are involved for the node calssification task, we need pre-process node features first by runing:

```bash
python 120MAGpre.py --pos 
python 120MAGpre.py 
```
Then, we can train the model based on pre-processed node features by run:

``` bash
python3 mainbatch120M.py   --n-epochs 300 --lr 0.00003  --gpu 0  --model MGC   --weight-decay 0.00000  --dropout1 0.1 --dropout2 0.1  --n-hid 2048 --alpha 0.05
```

Note that, due to the file size of the MAG240M node feature matrix, some scripts may require up to 256GB RAM.