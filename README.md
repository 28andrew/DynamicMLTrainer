## Group work

### Andrew

* Worked on integrating the algorithms with PyTorch
* Wrote GPU benchmarking script
* Wrote the PyTorch model splitting/pipelining code + that side of the demo code
* Worked on setting up distributed environment, attempting NCCL over virtual Docker network which failed

### Jeffrey

* Wrote most sections of the write-up
* Helped Andrew with the PyTorch pipelining implementation

### Jiakang

* Worked on implementing/testing the brute-force partitioning algorithm
* Worked on implementing/testing the heuristic partitioning algorithm
* Worked on implementing/testing the hierarchical partitioning algorithm
* Helped write the algorithm section of the write-up

## Demo Link

[Demo Video](https://drive.google.com/file/d/12iBG6FmjIcAbXFyMgF-ip_UwRgYhLpni/view?usp=sharing)

## Instructions

The environment can be setup in a new Conda environment. CUDA 12.4 compatible GPUs and an x86_64 OS architecture can use these commands to set up the environment:
```
pip3 install torch torchvision torchaudio &&
apt-get -y install g++ &&
pip install torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html &&
pip install torch_geometric torchinfo matplotlib &&
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.5.1+cu124.html 
```

### Running the Partioning Algorithms Separately

We have not only implemented the partitioning algorithms, but we also written a few toy examples that you run separately to see the performance of the various herustics. 

You can run
```
python3 algo_bash.py
```
and can see what the best partitioning that is obtained by the brute-force algorithm performs after running for 2 minutes on a graph with 201 nodes split over 3 GPUs.

Similarly you can run
```
python3 algo_heuristic.py
```
and can see what the best partitioning that is obtained by the heuristic algorithm with `max_iterations = 10` and trying 20 different initial random partitions. 

Finally you can run
```
python3 algo_hierarchical.py
```
and observe how this algorithm performs on a toy "datacenter" example with 201 nodes but split over 96 GPUs.

You can also run the test suite for the algorithms and their helper functions via
```
pytest test_algos.py
```
You should see 8 tests passing.

## Running the Training
For our method
```
python main.py
```
For the baseline PyTorch FSDP
```
python naive.py
```