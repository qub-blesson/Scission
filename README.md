# Scission
## A Tool for Maximising Performance of Deep Neural Networks in Edge Computing 

### Viewing Help
```python3 ./scission_benchmark.py --help```  
```python3 ./scission_predict.py --help```

### Scission benchmarking

#### Dependencies
* Tensorflow 2.1
* NumPy
* Pillow

#### Benchmarking

```python3 ./scission_benchmark.py cloud 6700k cat.jpeg```

An output file "cloud-6700k(GPU).dat" will be saved which contains the benchmark data for each benchmarked model.

### Scission prediction

#### Dependencies
* NumPy
* Matplotlib

#### Network Statistics 

Predictions use externally provided network statistics in a csv file formatted as below:

```
source,     destination,     ping(ms),      bandwidth(mbps)
device,     edge,            66.7,          1.6
device,     cloud,           100,           1.6
edge,       cloud,           20,            50
```

#### Querying - Criteria Format

* A number to indicate that the layer with the corresponding number must be executed on that platform.
* A number preceded by a "!" to indicate that the layer with corresponding number must not be executed on that platform.
* A string to indicate that only the system with the specified name is used for that platform.
* A string predceded by a "!" to indicate that the specified system must not be used. 

Criteria | Explanation 
-------- | ---------- 
-d “3,!10,device1” | Layer 3 must be executed on device, layer 10 must not. The device with name "device1" must be used.
-e “-1” | The edge must not be used.
-c “30,!cloud1)” | Layer 30 must be executed on the cloud, the device with name “cloud1” must not be used.

#### Prediction

```python3 ./scission_predict.py benchmark_data network_stats.csv vgg19 -d "!-1" -e 10 -c 16```

The specified benchmark_data folder must contain data files produced by "scission_benchmark" and a network statistics file which contains network connection information for each of the systems benchmarked. An example folder containing benchmark files and network statistic files can be found in benchmark_data.

The fastest configurations that fit the specified criteria will be printed, additionally a graph of the fastest configuration showing per layer latencies and network overheads will be saved to the working directory.