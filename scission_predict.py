# Scission Prediction
# Creates scenarios using benchmark data created from Scission Benchmark
# Allows querying to reveal fastest scenarios
# Author: Luke Lockhart

import argparse
import csv
import fnmatch
import os
import pickle
from enum import Enum
from typing import List, Dict


class DeviceType(Enum):
    DEVICE = 1
    EDGE = 2
    CLOUD = 3


class LayerBenchmark:
    def __init__(self):
        self.model = ""
        self.platform = ""
        self.input_layer = 0
        self.output_layer = 0
        self.second_prediction = 0
        self.output_size = 0


class Device:
    def __init__(self, name: str, device_type: DeviceType):
        self.name = name
        self.type = device_type
        self.bandwidth = 0
        self.ping = 0
        self.benchmarks: Dict(value, Scenario_result) = {}


class Scenario:
    global input_size

    def __init__(self):
        self.device = ""  # Name of the device
        self.device_block = (-1, -1)
        self.device_layers = None
        self.device_output_size = input_size
        self.edge = None  # Name of edge if present
        self.edge_block = (-1, -1)
        self.edge_layers = None
        self.edge_output_size = 0
        self.cloud = None  # Name of cloud if present
        self.cloud_block = (-1, -1)
        self.cloud_layers = None
        self.total_processing_time = 0
        self.config = ""
        self.application = ""

    def __eq__(self, other):
        if not isinstance(other, Scenario):
            return NotImplemented

        return self.config == other.config

    def __hash__(self):
        return hash((self.config, self.config))


class NetworkStats:

    def __init__(self, ping, bandwidth):
        self.ping = ping
        self.bandwidth = bandwidth


# Loads pickles input file
def load_data(filename):
    try:
        with open(filename) as f:
            x = pickle.load(f)
    except:
        x = []
    return x


# Returns biggest number between two numbers
def get_biggest(x, y):
    if x > y:
        return x
    else:
        return y


# Creates possible distributions using data from benchmark files
def create_splits(length):
    device_splits = []
    edge_splits = []
    cloud_splits = []

    for x in range(-1, length):
        for y in range(-1, length):
            if y > x or y == -1:

                z = -1
                if x is not length - 1 and y is not length - 1:
                    z = length - 1

                x_layers = ("-1", "-1")
                if x != -1:
                    x_layers = ("0", str(x))

                y_layers = ("-1", "-1")
                if y != -1:
                    y_layers = (str(x + 1), str(y))

                z_layers = ("-1", "-1")
                if z != -1:
                    z_layers = (str(get_biggest(x, y) + 1), str(z))

                device_splits.append(x_layers)
                edge_splits.append(y_layers)
                cloud_splits.append(z_layers)

    return device_splits, edge_splits, cloud_splits


# Calculates the time for a distribution to execute along with the output of that block
def get_time(result: List[LayerBenchmark], start_index, end_index):
    total = 0
    layers = []

    for x in range(start_index, end_index + 1):
        total += result[x].second_prediction
        layers.append((result[x].input_layer, result[x].output_layer, result[x].second_prediction))

    output_size = result[end_index].output_size

    return total, output_size, layers


# Creates all possible scenarios across all loaded devices
def create_scenarios(application: str, device_devices, edge_devices, cloud_devices):
    scenarios = []

    device_splits, edge_splits, cloud_splits = create_splits(len(devices[0].benchmarks[application]))

    for device_device in device_devices:
        for idx_edge, edge_device in enumerate(edge_devices):
            for idx_cloud, cloud_device in enumerate(cloud_devices):

                for x in range(len(device_splits)):

                    scenario = Scenario()
                    scenario.application = application

                    device_time = 0
                    edge_time = 0
                    cloud_time = 0

                    scenario.device = device_device.name

                    if int(device_splits[x][0]) != -1:
                        device_time, output, device_layers = get_time(device_device.benchmarks[application],
                                                                      int(device_splits[x][0]),
                                                                      int(device_splits[x][1]))

                        scenario.device_layers = device_layers
                        scenario.device_output_size = output
                        scenario.device_block = (device_layers[0][0], device_layers[-1][1])

                    if int(edge_splits[x][0]) != -1:
                        edge_time, output, edge_layers = get_time(edge_device.benchmarks[application],
                                                                  int(edge_splits[x][0]),
                                                                  int(edge_splits[x][1]))

                        scenario.edge = edge_device.name
                        scenario.edge_layers = edge_layers
                        scenario.edge_output_size = output
                        scenario.edge_block = (edge_layers[0][0], edge_layers[-1][1])

                    if int(cloud_splits[x][0]) != -1:
                        cloud_time, unused, cloud_layers = get_time(cloud_device.benchmarks[application],
                                                                    int(cloud_splits[x][0]),
                                                                    int(cloud_splits[x][1]))

                        scenario.cloud = cloud_device.name
                        scenario.cloud_layers = cloud_layers
                        scenario.cloud_block = (cloud_layers[0][0], cloud_layers[-1][1])

                    if edge_time is 0 and cloud_time is 0:
                        scenario.device_output_size = 0
                    elif cloud_time is 0:
                        scenario.edge_output_size = 0

                    total_time = device_time + edge_time + cloud_time
                    scenario.total_processing_time = total_time

                    scenario.config = "Device(" + str(scenario.device) + ") = " + str(
                        scenario.device_block[0]) + " - " + str(
                        scenario.device_block[1]) + " Edge(" + str(scenario.edge) + ") = " + str(
                        scenario.edge_block[0]) + " - " + str(
                        scenario.edge_block[1]) + " Cloud(" + str(scenario.cloud) + ") = " + str(
                        scenario.cloud_block[0]) + " - " + str(
                        scenario.cloud_block[1])

                    scenarios.append(scenario)

    return scenarios


# Returns a list of results, sorted by execution time
def get_predictions_list_execution(scenarios: [Scenario]):
    global list_count
    outputs = []
    scenarios_sorted = []

    for _ in range(list_count):
        outputs.append(None)

    s: Scenario
    for s in scenarios:
        total_time = s.total_processing_time + get_transfer_overhead(s)

        for idx, result in enumerate(outputs[:list_count]):
            if result is None or result[0] > total_time:
                data = (total_time,
                        f"{round(total_time, 4)}s - {s.config} - {round(bytes_to_megabytes(s.device_output_size + s.edge_output_size), 4)}MB")
                outputs.insert(idx, data)
                scenarios_sorted.insert(idx, s)
                break

    return outputs[:list_count], scenarios_sorted[:list_count]


# Calcualtes the transfer latency for a scenario
def get_transfer_overhead(s: Scenario):
    transfer_overhead = 0

    filesize_to_send = s.device_output_size

    if s.edge is not None:
        stats: NetworkStats
        stats = device_stats[(s.device, s.edge)]

        transfer_overhead += (stats.ping + (filesize_to_send / stats.bandwidth))
    elif s.cloud is not None:
        stats: NetworkStats
        stats = device_stats[(s.device, s.cloud)]

        transfer_overhead += (stats.ping + (filesize_to_send / stats.bandwidth))
    elif s.edge is None and s.cloud is None:
        return transfer_overhead

    if s.edge is not None:
        if s.cloud is not None:
            stats: NetworkStats
            stats = device_stats[(s.edge, s.cloud)]

            transfer_overhead += (stats.ping + (s.edge_output_size / stats.bandwidth))

    return transfer_overhead


# Calculates the transfer latency between two devices given file size
def get_specific_transfer_overhead(source, destination, size):
    stats: NetworkStats
    stats = device_stats[(source, destination)]

    transfer_overhead = (stats.ping + (size / stats.bandwidth))

    return transfer_overhead


def megabytes_to_bytes(x):
    return x * 1000 * 1000


def bytes_to_megabytes(x):
    return x / 1000 / 1000


def megabits_to_bytes(x):
    return round(x * 125000)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Creates a distribution graph of model over the edge pipeline
def create_graph(s: Scenario, filename):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import ScalarFormatter
    import numpy as np
    from pathlib import Path

    plt.rcParams.update({'font.size': 35})

    file_path = "results/Best Graphs/"
    Path(file_path).mkdir(parents=True, exist_ok=True)

    bars = []
    execution_times = []
    colors = []
    handles = []

    if s.device_layers is not None:
        handles.append(mpatches.Patch(color='red', label='Device'))
        execution_times.append([time[2] * 1000 for time in s.device_layers])
        bars.append(
            [f"{result[0]}-{result[1]}" if result[0] != result[1] else f"{result[0]}" for result in s.device_layers])
        colors.append(["r" for _ in s.device_layers])

    if s.edge_layers is not None:
        handles.append(mpatches.Patch(color='green', label='Edge'))

        execution_times.append(get_specific_transfer_overhead(s.device, s.edge, s.device_output_size) * 1000)
        bars.append("COMM.")
        colors.append("yellow")

        execution_times.append([(time[2] if time[2] > 0 else 0.0006) * 1000 for time in s.edge_layers])
        bars.append(
            [f"{result[0]}-{result[1]}" if result[0] != result[1] else f"{result[0]}" for result in s.edge_layers])
        colors.append(["g" for _ in s.edge_layers])

        if s.edge_output_size is not 0:
            execution_times.append(get_specific_transfer_overhead(s.edge, s.cloud, s.edge_output_size) * 1000)
            bars.append("COMM.")
            colors.append("yellow")

    if s.cloud_layers is not None:
        handles.append(mpatches.Patch(color='blue', label='Cloud'))

        if s.edge_layers is None:
            execution_times.append(get_specific_transfer_overhead(s.device, s.cloud, s.device_output_size) * 1000)
            bars.append("COMM.")
            colors.append("yellow")

        execution_times.append([(time[2] if time[2] > 0 else 0.0006) * 1000 for time in s.cloud_layers])
        bars.append(
            [f"{result[0]}-{result[1]}" if result[0] != result[1] else f"{result[0]}" for result in s.cloud_layers])
        colors.append(["b" for _ in s.cloud_layers])

    width = 0.8

    execution_times = np.hstack(execution_times)
    bars = np.hstack(bars)
    colors = np.hstack(colors)

    ind = np.arange(len(bars))

    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax1.set_ylabel("Execution time (ms)", labelpad=10)
    ax1.bar(ind, execution_times, width, align='center', color=colors)
    ax1.set_xticks(ind)
    ax1.set_xticklabels(bars)

    plt.yscale("log")
    plt.setp(ax1.get_xticklabels(), rotation=270, horizontalalignment='center')

    for axis in [ax1.yaxis]:
        axis.set_major_formatter(ScalarFormatter())

    ax1.set_ylim(ymin=0)

    plt.legend(handles=handles)

    file_to_open = Path(dname) / (f"{filename}.png")
    plt.savefig(file_to_open, bbox_inches='tight')

    plt.show()


# Parse Args

parser = argparse.ArgumentParser(description="Scission Prediction")

parser.add_argument('-f', '--folder', dest='benchmark_folder', action='store', type=str, required=True,
                    help="Results Folder")
parser.add_argument('-s', '--statistics', dest='statistics_file', action='store', type=str, required=True,
                    help="Network Statistics File Name")

parser.add_argument('-m', '--model', dest='model', action='store', type=str, required=True, help="Model to predict for")
parser.add_argument('-rc', '-rcount', dest='count', action='store', type=int, required=False,
                    help="Number of results to return - Default 5")
parser.add_argument('-i', '-input', dest='input_size', action='store', type=int, required=False,
                    help="Input image size (KB) - Default 150")

parser.add_argument('-d', '--device', dest='device', action='store', type=str, required=False, help="Device criteria")
parser.add_argument('-du', '--deviceupload', dest='device_upload', action='store', type=str, required=False, help="Device upload limit")
parser.add_argument('-e', '--edge', dest='edge', action='store', type=str, required=False, help="Edge criteria")
parser.add_argument('-eu', '--edgeupload', dest='edge_upload', action='store', type=str, required=False, help="Edge upload limit")
parser.add_argument('-c', '--cloud', dest='cloud', action='store', type=str, required=False, help="Cloud criteria")

args = parser.parse_args()

benchmark_folder = args.benchmark_folder
network_statistics_file = args.statistics_file

if args.count is not None:
    list_count = args.count
else:
    list_count = 5

# Parse the device specific criteria
criteria_devices_inc = []
criteria_devices_excl = []
criteria_device_layers_inc = []
criteria_device_layers_excl = []

if args.device is not None:
    for c in args.device.split(","):
        c = c.strip()

        if c.isdigit() or c == "-1":
            criteria_device_layers_inc.append(int(c))
        elif c[0] == "!" and c[1:].isdigit() or c[1:] == "-1":
            criteria_device_layers_excl.append(int(c[1:]))
        elif c[0] == "!":
            criteria_devices_excl.append(c[1:])
        else:
            criteria_devices_inc.append(c)

# Parse the edge specific criteria
criteria_edges_inc = []
criteria_edges_excl = []
criteria_edge_layers_inc = []
criteria_edge_layers_excl = []

if args.edge is not None:
    for c in args.edge.split(","):
        c = c.strip()

        if c.isdigit() or c == "-1":
            criteria_edge_layers_inc.append(int(c))
        elif c[0] == "!" and c[1:].isdigit() or c[1:] == "-1":
            criteria_edge_layers_excl.append(int(c[1:]))
        elif c[0] == "!":
            criteria_edges_excl.append(c[1:])
        else:
            criteria_edges_inc.append(c)

# Parse the cloud specific criteria
criteria_clouds_inc = []
criteria_clouds_excl = []
criteria_cloud_layers_inc = []
criteria_cloud_layers_excl = []

if args.cloud is not None:
    for c in args.cloud.split(","):
        c = c.strip()

        if c.isdigit() or c == "-1":
            criteria_cloud_layers_inc.append(int(c))
        elif c[0] == "!" and c[1:].isdigit() or c[1:] == "-1":
            criteria_cloud_layers_excl.append(int(c[1:]))
        elif c[0] == "!":
            criteria_clouds_excl.append(c[1:])
        else:
            criteria_clouds_inc.append(c)

# Misc criteria

if args.input_size is not None:
    input_size = megabytes_to_bytes(args.input_size)
else:
    input_size = megabytes_to_bytes(0.15)
    
if args.device_upload is not None:
    device_upload = megabytes_to_bytes(float(args.device_upload))
else:
    device_upload = None 

if args.edge_upload is not None:
    edge_upload = megabytes_to_bytes(float(args.edge_upload))
else:
    edge_upload = None

if args.model is not None:
    application = args.model.lower()

# End Parse Args

# Set path to script directory then to benchmark_data folder
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir(benchmark_folder)
devices = []

print("[+] Loading Data")
for filename in os.listdir(os.getcwd()):
    if not fnmatch.fnmatch(filename, "*-*.dat"):
        continue

    pickle_in = open(filename, "rb")
    data = pickle.load(pickle_in)

    full_name = filename.split(".")[0]
    device_type, name = full_name.split("-")

    device_type_enum = DeviceType[device_type.upper()]

    new_device = Device(name, device_type_enum)
    new_device.benchmarks = data

    devices.append(new_device)

if len(devices) is 0:
    print("[+] No .dat benchmark files stored in benchmark_data. Exiting...")
    exit()

device_stats = {}
with open(network_statistics_file, newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        device_stats[(row[0], row[1])] = NetworkStats(float(row[2]) / 1000, megabits_to_bytes(float(row[3])))

device = [d for d in devices if d.type == DeviceType.DEVICE]
edge = [d for d in devices if d.type == DeviceType.EDGE]
cloud = [d for d in devices if d.type == DeviceType.CLOUD]

print("[+]" + str(len(devices)) + " devices loaded.")
print("[+]" + str(len(device)) + " device.")
print("[+]" + str(len(edge)) + " edge.")
print("[+]" + str(len(cloud)) + " cloud.")

scenarios_raw = create_scenarios(application, device, edge, cloud)
s: Scenario
scenarios = set(scenarios_raw)

if list_count > len(scenarios):
    list_count = len(scenarios)

# Device filtering
if criteria_devices_inc:
    scenarios = [s for s in scenarios if s.device in criteria_devices_inc and s.device_layers is not None]
if criteria_devices_excl:
    scenarios = [s for s in scenarios if s.device not in criteria_devices_excl and s.device_layers is not None]
if criteria_device_layers_inc:
    scenarios = [s for s in scenarios if
                 all(x in range(s.device_block[0], s.device_block[1] + 1) for x in criteria_device_layers_inc)]
if criteria_device_layers_excl:
    scenarios = [s for s in scenarios if
                 all(x not in range(s.device_block[0], s.device_block[1] + 1) for x in criteria_device_layers_excl)]
if device_upload is not None:
    scenarios = [s for s in scenarios if s.device_output_size <= device_upload]
                

# Edge filtering
if criteria_edges_inc:
    scenarios = [s for s in scenarios if s.edge in criteria_edges_inc and s.edge_layers is not None]
if criteria_edges_excl:
    scenarios = [s for s in scenarios if s.edge not in criteria_edges_excl and s.edge_layers is not None]
if criteria_edge_layers_inc:
    scenarios = [s for s in scenarios if
                 all(x in range(s.edge_block[0], s.edge_block[1] + 1) for x in criteria_edge_layers_inc)]
if criteria_edge_layers_excl:
    scenarios = [s for s in scenarios if
                 all(x not in range(s.edge_block[0], s.edge_block[1] + 1) for x in criteria_edge_layers_excl)]
if edge_upload is not None:
    scenarios = [s for s in scenarios if s.edge_output_size <= edge_upload]

# Cloud filtering
if criteria_clouds_inc:
    scenarios = [s for s in scenarios if s.cloud in criteria_clouds_inc and s.cloud_layers is not None]
if criteria_clouds_excl:
    scenarios = [s for s in scenarios if s.cloud not in criteria_clouds_excl and s.cloud_layers is not None]
if criteria_cloud_layers_inc:
    scenarios = [s for s in scenarios if
                 all(x in range(s.cloud_block[0], s.cloud_block[1] + 1) for x in criteria_cloud_layers_inc)]
if criteria_cloud_layers_excl:
    scenarios = [s for s in scenarios if
                 all(x not in range(s.cloud_block[0], s.cloud_block[1] + 1) for x in criteria_cloud_layers_excl)]

results, sorted_scenarios = get_predictions_list_execution(scenarios)

if results[0] is None:
    print("No results for the specified configuration")
    exit()

create_graph(sorted_scenarios[0], f"{application}-CommandLine")

for idx, result in enumerate(results):
    if result is not None:
        print(f"[{idx + 1}] {result[1]}")
