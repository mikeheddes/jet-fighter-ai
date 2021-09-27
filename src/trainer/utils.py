import os
import argparse
import psutil
import datetime
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

def do_every(interval, step):
    return step % interval == interval - 1


def get_preferred_device():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default="cuda:0")

    return parser.parse_args().device


def host_metrics(device, step):
    gpu_mem = torch.cuda.memory_allocated(device) / 1e6
    yield ("scalar", "host/gpu_memory_usage", gpu_mem, step.value)

    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1e6
    yield ("scalar", "host/cpu_memory_usage", cpu_mem, step.value)



def run_writer(metric_queue: mp.Queue, name):
    timezone = datetime.timezone.utc
    current_date = datetime.datetime.now(timezone)
    version = current_date.strftime("d%Y_%m_%d-t%H_%M_%S")
    writer = SummaryWriter(os.path.join("../runs", name, version))

    while True:
        metric = metric_queue.get()
        m_type = metric[0]
        if m_type == "histogram":
            writer.add_histogram(*metric[1:])
        if m_type == "scalar":
            writer.add_scalar(*metric[1:])
        if m_type == "video":
            writer.add_video(*metric[1:])
        if m_type == "graph":
            writer.add_graph(*metric[1:])
