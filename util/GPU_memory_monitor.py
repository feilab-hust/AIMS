import os
import psutil
import pynvml

def get_gpu_mem_info(gpu_id = 0):
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        raise ValueError(r'invalid gpu_id:{}'.format(gpu_id))
    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    remained = round(meminfo.free / 1024 / 1024, 2)
    print(f'Total gpu memory: {total}, index: {gpu_id}')
    print(f'Used gpu memory: {used}, index: {gpu_id}')
    print(f'Remained gpu memory: {remained}, index: {gpu_id}')
    return total, used, remained

if __name__ == '__main__':
    get_gpu_mem_info(0)