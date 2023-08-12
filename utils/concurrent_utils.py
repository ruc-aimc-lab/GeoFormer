# -*- coding: utf-8 -*-
from concurrent.futures import ThreadPoolExecutor

def concurrent_process(func, inputs, num_threads):
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as t:
        for batch_res in t.map(func, inputs):
            results.append(batch_res)
    return results

def concurrent_process_xy(func, num_threads, input_xs, input_ys, labels):
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as t:
        for res in t.map(func, input_xs, input_ys, labels):
            results.append(res)
    return results

# def concurrent_process_ljz(func, num_threads, input_xs, input_ys, labels, deivce):
#     results = []
#     with ThreadPoolExecutor(max_workers=num_threads) as t:
#         for res in t.map(func, input_xs, input_ys, labels, deivce):
#             results.append(res)
#     return results

def concurrent_process_xyz(func, num_threads, input_xs, input_ys, input_zs):
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as t:
        for res in t.map(func, input_xs, input_ys, input_zs):
            results.append(res)
    return results