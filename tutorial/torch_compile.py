import numpy as np
import torch

torch._logging.set_logs(graph_code=False)


def foo3(x):
    y = x + 1
    z = torch.nn.functional.relu(y)
    u = z * 2
    return u


opt_foo3 = torch.compile(foo3)


def timed(fn):
    start = torch.mps.Event(enable_timing=True)
    end = torch.mps.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.mps.synchronize()
    return result, start.elapsed_time(end) / 1000


inp = torch.randn(4096, 4096)


eager_times = []
for i in range(10):
    _, eager_time = timed(lambda: foo3(inp))
    eager_times.append(eager_time)
    print(f'eager time {i}: {eager_time}')
print('~' * 10)

compile_times = []
for i in range(10):
    _, compile_time = timed(lambda: opt_foo3(inp))
    compile_times.append(compile_time)
    print(f'compile time {i}: {compile_time}')
print('~' * 10)


eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
assert speedup > 1
print(
    f'(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x'
)
print('~' * 10)
