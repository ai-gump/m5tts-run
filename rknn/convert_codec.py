#!/usr/bin/env python3
"""codec_decoder_cached.onnx -> codec.rknn（RK3566 NPU，Phase 1 探路脚本，跑在 Ubuntu 上）。

用法：
  python3 convert_codec.py codec_decoder_cached.onnx fp   # 非量化，先验证连通
  python3 convert_codec.py codec_decoder_cached.onnx i8   # INT8 量化（自动生成随机校准集）

输出固定写到 ./rknn_out/。报错时把完整终端输出回传，不要截断。
"""
import os
import sys
import numpy as np
from rknn.api import RKNN

MODE_QUANT = {'fp': False, 'i8': True}


def make_calib_dir(inputs_meta, samples=8):
    """多输入校准集：一个样本一个目录，目录内按输入名放 .npy。随机分布，仅供性能探路。"""
    root = 'rknn_calib'
    lines = []
    for s in range(samples):
        d = os.path.join(root, f's{s}')
        os.makedirs(d, exist_ok=True)
        for name, shape, dtype in inputs_meta:
            if 'int' in dtype:
                np.save(os.path.join(d, name + '.npy'),
                        np.random.randint(0, 2048, size=shape).astype(np.int64))
            else:
                np.save(os.path.join(d, name + '.npy'),
                        np.random.randn(*shape).astype(np.float32) * 0.1)
        lines.append(d)
    with open('rknn_calib.txt', 'w') as f:
        f.write('\n'.join(lines))
    return 'rknn_calib.txt'


def probe_inputs(onnx_path):
    """粗读输入清单（onnx 包直接读图，不依赖 rknn）。int64 输入是 NPU 风险点，打印出来。"""
    import onnx
    m = onnx.load(onnx_path)
    meta = []
    for i in m.graph.input:
        dims = [(d.dim_value if d.dim_value > 0 else 1) for d in i.type.tensor_type.shape.dim]
        et = i.type.tensor_type.elem_type  # 7=int64, 1=float32
        meta.append((i.name, dims, 'int64' if et == 7 else 'float32'))
        print(f'  input {i.name} dims~{dims} {"int64<--风险" if et == 7 else ""}')
    return meta


def build(src, quant):
    tag = 'i8' if quant else 'fp'
    print(f'== [{tag}] source={src}')
    inputs_meta = probe_inputs(src)

    rknn = RKNN(verbose=False)
    rknn.config(target_platform='rk3566', optimization_level=3)
    ret = rknn.load_onnx(model=src)
    print(f'[{tag}] load_onnx -> {ret}')
    if ret != 0:
        rknn.release(); return 1

    dataset = None
    if quant:
        dataset = make_calib_dir(inputs_meta)
        print(f'[{tag}] calib dataset: {dataset}')
    ret = rknn.build(do_quantization=quant, dataset=dataset)
    print(f'[{tag}] build(quant={quant}) -> {ret}')
    if ret != 0:
        rknn.release(); return 1

    out = os.path.join('rknn_out', f'codec_{tag}.rknn')
    os.makedirs('rknn_out', exist_ok=True)
    ret = rknn.export_rknn(out)
    print(f'[{tag}] export -> {ret} : {out} ({os.path.getsize(out)/1e6:.1f} MB)')
    rknn.release()
    return 0 if ret == 0 else 1


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[2] not in MODE_QUANT:
        print(__doc__); sys.exit(2)
    sys.exit(build(sys.argv[1], MODE_QUANT[sys.argv[2]]))
