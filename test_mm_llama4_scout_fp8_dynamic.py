"""
# To run OSS libraries (vLLM CUTLASS):
python ~/ncu_trace.py --exclude-kernel-name-substr=elementwise python test_mm_llama4_scout_fp8_dynamic.py --n-iter=10
python test_mm_llama4_scout_fp8_dynamic.py --n-iter=50 --enable-profiler

alias buckrun='(rm -rf /tmp/willfeng/deep_gemm || true) && buck2 run @//mode/{opt,inplace} -c fbcode.enable_gpu_sections=true -c fbcode.nvcc_arch=h100a -c fbcode.platform010_cuda_version=12.8'

# To run fbcode libraries (torch._scaled_mm, FBGEMM CUTLASS, FBGEMM DeepGEMM) with ncu_trace:
# NOTE: need these diffs:
# D73088151: this is to revert the internal DeepGEMM repo state to be back to Natalia's row-wise scaling branch. 
# D73088152: scripts BUCK target
# D73220992: DeepGEMM fp8 row-wise with fast-accum
cp test_mm_llama4_scout_fp8_dynamic.py ~/local/fbsource/fbcode/scripts/willfeng/test_mm_llama4_scout_fp8_dynamic.py && \
pushd ~/local/fbsource/fbcode && \
(python ~/ncu_trace.py --exclude-kernel-name-substr=elementwise buckrun @mode/opt //scripts/willfeng:test_mm_llama4_scout_fp8_dynamic -- --n-iter=1 || true) && \
popd

# To run fbcode libraries without ncu_trace (and with PyTorch profiler):
# NOTE: must revert D72751398 to use FBGEMM DeepGEMM with rowwise scaling (ngimel): hg backout D72751398; and also need D73088152 for scripts BUCK target
cp test_mm_llama4_scout_fp8_dynamic.py ~/local/fbsource/fbcode/scripts/willfeng/test_mm_llama4_scout_fp8_dynamic.py && \
pushd ~/local/fbsource/fbcode && \
(buckrun @mode/opt //scripts/willfeng:test_mm_llama4_scout_fp8_dynamic -- --n-iter=50 --enable-profiler || true) && \
popd

cp test_mm_llama4_scout_fp8_dynamic.py ~/local/fbsource/fbcode/scripts/willfeng/test_mm_llama4_scout_fp8_dynamic.py && \
pushd ~/local/fbsource/fbcode && \
(buckrun @mode/opt //scripts/willfeng:test_mm_llama4_scout_fp8_dynamic -- --n-iter=15 --enable-profiler-trace || true) && \
popd

# To run accuracy check:
cp test_mm_llama4_scout_fp8_dynamic.py ~/local/fbsource/fbcode/scripts/willfeng/test_mm_llama4_scout_fp8_dynamic.py && \
pushd ~/local/fbsource/fbcode && \
(buckrun @mode/opt //scripts/willfeng:test_mm_llama4_scout_fp8_dynamic -- --check-accuracy || true) && \
popd
"""

"""
How to collect NCU metrics for mm kernels:
1. Run the llama model with command: `wp <vllm command> >output.txt 2>&1` (make sure vLLM is using our custom FlopCounterMode),
2. Extract the mm kernel shapes from the log file with `python ~/extract_kernel_shapes.py <kernel_name> output.txt >kernel_shapes.csv 2>&1` (optionally only extract top-N-flops kernels),
3. Put the mm kernel shapes into this file and run the script with `ncu_trace`.
4. Run `python ~/ncu_trace_parser.py <combined_CSV_file>` to get the NCU metrics summary.
"""

import torch
from collections import defaultdict
import csv
from io import StringIO
from datetime import datetime
import argparse
import contextlib


import torch
from typing import Optional

# Global dictionary to store runtime results
runtime_results = {}

# Inspiration taken from:
# https://gist.github.com/malfet/7874d96b99670c3da83cbb779ab770c6
# https://github.com/pytorch/pytorch/pull/130059/files
# https://github.com/vllm-project/vllm/blob/3e397a948431755b668fb95af679ec0a33daea89/tests/kernels/test_cutlass.py#L147


def to_fp8(tensor: torch.Tensor, fp8_dtype):
    finfo = torch.finfo(fp8_dtype)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=fp8_dtype)


def baseline_scaled_mm(a: torch.Tensor,
                       b: torch.Tensor,
                       scale_a: torch.Tensor,
                       scale_b: torch.Tensor,
                       out_dtype: type[torch.dtype],
                       bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    # We treat N-dimensional group scaling as extended numpy-style broadcasting
    # in numpy simply stretches dimensions with an extent of 1 to match the
    # the target shape by repeating the data along that dimension (broadcasting)
    # , we extend these semantics to say if the extent of a dimension in the
    # source shape is not 1 and does not match the target shape we repeat each
    # element along that dimension src_shape[dim] // target_shape[dim] times
    # example if we have:
    #       a = [[1, 2], and target_shape = (2, 4)
    #            [3, 4]]
    # then we would expand a to:
    #       a = [[1, 1, 2, 2],
    #            [3, 3, 4, 4]]
    # NOTE this function this function does not explicitly broadcast dimensions
    # with an extent of 1, since this can be done implicitly by pytorch
    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s % t.shape[i] == 0
                t = t.unsqueeze(i + 1)\
                  .expand(*t.shape[:i+1], s // t.shape[i], *t.shape[i+1:])\
                  .flatten(i, i + 1)
        return t

    scale_a = group_broadcast(scale_a, a.shape)
    scale_b = group_broadcast(scale_b, b.shape)

    output = torch.mm((scale_a * a.to(dtype=torch.float32)),
                      (scale_b * b.to(dtype=torch.float32))).to(out_dtype)

    if bias is not None:
        output = output + bias

    return output


# -1 means full extent in that dimension
PER_TOKEN_GROUP_SHAPE = (1, -1)
PER_OUT_CH_GROUP_SHAPE = (-1, 1)


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def group_scale_helper(shape, group_shape):
    return [shape[i] if s < 0 else s for i, s in enumerate(group_shape)]


def scale_shape(shape, group_shape):
    assert len(shape) == len(group_shape)
    group_shape = group_scale_helper(shape, group_shape)
    return tuple(
        cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


def run_fp8_mm_benchmark(m, k, n, libs, n_iter, enable_profiler, enable_profiler_trace, check_accuracy=False):
    # Test for fp8 GEMM kernel with per-token activation quantization
    # and per-output channel weight quantization.

    # Initialize output variables
    outs = {}

    for lib in libs:
        assert lib in ["vllm.fast_accum", "torch._scaled_mm.fast_accum", "torch._scaled_mm.slow_accum", "fbgemm_cutlass.fast_accum", "fbgemm_deepgemm.slow_accum", "fbgemm_deepgemm.fast_accum"]

        torch.manual_seed(12345)

        device = torch.device("cuda")
        # Output and input dtypes
        out_dtype = torch.bfloat16
        base_dtype = torch.bfloat16
        a_dtype = torch.float8_e4m3fn
        b_dtype = torch.float8_e4m3fn
        scale_dtype = torch.float32  # Both scales use same type (float32)

        a_scale_group_shape = PER_TOKEN_GROUP_SHAPE
        b_scale_group_shape = PER_OUT_CH_GROUP_SHAPE

        a = to_fp8(torch.randn((m, k), device=device, dtype=base_dtype), a_dtype)
        b = to_fp8(torch.randn((n, k), device=device, dtype=base_dtype).t(), b_dtype)

        a_scales_shape = scale_shape(a.shape, a_scale_group_shape)
        b_scales_shape = scale_shape(b.shape, b_scale_group_shape)

        scale_a = (torch.randn(a_scales_shape, device=device, dtype=scale_dtype))
        scale_b = (torch.randn(b_scales_shape, device=device, dtype=scale_dtype))

        assert a.stride() == (k, 1)
        assert b.stride() == (1, k)

        assert scale_a.shape == (m, 1)
        assert scale_a.stride() == (1, 1)
        # NOTE: this doesn't exactly match the extracted scale_b shape and stride from FlopCounterMode, but should be okay
        assert scale_b.shape == (1, n)
        assert scale_b.stride() == (n, 1)

        iter_count = 0
        num_loops = n_iter

        def trace_handler(prof):
            # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=1000))
            kernel_entry = next(
                (event for event in prof.key_averages() 
                if event.cpu_time_total == 0 and event.key == lib), 
                None
            )
            assert kernel_entry is not None
            cuda_time_us = kernel_entry.cuda_time
            print(f"lib,m,k,n,avg_cuda_time_us: {lib},{m},{k},{n},{cuda_time_us}")
            
            # Store runtime in global dictionary
            runtime_results[(lib, m, k, n)] = cuda_time_us

            if enable_profiler_trace:
                import datetime
                import subprocess
                import getpass
                import os

                username = getpass.getuser()
                timestamp = int(datetime.datetime.now().timestamp())
                cur_working_dir = os.getcwd()
                trace_dir = os.path.join(cur_working_dir, "gpu_traces")
                os.makedirs(trace_dir, exist_ok=True)
                trace_path = f"{cur_working_dir}/gpu_traces/trace_{timestamp}.json"
                manifold_path = f"gpu_traces/tree/{username}/vllm/trace_{timestamp}.json"
                
                prof.export_chrome_trace(trace_path)
                
                # Run the manifold upload command
                subprocess.run(
                    ["manifold", "mkdir", f"gpu_traces/tree/{username}/vllm"],
                    capture_output=True,
                    text=True,
                )
                result = subprocess.run(
                    ["manifold", "put", "--threads", "20", trace_path, manifold_path],
                    capture_output=True,
                    text=True
                )
                
                print(f"GPU trace local file: {trace_path}")
                if result.returncode == 0:
                    print(f"GPU trace URL (requires VPN): https://interncache-all.fbcdn.net/manifold/perfetto-artifacts/tree/ui/index.html#!/?url=https://interncache-all.fbcdn.net/manifold/gpu_traces/tree/{username}/vllm/trace_{timestamp}.json")
                else:
                    print(f"Failed to upload trace: {result.stderr}")

        # Setup profiler context based on the flag
        if enable_profiler:
            # Calculate profiler schedule based on n_iter
            warmup_iters = 10
            active_iters = n_iter - warmup_iters - 2 # Remaining iters after skip_first=2 and warmup

            profiler_schedule = torch.profiler.schedule(skip_first=2, wait=0, warmup=warmup_iters, active=active_iters)
            profiler_context = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=profiler_schedule,
                on_trace_ready=trace_handler,
                with_stack=False,
                record_shapes=False,
            )
        else:
            profiler_context = contextlib.nullcontext()

        with profiler_context as prof:
            while iter_count < num_loops:
                if lib == "torch._scaled_mm.fast_accum":
                    with torch.profiler.record_function(lib):
                        outs[lib] = torch._scaled_mm(a, b, out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b, use_fast_accum=True)
                elif lib == "torch._scaled_mm.slow_accum":
                    with torch.profiler.record_function(lib):
                        outs[lib] = torch._scaled_mm(a, b, out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b, use_fast_accum=False)
                elif lib == "vllm.fast_accum":
                    # vLLM CUTLASS (NOTE: fast_accum=True by default)
                    try:
                        from vllm import _custom_ops
                        with torch.profiler.record_function(lib):
                            outs[lib] = _custom_ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, None)
                    except ImportError:
                        print("vllm cutlass_scaled_mm: SKIPPED (vllm not installed)")
                        outs[lib] = None # Ensure it stays None if skipped
                    except Exception as e:
                        raise
                elif lib == "fbgemm_cutlass.fast_accum":
                    # FBGEMM CUTLASS (NOTE: fast_accum=True by default)
                    try:
                        assert hasattr(torch.ops, "fbgemm") and hasattr(torch.ops.fbgemm, "f8f8bf16_rowwise")
                        # FBGEMM fp8 rowwise kernel expects B to be transposed
                        # Ref: https://www.internalfb.com/code/fbsource/[0a767c15edca9249b5d859898cab2375cba015fc]/fbcode/deeplearning/fbgemm/fbgemm_gpu/experimental/gen_ai/test/quantize/quantize_test.py?lines=318
                        # Original: x @ w.T
                        b_t = b.t()
                        scale_b_restrided = scale_b.as_strided((n,), (1,))
                        with torch.profiler.record_function(lib):
                            outs[lib] = torch.ops.fbgemm.f8f8bf16_rowwise(a, b_t, scale_a, scale_b_restrided)
                    except AssertionError as e:
                        print("fbgemm CUTLASS: SKIPPED (fbgemm not installed)")
                        outs[lib] = None # Ensure it stays None if skipped
                    except Exception as e:
                        raise
                elif lib == "fbgemm_deepgemm.slow_accum":
                    # FBGEMM DeepGEMM (NOTE: fast_accum=False by default)
                    try:
                        # Ref: FBGEMM integration: https://www.internalfb.com/diff/D71748927
                        from deep_gemm.jit_kernels import gemm_fp8_fp8_bf16_nt, get_col_major_tma_aligned_tensor
                        out_fbgemm_deepgemm_slow_accum = torch.empty(m, n, device=device, dtype=out_dtype)
                        scale_a_deepgemm = get_col_major_tma_aligned_tensor(scale_a.as_strided((m,), (1,)), rowwise_scaling=True)
                        b_t = b.t()
                        scale_b_restrided = scale_b.as_strided((n,), (1,))
                        with torch.profiler.record_function(lib):
                            gemm_fp8_fp8_bf16_nt((a, scale_a_deepgemm), (b_t, scale_b_restrided), out_fbgemm_deepgemm_slow_accum, fast_accum=False)
                        outs[lib] = out_fbgemm_deepgemm_slow_accum
                    except ImportError as e:
                        print("fbgemm DeepGEMM: SKIPPED (deep_gemm not installed)")
                        outs[lib] = None # Ensure it stays None if skipped
                    except Exception as e:
                        raise
                elif lib == "fbgemm_deepgemm.fast_accum":
                    # FBGEMM DeepGEMM (NOTE: fast_accum=False by default)
                    try:
                        # Ref: FBGEMM integration: https://www.internalfb.com/diff/D71748927
                        from deep_gemm.jit_kernels import gemm_fp8_fp8_bf16_nt, get_col_major_tma_aligned_tensor
                        out_fbgemm_deepgemm_slow_accum = torch.empty(m, n, device=device, dtype=out_dtype)
                        scale_a_deepgemm = get_col_major_tma_aligned_tensor(scale_a.as_strided((m,), (1,)), rowwise_scaling=True)
                        b_t = b.t()
                        scale_b_restrided = scale_b.as_strided((n,), (1,))
                        with torch.profiler.record_function(lib):
                            gemm_fp8_fp8_bf16_nt((a, scale_a_deepgemm), (b_t, scale_b_restrided), out_fbgemm_deepgemm_slow_accum, fast_accum=True)
                        outs[lib] = out_fbgemm_deepgemm_slow_accum
                    except ImportError as e:
                        print("fbgemm DeepGEMM: SKIPPED (deep_gemm not installed)")
                        outs[lib] = None # Ensure it stays None if skipped
                    except Exception as e:
                        raise
                else:
                    raise NotImplementedError(f"Library {lib} not implemented")

                # Only step the profiler if it's enabled
                if enable_profiler:
                    prof.step()
                iter_count += 1

    def compare_accuracy_with_baseline(output_tensor: Optional[torch.Tensor], baseline_tensor: torch.Tensor, lib_name: str):
        """Helper function to check accuracy against baseline."""
        if output_tensor is not None:
            print(f"Checking {lib_name} vs Baseline...")
            try:
                # Compare against baseline with original tolerances
                torch.testing.assert_close(output_tensor, baseline_tensor, rtol=1e-2, atol=5e-2)
                print(f"  {lib_name} vs Baseline: PASSED (rtol=1e-2, atol=5e-2)")
            except AssertionError as e:
                print(f"  {lib_name} vs Baseline: FAILED (rtol=1e-2, atol=5e-2)\n{e}")
        else:
            print(f"Skipping baseline check for {lib_name} (output is None)")

    def check_pairwise_accuracy_helper(output1: torch.Tensor, output2: torch.Tensor, name1: str, name2: str):
        """Helper function to check accuracy between two library outputs."""
        print(f"Checking {name1} vs {name2}...")
        print("NOTE: differing by up to 0.25 abs diff is acceptable.")
        try:
            # Compare two library outputs with stricter tolerances
            torch.testing.assert_close(output1, output2, rtol=1e-4, atol=0.25)
            print(f"  {name1} vs {name2}: PASSED (rtol=1e-4, atol=0.25)")
        except AssertionError as e:
            print(f"  {name1} vs {name2}: FAILED (rtol=1e-4, atol=0.25)\n{e}")
        print("\n")

    if check_accuracy:
        print(f"\n--- Accuracy Check for (m={m}, k={k}, n={n}), lib={lib} ---")
        baseline = baseline_scaled_mm(a, b, scale_a, scale_b, out_dtype, None)

        # Store outputs with their names if they are not None
        # Retrieve outputs from the 'outs' dictionary
        valid_outputs = []
        for lib_name in libs:
            output_tensor = outs.get(lib_name) # Use .get() to handle missing keys gracefully
            if output_tensor is not None:
                valid_outputs.append((lib_name, output_tensor))
            else:
                print(f"Output for {lib_name} is None, skipping accuracy check.")

        print("\n--- Baseline Comparisons (rtol=1e-2, atol=5e-2) ---")
        # Call the helper function for each library's output against baseline
        for lib_name, output_tensor in valid_outputs:
             compare_accuracy_with_baseline(output_tensor, baseline, lib_name)

        print("\n--- Pairwise Comparisons (rtol=1e-4, atol=1e-4) ---")
        # Perform pairwise comparisons only if there are at least two valid outputs
        if len(valid_outputs) >= 2:
            for i in range(len(valid_outputs)):
                for j in range(i + 1, len(valid_outputs)):
                    name1, tensor1 = valid_outputs[i]
                    name2, tensor2 = valid_outputs[j]
                    check_pairwise_accuracy_helper(tensor1, tensor2, name1, name2)
        else:
            print("Skipping pairwise comparisons (less than two library outputs available).")

        print("\n" + "-" * (len(f"--- Accuracy Check for (m={m}, k={k}, n={n}), lib={libs} ---")))
    # Print pass message only if not doing accuracy check (which prints its own messages)
    elif not check_accuracy:
        print(f"Finished {libs} for shape (m={m}, k={k}, n={n})")

def run_shapes(n_iter, enable_profiler, enable_profiler_trace, check_accuracy, lib):
    assert torch.cuda.is_available()

    # Use 1 iteration and disable profiler if checking accuracy
    run_n_iter = 1 if check_accuracy else n_iter
    if check_accuracy:
        enable_profiler = False
        enable_profiler_trace = False
    elif enable_profiler_trace:
        enable_profiler = True
        check_accuracy = False
    elif enable_profiler:
        check_accuracy = False

    # Each entry is now a tuple of:
    # ((output_shape, output_stride, output_type),
    # ((input_a_shape, input_a_stride, input_a_type),
    #  (input_a_scale_shape, input_a_scale_stride, input_a_scale_type)),
    # ((input_b_shape, input_b_stride, input_b_type),
    #  (input_b_scale_shape, input_b_scale_stride, input_b_scale_type)))
    shapes = [
        # TP=2 Shapes
        (((64, 8192), (8192, 1), 'torch.bfloat16'),
        ((64, 5120), (5120, 1), 'torch.float8_e4m3fn'),
        ((5120, 8192), (1, 5120), 'torch.float8_e4m3fn')),

        (((64, 5120), (5120, 1), 'torch.bfloat16'),
        ((64, 4096), (4096, 1), 'torch.float8_e4m3fn'),
        ((4096, 5120), (1, 4096), 'torch.float8_e4m3fn')),

        (((64, 3584), (3584, 1), 'torch.bfloat16'),
        ((64, 5120), (5120, 1), 'torch.float8_e4m3fn'),
        ((5120, 3584), (1, 5120), 'torch.float8_e4m3fn')),

        (((64, 5120), (5120, 1), 'torch.bfloat16'),
        ((64, 2560), (2560, 1), 'torch.float8_e4m3fn'),
        ((2560, 5120), (1, 2560), 'torch.float8_e4m3fn')),

        # TP=4 Shapes
        (((64, 4096), (4096, 1), 'torch.bfloat16'),
        ((64, 5120), (5120, 1), 'torch.float8_e4m3fn'),
        ((5120, 4096), (1, 5120), 'torch.float8_e4m3fn')),

        (((64, 5120), (5120, 1), 'torch.bfloat16'),
        ((64, 2048), (2048, 1), 'torch.float8_e4m3fn'),
        ((2048, 5120), (1, 2048), 'torch.float8_e4m3fn')),

        (((64, 1792), (1792, 1), 'torch.bfloat16'),
        ((64, 5120), (5120, 1), 'torch.float8_e4m3fn'),
        ((5120, 1792), (1, 5120), 'torch.float8_e4m3fn')),

        (((64, 5120), (5120, 1), 'torch.bfloat16'),
        ((64, 1280), (1280, 1), 'torch.float8_e4m3fn'),
        ((1280, 5120), (1, 1280), 'torch.float8_e4m3fn')),

        # TP=8 Shapes
        (((64, 2048), (2048, 1), 'torch.bfloat16'),
        ((64, 5120), (5120, 1), 'torch.float8_e4m3fn'),
        ((5120, 2048), (1, 5120), 'torch.float8_e4m3fn')),

        (((64, 5120), (5120, 1), 'torch.bfloat16'),
        ((64, 1024), (1024, 1), 'torch.float8_e4m3fn'),
        ((1024, 5120), (1, 1024), 'torch.float8_e4m3fn')),

        (((64, 896), (896, 1), 'torch.bfloat16'),
        ((64, 5120), (5120, 1), 'torch.float8_e4m3fn'),
        ((5120, 896), (1, 5120), 'torch.float8_e4m3fn')),

        (((64, 5120), (5120, 1), 'torch.bfloat16'),
        ((64, 640), (640, 1), 'torch.float8_e4m3fn'),
        ((640, 5120), (1, 640), 'torch.float8_e4m3fn')),
    ]


    if check_accuracy:
        print(f"Running accuracy checks for {len(shapes)} shapes...")
    else:
        print(f"Running benchmarks for {len(shapes)} shapes with n_iter={run_n_iter}, profiler={'enabled' if enable_profiler else 'disabled'}...")

    libs_available = [
        # "torch._scaled_mm.fast_accum",
        # "torch._scaled_mm.slow_accum",
        # # "vllm.fast_accum",
        # "fbgemm_cutlass.fast_accum",
        "fbgemm_deepgemm.slow_accum",
        "fbgemm_deepgemm.fast_accum",
    ]
    if lib is not None:
        assert lib in libs_available, f"Library {lib} not found in available libraries: {libs_available}"
        libs_to_run = [lib]
    else:
        libs_to_run = libs_available

    # Track processed shapes for runtime comparison
    processed_shapes = []

    for i, shape_info in enumerate(shapes):
        # Extract components
        _, input_a_info, input_b_info = shape_info
        input_a_shape, _, _ = input_a_info
        input_b_shape, _, _ = input_b_info

        # Extract m, k, n from the input shapes
        m = input_a_shape[0]
        k = input_a_shape[1]
        n = input_b_shape[1] # Input B shape is (k, n)

        print(f"\n--- Processing Kernel {i+1} with (m={m}, k={k}, n={n}) ---")
        processed_shapes.append((m, k, n))

        for lib in libs_to_run:
             run_fp8_mm_benchmark(m, k, n, libs=libs_to_run, n_iter=run_n_iter, enable_profiler=enable_profiler, enable_profiler_trace=enable_profiler_trace, check_accuracy=check_accuracy)

    print("\nAll shapes processed.")

    # Print runtime comparison if running with the two DeepGEMM libraries
    if (not check_accuracy and enable_profiler and 
        set(libs_to_run) == {"fbgemm_deepgemm.slow_accum", "fbgemm_deepgemm.fast_accum"}):
        print("\n" + "="*80)
        print("RUNTIME COMPARISON: fbgemm_deepgemm.slow_accum vs fbgemm_deepgemm.fast_accum")
        print("="*80)
        print(f"{'Shape (m,k,n)':<20} {'slow_accum.runtime (us)':<15} {'fast_accum.runtime (us)':<15} {'Speedup':<10}")
        print("-"*60)
        
        for m, k, n in processed_shapes:
            slow_key = ("fbgemm_deepgemm.slow_accum", m, k, n)
            fast_key = ("fbgemm_deepgemm.fast_accum", m, k, n)
            
            if slow_key in runtime_results and fast_key in runtime_results:
                slow_time = runtime_results[slow_key]
                fast_time = runtime_results[fast_key]
                speedup = slow_time / fast_time
                shape_str = f"({m},{k},{n})"
                print(f"{shape_str:<20} {slow_time:<15.2f} {fast_time:<15.2f} {speedup:<10.3f}x")
            else:
                shape_str = f"({m},{k},{n})"
                print(f"{shape_str:<20} {'N/A':<15} {'N/A':<15} {'N/A':<10}")
        
        print("="*80)


def main():
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Run FP8 MM benchmarks.')
    parser.add_argument('--n-iter', type=int, default=100,
                        help='Number of iterations for the benchmark loop (default: 50). Ignored if --check-accuracy is set.')
    parser.add_argument('--enable-profiler', action='store_true',
                        help='Enable the PyTorch profiler. Ignored if --check-accuracy is set.')
    parser.add_argument('--enable-profiler-trace', action='store_true',
                        help='Enable the PyTorch profiler trace. Ignored if --check-accuracy is set.')
    parser.add_argument('--check-accuracy', action='store_true',
                        help='Run each kernel once for accuracy check against baseline instead of benchmarking.')
    parser.add_argument('--lib', type=str, default=None,
                        help='Kernel library to run. If not specified, all libraries will be run.')
    args = parser.parse_args()

    run_shapes(args.n_iter, args.enable_profiler, args.enable_profiler_trace, args.check_accuracy, args.lib)

if __name__ == "__main__":
    main()
