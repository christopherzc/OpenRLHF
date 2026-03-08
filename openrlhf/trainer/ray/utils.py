import os


# Address https://github.com/ray-project/ray/issues/51117
# This function is used to get the bundle indices of a placement group
# and ensure that the bundles placed on the same node are grouped together.
def get_bundle_indices(placement_group, index, length):
    import ray

    pg_infos = ray.util.placement_group_table(placement_group)

    node_id_to_bundles = {}
    for bundle, node_id in pg_infos["bundles_to_node_id"].items():
        node_id_to_bundles.setdefault(node_id, []).append(bundle)

    sorted_bundle_indices = sum(node_id_to_bundles.values(), [])
    return sorted_bundle_indices[index * length : (index + 1) * length]


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/3b9e729f6a669ffd85190f901f5e262af79771b0/python/ray/_private/accelerators/amd_gpu.py#L114-L115
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    import torch

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


def allow_subprocess_ptrace():
    """Allow child processes to access this process's file descriptors via pidfd_getfd.

    vLLM V1 runs EngineCore in a subprocess. CUDA IPC weight sync shares GPU memory
    handles across the process boundary using pidfd_getfd, which requires either the
    SYS_PTRACE capability or PR_SET_PTRACER permission on the target process.
    Most container runtimes don't grant SYS_PTRACE, so we call
    prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY) to allow any process to access our FDs.

    Must be called in the process that creates CUDA IPC handles (PolicyModelActor)
    AND in the process that hosts vLLM (LLMRayActor).
    """
    import ctypes
    import ctypes.util
    import logging
    import sys

    if sys.platform != "linux":
        return

    PR_SET_PTRACER = 0x59616D61
    PR_SET_PTRACER_ANY = ctypes.c_ulong(-1).value
    libc_name = ctypes.util.find_library("c") or "libc.so.6"
    libc = ctypes.CDLL(libc_name, use_errno=True)
    result = libc.prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0)
    if result != 0:
        logging.getLogger(__name__).warning(
            "prctl(PR_SET_PTRACER) failed (errno=%d). "
            "CUDA IPC weight sync may fail with pidfd_getfd errors.",
            ctypes.get_errno(),
        )
