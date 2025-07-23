# Copyright (c) 2025 Julien Fischer, Chair of Computer Graphics and Visualization, TUD Dresden University of Technology.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

import os
import re
import requests
from os import path as osp

UTILS_URL = "https://raw.githubusercontent.com/NeelayS/supr_convertor/a0988f69fdbb613858d09998cbeece579c332504/supr_convertor/utils.py"
LOSSES_URL = "https://raw.githubusercontent.com/NeelayS/supr_convertor/a0988f69fdbb613858d09998cbeece579c332504/supr_convertor/losses.py"
LICENSE_SUPR_CONVERTOR_URL = "https://raw.githubusercontent.com/NeelayS/supr_convertor/refs/heads/main/LICENSE"
LICENSE_SMPLX_URL = "https://raw.githubusercontent.com/vchoutas/smplx/refs/heads/main/LICENSE"
POSE_UTILS_URL = "https://raw.githubusercontent.com/vchoutas/smplx/cd860395e277580037363ba4f68e7d90054aca39/transfer_model/utils/pose_utils.py"

def download_and_save_file(url, filepath):
    print(f"Downloading file from: \n{url}")
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download file: HTTP {response.status_code}")
    if osp.exists(filepath):
        os.remove(filepath)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(response.text)


def patch_utils_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    del lines[3:26]
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Applied patch to {filepath}")


def patch_edge_loss_file(filepath, patched_file_path):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines[80] = lines[80][:4] + lines[80][5:]
    patched = lines[78:93]
    with open(patched_file_path, "a", encoding="utf-8") as f:
        f.writelines(patched)
    os.remove(filepath)
    print(f"Applied patch from {filepath} to {patched_file_path}")


def patch_pose_utils_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # replace Tensor with torch.Tensor
    pattern = r'\bTensor\b(?=[\s,:]|$)'
    patched_lines = [re.sub(pattern, 'torch.Tensor', s) for s in lines]
    patched = patched_lines[:29] + patched_lines[59:]
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(patched)
    print(f"Applied patch to {filepath}")


if __name__ == '__main__':
    script_dir = osp.dirname(osp.abspath(__file__))
    copy_dir = osp.join(
        osp.dirname(script_dir),
        'smpl_conversion',
        'utils',
        'mpi_code'
    )
    message_supr_convertor = f"""
    This script will now download and patch code from the SUPR Convertor by the
    Max Planck Institute for Intelligent Systems (MPI).
    This code is governed by a non-commercial research license:

    {LICENSE_SUPR_CONVERTOR_URL}
    This license can also be found at smpl_conversion/utils/mpi_code/LICENSE.supr_convertor

    By continuing, you acknowledge that:
    - You have read and agree to the MPI license terms.
    - You will use the code only for non-commercial scientific research.
    - You will not redistribute the code or any modified version.

    Press Enter to continue, or Ctrl+C to cancel.
    """
    message_smplx = f"""
    This script will now download and patch code from SMPLX by the
    Max Planck Institute for Intelligent Systems (MPI).
    This code is governed by a non-commercial research license:

    {LICENSE_SMPLX_URL}
    This license can also be found at smpl_conversion/utils/mpi_code/LICENSE.smplx

    By continuing, you acknowledge that:
    - You have read and agree to the MPI license terms.
    - You will use the code only for non-commercial scientific research.
    - You will not redistribute the code or any modified version.

    Press Enter to continue, or Ctrl+C to cancel.
    """
    input(message_supr_convertor)
    download_and_save_file(UTILS_URL, osp.join(copy_dir, 'mpi_edge_loss.py'))
    download_and_save_file(LOSSES_URL, osp.join(copy_dir, 'losses.py'))
    patch_utils_file(osp.join(copy_dir, 'mpi_edge_loss.py'))
    patch_edge_loss_file(osp.join(copy_dir, "losses.py"), osp.join(copy_dir, 'mpi_edge_loss.py'))
    input(message_smplx)
    download_and_save_file(POSE_UTILS_URL, osp.join(copy_dir, 'mpi_pose_utils.py'))
    patch_pose_utils_file(osp.join(copy_dir, 'mpi_pose_utils.py'))
