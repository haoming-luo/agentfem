# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Data-only Q1 service; never imports the caller's case or initializes MPI.

Private protocol: uint32 little-endian count, n*15 binary64 input values
(four vertices and one query). Reply n*14 values: status plus 13 projection
values. No pickle, model execution or mutable FEM state.
"""

import struct
import sys
import numpy as np

MAGIC = b"AFMQ1v1\n"
MAX_BATCH = 65536


def read_exact(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        block = stream.read(size - len(chunks))
        if not block:
            raise EOFError("Truncated Q1 geometry worker response")
        chunks.extend(block)
    return chunks


def serve(source, target):
    from ._q1_surface_geometry import q1_closest_point

    if any(name in sys.modules for name in ("mpi4py.MPI", "petsc4py.PETSc", "dolfinx")):
        raise RuntimeError("Geometry service must not initialize the FEM/MPI runtime")
    target.write(MAGIC)
    target.flush()
    while True:
        header = source.read(4)
        if not header:
            return
        if len(header) != 4:
            header += read_exact(source, 4 - len(header))
        (count,) = struct.unpack("<I", header)
        if not count:
            return
        if count > MAX_BATCH:
            raise ValueError("Q1 geometry service batch exceeds transport bound")
        data = np.frombuffer(read_exact(source, count * 15 * 8), dtype="<f8").reshape(
            count, 15
        )
        if not np.isfinite(data).all():
            raise ValueError("Q1 geometry service requires finite inputs")
        output = np.zeros((count, 14), dtype="<f8")
        for index, row in enumerate(data):
            try:
                uv, basis, point, normal, gap = q1_closest_point(
                    row[:12].reshape(4, 3), row[12:]
                )
            except ValueError:
                continue
            output[index, 0] = 1
            output[index, 1:] = np.concatenate((uv, basis, point, normal, [gap]))
        target.write(output.tobytes())
        target.flush()


if __name__ == "__main__":
    serve(sys.stdin.buffer, sys.stdout.buffer)
