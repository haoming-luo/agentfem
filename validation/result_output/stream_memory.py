"""Measure Python allocation peak; not total process RSS or solver memory."""

import json, sys, tracemalloc
from pathlib import Path
from dolfinx import fem
from agentfem import mesh, fields, results

frames = int(sys.argv[1])
destination = Path(sys.argv[2])
domain = mesh.rectangle((0.0, 0.0), (1.0, 1.0), (30, 30))
u = fem.Function(fields.displacement(domain, degree=2).space, name="U")
r = results.SimulationResult(name="memory")
tracemalloc.start()
with results.output_series(
    r, destination, primary_semantic_name="Displacement"
) as writer:
    for i in range(frames):
        u.x.array[:] = i * 0.001
        writer.write_accepted(coordinate=i, primary=u)
current, peak = tracemalloc.get_traced_memory()
print(
    json.dumps(
        dict(
            frames=frames,
            python_peak_bytes=peak,
            one_primary_array_bytes=u.x.array.nbytes,
            retained_all_primary_arrays_bytes=frames * u.x.array.nbytes,
        )
    )
)
