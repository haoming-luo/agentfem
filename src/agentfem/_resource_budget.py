# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Read-only resource facts and coefficient payloads, not solve-cost promises."""
from pathlib import Path
import shutil
import socket
from numbers import Integral

from . import _execution_environment as environment
from ._mpi_collectives import object_allgather


def memory_limits():
    """Inspect effective Linux cgroup ancestors and the process address limit."""
    limits = []
    memberships = environment._read('/proc/self/cgroup').splitlines()
    mounts = environment._read('/proc/self/mountinfo').splitlines()
    for line in memberships:
        fields = line.split(':', 2)
        if len(fields) != 3:
            continue
        _, controllers, member = fields
        version = 2 if controllers == '' else 1
        if version == 1 and 'memory' not in controllers.split(','):
            continue
        for mount_line in mounts:
            before, sep, after = mount_line.partition(' - ')
            left, right = before.split(), after.split()
            if not sep or len(left) < 5 or len(right) < 3:
                continue
            if right[0] != ('cgroup2' if version == 2 else 'cgroup'):
                continue
            if version == 1 and 'memory' not in right[2].split(','):
                continue
            root = Path(environment._unescape_mount(left[3]))
            mount = Path(environment._unescape_mount(left[4]))
            path = Path(member)
            # A cgroup namespace may report / relative to an already scoped mount.
            relative = path.relative_to(root) if path.is_relative_to(root) else Path('.') if member == '/' else None
            if relative is None or '..' in relative.parts:
                continue
            current = mount / relative
            while current.is_relative_to(mount):
                filename, usage = ('memory.max', 'memory.current') if version == 2 else ('memory.limit_in_bytes', 'memory.usage_in_bytes')
                raw = environment._read(current / filename).strip()
                used = environment._read(current / usage).strip()
                if raw.isdigit() and 0 <= int(raw) < 2**60:
                    limit = int(raw)
                    limits.append({'source': f'cgroup_v{version}', 'path': str(current),
                        'limit_bytes': limit, 'usage_bytes': int(used) if used.isdigit() else None,
                        'remaining_bytes': max(0, limit-int(used)) if used.isdigit() else None})
                if current == mount:
                    break
                current = current.parent
    address = None
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_AS)
        if 0 <= soft < 2**60:
            address = int(soft)
    except (ImportError, OSError, ValueError, AttributeError):
        pass
    return {'cgroups': limits, 'address_space_limit_bytes': address,
        'scope': 'readable limits only; absence means unknown or no readable finite limit, not unlimited capacity'}


def budget(model, *, target=None, step_options=None, fields=(), frames=None, output_directory=None, grid=None):
    from ._model_support import domain
    from ._model_review import discretization
    from .ir.export import describe
    from .provenance import collective_call
    mesh = domain(model.mesh)
    if mesh is None:
        raise ValueError('resource_budget requires a registered mesh.')
    options = dict(step_options or {})
    def prepare():
        Path(output_directory or Path.cwd()).expanduser().resolve()
        if frames is not None and (isinstance(frames, bool) or not isinstance(frames, Integral) or frames < 1):
            raise ValueError('output_frames must be a positive integer or None.')
        requested = []
        for item in fields:
            if isinstance(item, str):
                requested.append({'name': item, 'scalar_coefficients': None, 'bytes_per_frame': None})
                continue
            f = getattr(item, 'value', item)
            space = getattr(f, 'function_space', None)
            if space is None or space.mesh is not mesh:
                raise ValueError('Budget fields must be named unresolved fields or functions on the model mesh.')
            n = int(space.dofmap.index_map.size_global * space.dofmap.index_map_bs)
            requested.append({'name': f.name, 'scalar_coefficients': n,
                'bytes_per_frame': n * f.x.array.dtype.itemsize})
        if len({f['name'] for f in requested}) != len(requested):
            raise ValueError('Budget output field names must be distinct.')
        selected = {'name': getattr(target, 'name', None), 'status': 'not_checked'}
        if target is not None:
            space = getattr(target, 'function_space', None)
            if space is not None:
                if space.mesh is not mesh:
                    raise ValueError('Budget target must belong to the model mesh.')
                selected.update(status='declared', global_scalar_dofs=int(
                    space.dofmap.index_map.size_global * space.dofmap.index_map_bs))
        return requested, selected
    # Propagate rank-local bad field declarations before further collectives.
    from ._mpi_collectives import _defer_automatic_gc
    with _defer_automatic_gc():
        requested, selected_target = collective_call(prepare, comm=mesh.comm, label='resource budget inputs')
        from .provenance import collective_canonical_record
        collective_canonical_record({'fields': requested, 'frames': None if frames is None else int(frames),
            'step_options': describe(options), 'target': selected_target},
            comm=mesh.comm, label='resource budget output declaration')
    selected_grid = discretization(model) if grid is None else grid
    count, basis = None if frames is None else int(frames), 'explicit requested frames' if frames is not None else 'unknown accepted/output schedule'
    if count is None and getattr(model.study, 'analysis', None) in ('first_order_transient', 'second_order_dynamics', 'explicit_dynamics'):
        steps = options.get('steps')
        every = options.get('save_every')
        if all(isinstance(x, Integral) and not isinstance(x, bool) and x > 0 for x in (steps, every)):
            count = int(1 + steps//every + bool(steps % every))
            basis = 'planned fixed increments including initial/final; not proof of saved frames'
    destination = Path(output_directory or Path.cwd()).expanduser().resolve()
    def inspect_local():
        parent = destination
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        try:
            disk = {'free_bytes': shutil.disk_usage(parent).free, 'filesystem_probe': str(parent),
                'quota_checked': False}
        except OSError:
            disk = {'free_bytes': None, 'filesystem_probe': str(parent), 'quota_checked': False}
        runtime = environment.execution_environment(project=Path.cwd(), output=destination)
        limits = memory_limits()
        available = [runtime['memory_bytes']['MemAvailable']] if 'MemAvailable' in runtime['memory_bytes'] else []
        available.extend(item['remaining_bytes'] for item in limits['cgroups'] if item['remaining_bytes'] is not None)
        local = {'rank': mesh.comm.rank, 'host': socket.gethostname(), 'visible_cpus': runtime['visible_cpus'],
            'host_memory_bytes': runtime['memory_bytes'], 'limits': limits,
            'available_memory_ceiling_bytes': min(available) if available else None,
            'disk': disk, 'findings': runtime['findings']}
        return local
    with _defer_automatic_gc():
        local = collective_call(inspect_local, comm=mesh.comm, label="resource availability snapshot")
    ranks = object_allgather(mesh.comm, local)
    partial = any(f['bytes_per_frame'] is None for f in requested)
    payload = (sum(f['bytes_per_frame'] or 0 for f in requested)
               if any(f['bytes_per_frame'] is not None for f in requested) else None)
    return {'schema': 'agentfem.resource-budget.v1', 'discretization': selected_grid,
        'target': selected_target,
        'solver': {'requested': describe(options.get('solver_options')),
            'selection': 'declared' if options.get('solver_options') is not None else 'provider_default_not_built',
            'peak_memory_bytes': None, 'reason': 'no symbolic assembly/factorization; coefficient bytes do not predict LU fill'},
        'output': {'directory': str(destination), 'fields': requested, 'planned_frames': count,
            'frame_count_basis': basis, 'coefficient_bytes_per_frame': payload,
            'coefficient_payload_bytes': None if count is None or payload is None else payload*count,
            'payload_complete': bool(requested) and not partial,
            'file_size_bytes': None,
            'estimate_scope': 'uncompressed FE coefficient payload only; not a file-size bound; projection, padding, mesh, topology, compression and histories excluded'},
        'rank_resources': ranks,
        'resource_scope': 'rank snapshots may share host/cgroup/disk limits; never sum them as independent capacity',
        'solver_executed': False, 'settings_modified': False}
