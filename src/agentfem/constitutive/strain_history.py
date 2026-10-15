# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Declared strain-history VUMAT lowering; not general Abaqus emulation.

Core owns transport, objective kinematics and native batching. The reviewed
source owns its equations. Global acceptance and restart stay with the Step.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import tempfile
import tomllib

import numpy as np

from ._explicit_material import (
    ExplicitMaterialBatchOutput, ExplicitMaterialCutbackRequest,
    ExplicitMaterialIncrementLimit,
)
from .explicit_user_material import explicit_user_material
from .material_array_batch import MaterialPointArrayBatchInput
from ._vumat_build import _WRAPPER, _check_signature
from .compiled_corotational import CompiledCorotationalIncrement
from .corotational import push_corotational_stress
from .native_build import build_native_libraries
from .user_material import MaterialStateSchema, MaterialStateVariable, inspect_abaqus_user_material
from .vumat import VumatBatchInput, compile_vumat

_UNUSED = frozenset({"coordinates", "characteristic_length", "relative_spin",
    "temperature", "stretch", "deformation_gradient", "fields", "absolute_time",
    "material_name", "old_energy_for_response"})
_COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


def _real(value, name, *, positive=False):
    if isinstance(value, (bool, complex)) or not isinstance(value, (int, float)):
        raise ValueError(f"AFM-HISTORY-SPEC: {name} must be real")
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"AFM-HISTORY-SPEC: invalid {name}")
    return float(value)


def _read_spec(manifest):
    path = Path(manifest).resolve()
    spec = tomllib.loads(path.read_text())
    required = {"schema", "source", "property_names", "state_count", "unused_channels",
                "kinematics", "energy", "maximum_stretch_increment", "increment_ceiling",
                "thread_safe"}
    if set(spec) - required - {"accelerated_source", "state_bounds", "parameter_bounds"} or required - set(spec):
        raise ValueError("AFM-HISTORY-SPEC: unknown or missing declaration")
    if spec["schema"] != "agentfem.strain-history.v1":
        raise ValueError("AFM-HISTORY-SPEC: unsupported schema")
    if spec["kinematics"] != "linear_right_stretch_path" or spec["energy"] != "incremental_internal_work":
        raise ValueError("AFM-HISTORY-SPEC: unsupported kinematics or energy")
    unused = spec["unused_channels"]
    if not isinstance(unused, list) or any(not isinstance(x, str) for x in unused) or set(unused) != _UNUSED or len(unused) != len(_UNUSED):
        raise ValueError("AFM-HISTORY-CHANNELS: explicitly review every unused channel; otherwise use compile_vumat")
    ns, names = spec["state_count"], spec["property_names"]
    if type(ns) is not int or not 0 <= ns <= 100000:
        raise ValueError("AFM-HISTORY-SPEC: state_count must be a nonnegative bounded integer")
    if not isinstance(names, list) or not names or any(not isinstance(x, str) or not x.strip() for x in names) or len(set(names)) != len(names):
        raise ValueError("AFM-HISTORY-SPEC: unique nonempty property names required")
    if type(spec["thread_safe"]) is not bool:
        raise ValueError("AFM-HISTORY-SPEC: declare thread_safe explicitly")
    parameter_bounds = spec.get("parameter_bounds", [])
    if not isinstance(parameter_bounds, list):
        raise ValueError("AFM-HISTORY-SPEC: parameter_bounds must be a list")
    for bound in parameter_bounds:
        if not isinstance(bound, dict) or set(bound) - {"name", "minimum", "maximum", "exclusive_minimum", "exclusive_maximum"} or bound.get("name") not in names:
            raise ValueError("AFM-HISTORY-SPEC: invalid parameter bound")
        if not {"minimum", "maximum"} & set(bound):
            raise ValueError("AFM-HISTORY-SPEC: parameter bound needs an endpoint")
        for key in ("minimum", "maximum"):
            if key in bound:
                _real(bound[key], key)
            flag = "exclusive_" + key
            if flag in bound and (key not in bound or type(bound[flag]) is not bool):
                raise ValueError("AFM-HISTORY-SPEC: invalid endpoint closure")
        if "minimum" in bound and "maximum" in bound and bound["minimum"] > bound["maximum"]:
            raise ValueError("AFM-HISTORY-SPEC: inverted parameter interval")
    limit = _real(spec["maximum_stretch_increment"], "maximum_stretch_increment", positive=True)
    if limit >= 1:
        raise ValueError("AFM-HISTORY-SPEC: stretch increment must be below one")
    _real(spec["increment_ceiling"], "increment_ceiling", positive=True)
    bounds = spec.get("state_bounds", [])
    if not isinstance(bounds, list):
        raise ValueError("AFM-HISTORY-SPEC: state_bounds must be a list")
    for bound in bounds:
        if not isinstance(bound, dict) or set(bound) != {"index", "minimum", "maximum", "tolerance"}:
            raise ValueError("AFM-HISTORY-SPEC: state bound requires index, minimum, maximum, tolerance")
        if type(bound["index"]) is not int or not 0 <= bound["index"] < ns:
            raise ValueError("AFM-HISTORY-SPEC: state bound index outside history")
        lo, hi, tol = (_real(bound[k], k) for k in ("minimum", "maximum", "tolerance"))
        if lo > hi or tol < 0 or not np.isfinite(lo-tol) or not np.isfinite(hi+tol):
            raise ValueError("AFM-HISTORY-SPEC: invalid state interval")
    for key in ("source", "accelerated_source"):
        if key not in spec:
            continue
        name = spec[key]
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("AFM-HISTORY-SOURCE: require a relative source")
        if not (path.parent / name).resolve().is_relative_to(path.parent):
            raise ValueError("AFM-HISTORY-SOURCE: source escapes material directory")
    return path, spec


def _source_payload(source):
    _check_signature(source)
    report = inspect_abaqus_user_material(source, kind="VUMAT")
    if report.status != "adapter_candidate" or report.external_calls:
        raise ValueError("AFM-HISTORY-SOURCE: unresolved dependencies; " + report.format())
    if set(report.source_graph.runtime_includes) - {"VABA_PARAM.INC"}:
        raise ValueError("AFM-HISTORY-SOURCE: unsupported runtime include")
    payload = {}
    for item in report.source_graph.files:
        name = Path(item.logical_path)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("AFM-HISTORY-SOURCE: escaping include")
        data = item.path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item.source_sha256:
            raise ValueError("AFM-HISTORY-SOURCE: changed during inspection")
        payload[name.as_posix()] = data
    for edge in report.source_graph.edges:
        if edge.status == "runtime_provided":
            name = Path(edge.source).parent / edge.declaration
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("AFM-HISTORY-SOURCE: escaping runtime include")
            payload[name.as_posix()] = b"      implicit real*8(a-h,o-z)\n"
    return payload


def _build(path, spec, cache, compiler):
    source = path.parent / spec.get("accelerated_source", spec["source"])
    payload = {"law/" + k: v for k, v in _source_payload(source).items()}
    native = Path(__file__).parent / "native"
    template = (native / "strain_history.f90").read_text()
    for token, value in {"@NS@": str(spec["state_count"]), "@NP@": str(len(spec["property_names"])),
                         "@LIMIT@": format(spec["maximum_stretch_increment"], ".17e").replace("e", "d")}.items():
        template = template.replace(token, value)
    checks = []
    for bound in spec.get("state_bounds", []):
        entry = f"state({bound['index']+1},p)"
        lo = format(bound["minimum"] - bound["tolerance"], ".17e").replace("e", "d")
        hi = format(bound["maximum"] + bound["tolerance"], ".17e").replace("e", "d")
        checks.append(f"if (.not.ieee_is_finite({entry}).or.{entry}<{lo}.or.{entry}>{hi}) status(p)=0")
    template = template.replace("@BOUNDS@", "\n ".join(checks))
    payload["history.f90"] = template.encode()
    payload["bridge.f90"] = _WRAPPER.encode()
    payload["wide.f90"] = (native / "polar_wider.f90").read_bytes()
    payload["series.f90"] = (native / "polar_series.f90").read_bytes()
    # The existing content-addressed builder owns compilation, locking and
    # integrity. This temporary input graph contains no installed executable.
    with tempfile.TemporaryDirectory(prefix="agentfem-history-inputs-") as tmp:
        root = Path(tmp)
        for name, data in payload.items():
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        text = 'schema = "agentfem.native-libraries.v1"\n'
        text += 'files = ' + json.dumps(list(payload)) + '\n'
        text += 'include_directories = ["law"]\n'
        text += 'flags = ["-O3", "-fno-fast-math", "-ffp-contract=off", "-fPIC", "-frecursive", "-fdefault-real-8", "-fdefault-double-8", "-ffree-line-length-none", "-ffixed-line-length-none", "-shared"]\n'
        text += '[targets]\nfused = ' + json.dumps(["wide.f90", "history.f90", "bridge.f90", "law/" + source.name]) + '\n'
        text += 'kinematics = ["series.f90"]\n'
        (root / "build.toml").write_text(text)
        return build_native_libraries(root / "build.toml", cache_directory=cache,
                                      trusted=True, compiler=compiler)


class StrainHistoryRuntime:
    """Experimental 3D strain-history source and optional fused Hex execution.

    Zero-initialized history plus six prior corotational stresses are stored in
    the ordinary schema. No hidden provider state, solver loop or checkpoint.
    Source review must establish independence of the declared unused channels,
    complete history in STATEV, and additive internal work per current mass.
    The chosen polar path is not a claim of arbitrary VUMAT equivalence.
    """
    def __init__(self, manifest, *, parameters, density, cache_directory,
                 trusted=False, workers=1, cell_workers=1, compiler="gfortran",
                 tile_size=None):
        if trusted is not True:
            raise ValueError("AFM-HISTORY-TRUST: source and channel review required")
        path, spec = _read_spec(manifest)
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError("AFM-HISTORY-WORKERS: require integer in [1,16]")
        if tile_size is not None and (type(tile_size) is not int or not 1 <= tile_size <= 2147483647):
            raise ValueError("AFM-HISTORY-TILE: require positive 32-bit integer or None")
        if not spec["thread_safe"]:
            raise ValueError("AFM-HISTORY-THREADS: fused source reentrancy must be reviewed; otherwise use compile_vumat")
        if not isinstance(parameters, dict) or set(parameters) != set(spec["property_names"]):
            raise ValueError("AFM-HISTORY-PARAMETERS: names must match the declaration")
        self._properties = np.array([_real(parameters[k], k) for k in spec["property_names"]])
        for bound in spec.get("parameter_bounds", []):
            value = parameters[bound["name"]]
            below = "minimum" in bound and (value < bound["minimum"] or (bound.get("exclusive_minimum", False) and value == bound["minimum"]))
            above = "maximum" in bound and (value > bound["maximum"] or (bound.get("exclusive_maximum", False) and value == bound["maximum"]))
            if below or above:
                raise ValueError(f"AFM-HISTORY-PARAMETERS: {bound['name']} outside declared domain")
        self._density = _real(density, "density", positive=True)
        self._spec = spec
        self._ns = spec["state_count"]
        source_schema = MaterialStateSchema("vumat_history", (
            (MaterialStateVariable("history", shape=(self._ns,)),) if self._ns else ()))
        self.state_schema = MaterialStateSchema("strain_history_and_stress", (
            MaterialStateVariable("history", shape=(self._ns + 6,)),))
        self._reference = compile_vumat(path.parent / spec["source"], state_schema=source_schema,
            properties=self._properties, cache_directory=Path(cache_directory)/"reference",
            trusted=True, compiler=compiler)
        self._bundle = _build(path, spec, Path(cache_directory)/"native", compiler)
        self._kinematics = CompiledCorotationalIncrement(self._bundle, "kinematics")
        self._library = ctypes.CDLL(str(self._bundle.library("fused")))
        real = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        integer = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        function = self._library.fused_material
        function.argtypes = [ctypes.c_int] + [real]*4 + [ctypes.c_double]*2 + [real]*3 + [integer]
        function.restype = None
        self._screen = self._library.history_increment_screen
        self._screen.argtypes = [ctypes.c_int, real, real, integer]
        self._screen.restype = None
        self._state_screen = self._library.history_state_screen
        self._state_screen.argtypes = [ctypes.c_int, real, integer]
        self._state_screen.restype = None
        self._identity = {"kind": "strain_history_runtime", "revision": 1, "spec": spec,
                          "reference": self._reference.summary(), "native": self._bundle.summary(),
                          "density": self._density}
        self._policy_id = hashlib.sha256(json.dumps(self._identity, sort_keys=True).encode()).hexdigest()
        self.material = explicit_user_material(state_schema=self.state_schema, update=self.update,
            density=self._density, identity=self._identity, increment_limit=self.increment_limit,
            increment_policy_identity=self._policy_id)
        from ..elements import native_explicit_hex_kernel
        self.kernel = native_explicit_hex_kernel(self.material, function,
            properties=self._properties, density=self._density, identity=self._identity,
            fallback=self.update, validate=lambda response: self._check_state(response.state_new),
            workers=workers, cell_workers=cell_workers, tile_size=tile_size)

    def summary(self):
        return json.loads(json.dumps(self._identity))

    def _check_state(self, state):
        if not isinstance(state, np.ndarray) or state.dtype != np.float64 or np.ndim(state) != 2 or np.shape(state)[1:] != (self._ns + 6,) or len(state) > np.iinfo(np.int32).max:
            raise ValueError("AFM-HISTORY-STATE: invalid state width")
        if self._spec.get("state_bounds"):
            status = np.empty(len(state), dtype=np.int32)
            self._state_screen(len(state), np.ascontiguousarray(state, dtype=np.float64), status)
            if np.any(status != 1):
                point = int(np.flatnonzero(status != 1)[0])
                raise ValueError(f"AFM-HISTORY-STATE: history outside declared domain at batch point {point}")

    def increment_limit(self, request):
        if not isinstance(request, MaterialPointArrayBatchInput) or request.state_schema != self.state_schema:
            raise ValueError("AFM-HISTORY-STATE: expected matching material request")
        self._check_state(request.state_old)
        if any(x.dtype != np.float64 for x in (request.deformation_gradient_old, request.deformation_gradient_new)):
            raise ValueError("AFM-HISTORY-STATE: mutated gradient dtype")
        old, new = (np.ascontiguousarray(x, dtype=np.float64) for x in
                    (request.deformation_gradient_old, request.deformation_gradient_new))
        if old.shape != (request.point_count, 3, 3) or new.shape != old.shape or request.point_count > np.iinfo(np.int32).max:
            raise ValueError("AFM-HISTORY-STATE: invalid native batch shape")
        status = np.empty(request.point_count, dtype=np.int32)
        self._screen(request.point_count, old, new, status)
        pending = np.flatnonzero(status != 1)
        if len(pending):
            relative = np.linalg.solve(old[pending].swapaxes(1, 2), new[pending].swapaxes(1, 2)).swapaxes(1, 2)
            changes = np.max(abs(np.linalg.svd(relative, compute_uv=False) - 1), axis=1)
            bad = changes > self._spec["maximum_stretch_increment"]
            if np.any(bad):
                point = int(pending[np.flatnonzero(bad)[0]])
                raise ExplicitMaterialCutbackRequest(f"AFM-HISTORY-INCREMENT: stretch limit exceeded at batch point {point}")
        return ExplicitMaterialIncrementLimit(np.full(request.point_count, self._spec["increment_ceiling"]), self._policy_id)

    def update(self, request):
        if not isinstance(request, MaterialPointArrayBatchInput) or request.state_schema != self.state_schema:
            raise ValueError("AFM-HISTORY-STATE: schema mismatch")
        if any(getattr(request, name) is not None for name in ("temperature", "temperature_increment", "field_variables")):
            raise ValueError("AFM-HISTORY-CHANNELS: this runtime does not carry thermal/field inputs")
        self._check_state(request.state_old)
        kin = self._kinematics(request.deformation_gradient_old, request.deformation_gradient_new)
        strain = np.stack([kin.strain_increment[:, i, j] for i, j in _COMPONENTS], axis=1)
        n = len(strain)
        jmid = (np.linalg.det(request.deformation_gradient_old) + np.linalg.det(request.deformation_gradient_new))/2
        stretch = np.tile([1., 1., 1., 0., 0., 0.], (n, 1))
        gradient = np.pad(stretch, ((0, 0), (0, 3)))
        r = VumatBatchInput(step_time=request.time_increment, total_time=request.time_increment,
            time_increment=request.time_increment, density=self._density/jmid,
            coordinates=np.zeros((n, 3)), characteristic_length=np.ones(n),
            strain_increment=strain, relative_spin_increment=np.zeros((n, 3)),
            temperature_old=np.zeros(n), temperature_new=np.zeros(n), stretch_old=stretch, stretch_new=stretch,
            deformation_gradient_old=gradient, deformation_gradient_new=gradient,
            field_old=np.zeros((n, 1)), field_new=np.zeros((n, 1)),
            stress_old=request.state_old[:, self._ns:], state_old=request.state_old[:, :self._ns],
            internal_energy_old=np.zeros(n), inelastic_energy_old=np.zeros(n))
        out = self._reference.update(r)
        if out.internal_energy_new is None:
            raise ValueError("AFM-HISTORY-ENERGY: source must supply incremental internal work")
        stress = np.zeros((n, 3, 3))
        for k, (i, j) in enumerate(_COMPONENTS):
            stress[:, i, j] = stress[:, j, i] = out.stress_new[:, k]
        state = np.concatenate((out.state_new, out.stress_new), axis=1)
        self._check_state(state)
        return ExplicitMaterialBatchOutput(push_corotational_stress(stress, kin.frame_new), state,
            self.state_schema, out.internal_energy_new * self._density)

    def close(self):
        self.kernel.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
