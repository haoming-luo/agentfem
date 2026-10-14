# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit, batched transport for reviewed legacy 3D VUMAT sources.

This is a material-point bridge, not an Abaqus runtime or a kinematics model.
Every physical input is supplied by the caller in VUMAT's corotational basis.
The bridge never commits state, fabricates a tangent or chooses a stable step.
"""
from dataclasses import dataclass, replace
import ctypes
import json

import numpy as np

from .material_array_batch import _array
from .user_material import MaterialStateSchema


_INPUTS = (
    "density", "coordinates", "characteristic_length", "strain_increment",
    "relative_spin_increment", "temperature_old", "stretch_old",
    "deformation_gradient_old", "field_old", "stress_old", "state_old",
    "internal_energy_old", "inelastic_energy_old", "temperature_new",
    "stretch_new", "deformation_gradient_new", "field_new",
)


@dataclass(frozen=True, kw_only=True)
class VumatBatchInput:
    """Complete legacy VUMAT input; no missing channel is filled implicitly.

    Symmetric components: 11,22,33,12,23,31 (tensor shear).
    Deformation gradient: 11,22,33,12,23,31,21,32,13.
    Times are the supplied beginning-of-increment times. Energies are per mass.
    Arrays describe one block and use arbitrary numbers of states and fields.
    """
    step_time: float
    total_time: float
    time_increment: float
    density: np.ndarray
    coordinates: np.ndarray
    characteristic_length: np.ndarray
    strain_increment: np.ndarray
    relative_spin_increment: np.ndarray
    temperature_old: np.ndarray
    stretch_old: np.ndarray
    deformation_gradient_old: np.ndarray
    field_old: np.ndarray
    stress_old: np.ndarray
    state_old: np.ndarray
    internal_energy_old: np.ndarray
    inelastic_energy_old: np.ndarray
    temperature_new: np.ndarray
    stretch_new: np.ndarray
    deformation_gradient_new: np.ndarray
    field_new: np.ndarray

    def __post_init__(self):
        for name in ("step_time", "total_time", "time_increment"):
            value = getattr(self, name)
            if isinstance(value, (bool, complex)) or not np.isscalar(value):
                raise ValueError(f"{name} must be a finite real scalar")
            value = float(value)
            if not np.isfinite(value) or value < 0 or (name == "time_increment" and value == 0):
                raise ValueError(f"Invalid {name}")
            object.__setattr__(self, name, value)
        if self.step_time > self.total_time:
            raise ValueError("step_time cannot exceed total_time")
        density = np.asarray(self.density)
        if density.ndim != 1 or density.size == 0:
            raise ValueError("A VUMAT batch requires at least one material point")
        n = density.size
        widths = {"coordinates": 3, "strain_increment": 6,
                  "relative_spin_increment": 3, "stretch_old": 6, "stretch_new": 6,
                  "deformation_gradient_old": 9, "deformation_gradient_new": 9,
                  "stress_old": 6}
        for name in _INPUTS:
            raw = getattr(self, name)
            if np.ma.isMaskedArray(raw):
                raise ValueError(f"Masked {name} is not supported")
            if name in ("state_old", "field_old", "field_new"):
                shape = np.shape(raw)
                if len(shape) != 2 or shape[0] != n:
                    raise ValueError(f"{name} requires (points, components)")
            else:
                shape = (n, widths[name]) if name in widths else (n,)
            object.__setattr__(self, name, _array(raw, shape, name))
        if self.field_old.shape != self.field_new.shape:
            raise ValueError("Old/new field variable layouts differ")
        if np.any(self.density <= 0) or np.any(self.characteristic_length <= 0):
            raise ValueError("Density and characteristic length must be positive")


@dataclass(frozen=True)
class VumatBatchOutput:
    """Uncommitted local response. Missing optional energy remains unavailable."""
    stress_new: np.ndarray
    state_new: np.ndarray
    internal_energy_new: np.ndarray | None
    inelastic_energy_new: np.ndarray | None

    def __post_init__(self):
        stress = np.asarray(self.stress_new)
        state = np.asarray(self.state_new)
        if stress.ndim != 2 or stress.shape[1] != 6:
            raise ValueError("VUMAT stress requires (points, 6)")
        n = len(stress)
        if state.ndim != 2 or state.shape[0] != n:
            raise ValueError("VUMAT state requires (points, states)")
        for name, shape in (("stress_new", (n, 6)), ("state_new", state.shape),
                            ("internal_energy_new", (n,)), ("inelastic_energy_new", (n,))):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _array(value, shape, name))


class CompiledVumat:
    """One loaded library; one native call per block, never per material point.

    Construct with ``compile_vumat``. Calls on this instance are serialized:
    legacy routines may have SAVE/COMMON state and are not presumed thread-safe.
    Such hidden state still makes a law unsuitable for rollback/restart; callers
    must review that all physical history lives in the supplied state arrays.
    """
    def __init__(self, path, *, state_schema, properties, material_name, identity):
        import threading
        if not isinstance(state_schema, MaterialStateSchema):
            raise TypeError("Expected MaterialStateSchema")
        props = np.asarray(properties)
        if props.ndim != 1 or props.size == 0 or np.ma.isMaskedArray(properties):
            raise ValueError("Properties require a nonempty real vector")
        self._properties = _array(props, props.shape, "properties")
        self.state_schema = state_schema
        if not isinstance(material_name, str) or not material_name.strip():
            raise ValueError("Material name must be nonempty ASCII")
        name = material_name.upper().encode("ascii")
        if len(name) > 80 or b"\0" in name or name.startswith(b"ABQ_"):
            raise ValueError("Invalid VUMAT material name")
        self._name = name.ljust(80, b" ")
        self._identity = json.dumps(identity, sort_keys=True, allow_nan=False)
        self._library = ctypes.CDLL(str(path))
        self._function = self._library.agentfem_vumat_batch
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
        self._function.argtypes = [ctypes.c_int]*4 + [ctypes.c_double]*3 + [pointer]*22 + [ctypes.c_char_p]
        self._function.restype = None
        self._lock = threading.Lock()

    @property
    def properties(self):
        return self._properties.view()

    def summary(self):
        return {"kind": "legacy_vumat_3d_batch", "experimental": True,
                "build": json.loads(self._identity), "properties": self.properties.tolist(),
                "state_schema": self.state_schema.summary(),
                "material_name": self._name.decode().rstrip(),
                "kinematics": "caller_supplied_corotational", "shear": "tensor",
                "energy": "specific_per_mass", "owns_state": False,
                "implicit_tangent": False, "automatic_stability": False}

    def update(self, request):
        if not isinstance(request, VumatBatchInput):
            raise TypeError("Expected VumatBatchInput")
        # Frozen dataclasses do not freeze ndarray shape/dtype metadata. Check
        # that metadata again before passing pointers to trusted native code.
        request = replace(request)
        n, ns = request.state_old.shape
        nf = request.field_old.shape[1]
        if ns != self.state_schema.size:
            raise ValueError("VUMAT state schema size mismatch")
        if max(n, ns, nf, len(self._properties)) > np.iinfo(np.int32).max:
            raise ValueError("VUMAT integer dimension overflow")
        # Copy even immutable inputs: trusted Fortran must not corrupt accepted
        # state through a mistakenly writable argument or a failed update.
        inputs = [np.array(value, dtype=np.float64, order="F", copy=True)
                  for value in (self._properties, *(getattr(request, k) for k in _INPUTS))]
        outputs = [np.full(shape, np.nan, order="F")
                   for shape in ((n, 6), (n, ns), (n,), (n,))]
        with self._lock:
            self._function(n, ns, len(self._properties), nf, request.step_time,
                           request.total_time, request.time_increment,
                           *inputs, *outputs, self._name)
        energy = [None if np.isnan(value).all() else value for value in outputs[2:]]
        return VumatBatchOutput(*outputs[:2], *energy)


def compile_vumat(source, *, state_schema, properties, cache_directory,
                  material_name="USER_MATERIAL", trusted=False, compiler="gfortran"):
    """Compile explicitly trusted legacy VUMAT source; no solve-time downloads.

    GNU Fortran, macOS/Linux/WSL2, binary64 and the legacy scalar-dt signature
    only. Compilation is not sandboxing or a guarantee of constitutive validity.
    """
    from ._vumat_build import build_vumat
    path, identity = build_vumat(source, cache_directory=cache_directory,
                                trusted=trusted, compiler=compiler)
    return CompiledVumat(path, state_schema=state_schema, properties=properties,
                         material_name=material_name, identity=identity)


__all__ = ["VumatBatchInput", "VumatBatchOutput", "CompiledVumat", "compile_vumat"]
