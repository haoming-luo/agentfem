# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Serial immutable frame storage behind one atomically published XDMF view."""

import os

import h5py

from .output import UnifiedXDMFTimeSeries


class LiveXDMFTimeSeries(UnifiedXDMFTimeSeries):
    """Reuse field/topology validation; never modify published heavy data.

    Readers may open the index during calculation. A refresh discovers later
    frames; an already open reader retains its consistent earlier view. This
    is not a checkpoint or a promise against hardware/storage failure.
    """

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.data_directory = self.path.with_name(f"{self.path.stem}-data")
        self._mesh_ready = False

    def __enter__(self):
        self._require_open()
        if self.path.exists() or self.data_directory.exists():
            raise FileExistsError("Live output exists; use a fresh result path.")
        return super().__enter__()

    @staticmethod
    def _publish_closed_file(temporary, destination):
        # Both paths belong to this new series directory, on the same filesystem.
        # Closing HDF5 precedes fsync and rename; readers never see the .tmp file.
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(destination)

    def _prepare_mesh(self):
        if self.path.exists():
            raise FileExistsError("Live output exists; use a fresh result path.")
        self.data_directory.mkdir(parents=True, exist_ok=False)
        temporary = self.data_directory / ".mesh.h5.tmp"
        original = self.h5_path
        try:
            self.h5_path = temporary
            super()._open_storage()
        finally:
            if self._h5 is not None:
                self._h5.close()
                self._h5 = None
            self.h5_path = original
        self._publish_closed_file(temporary, self.data_directory / "mesh.h5")
        self._mesh_ready = True

    def _store_frame(self, frame_name, time, geometry, attributes, component_metadata):
        if not self._mesh_ready:
            self._prepare_mesh()
        filename = f"frame-{frame_name}.h5"
        temporary = self.data_directory / f".{filename}.tmp"
        with h5py.File(temporary, "x") as store:
            store.attrs["agentfem_schema"] = "agentfem.immutable-xdmf-frame"
            store.attrs["schema_version"] = "0.1.0"
            group = store.create_group(f"Frames/{frame_name}")
            group.attrs["coordinate"] = time
            group.attrs["load_factor"] = time
            shared_geometry = self._effective_scale == 0 and self.store_reference_geometry
            if not shared_geometry:
                group.create_dataset("Geometry", data=geometry, **self._h5_options)
            self._write_frame_attributes(group, attributes, component_metadata)
        self._publish_closed_file(temporary, self.data_directory / filename)
        prefix = self.data_directory.name
        # Point XDMF directly at the shared dataset. Some VTK XDMF readers do
        # not follow HDF5 external links, even though h5py/meshio do.
        geometry_reference = (f"{prefix}/mesh.h5:/Mesh/ReferenceGeometry" if shared_geometry
                              else f"{prefix}/{filename}:/Frames/{frame_name}/Geometry")
        return f"{prefix}/mesh.h5", f"{prefix}/{filename}", geometry_reference

    def _publish_frame_index(self):
        self._publish_index(self.path, partial=True)
