# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0

"""Build only requested discovery sections from their existing owners."""

from . import __version__, public_api
from ._api_contract import CAPABILITIES_SCHEMA_VERSION, CLI_COMMANDS, model_methods, model_method_contract
from .presentation import capability_sections


def build_capability_record(topic=None, *, templates):
    sections = set(capability_sections(topic))
    record = {
        "schema": "agentfem.capabilities",
        "schema_version": CAPABILITIES_SCHEMA_VERSION,
        "agentfem_version": __version__,
    }
    if "public_modules" in sections:
        record["public_modules"] = public_api()
    if "commands" in sections:
        from ._architecture_contract import ownership_contract

        record.update({
            "commands": CLI_COMMANDS,
            "templates": templates(),
            "public_api": {level: public_api(level) for level in ("core", "advanced", "expert")},
            "model_api": {level: model_methods(level) for level in ("core", "advanced", "compatibility")},
            "model_api_contract": model_method_contract(),
            "ownership_contract": ownership_contract(),
        })
    if "runtime" in sections:
        from . import platforms

        record["runtime"] = platforms.runtime_report().summary()
    if "constitutive" in sections:
        from . import benchmarks, constitutive

        record["constitutive"] = tuple(item.as_dict() for item in constitutive.capabilities())
        record["constitutive_evidence"] = tuple(item.as_dict() for item in benchmarks.audit_capability_evidence())
    if "mesh_cells" in sections:
        from . import mesh

        record["mesh_cells"] = tuple(item.summary() for item in mesh.compatibility_matrix())
    if "step_providers" in sections:
        from . import step_providers

        record["step_providers"] = tuple(item.summary() for item in step_providers.step_providers())
    if "extensions" in sections:
        from . import extensions

        record["extensions"] = extensions.extension_status()
    return record
