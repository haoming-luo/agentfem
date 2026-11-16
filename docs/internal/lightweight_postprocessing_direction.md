# Lightweight local postprocessing direction

Research decision, 2026-11-16. This does not declare new viewer functionality.
Implementation remains on the independent `feat/mesh-result-inspection` branch.

## Reuse rather than build another CAE framework

- [Kitware Glance](https://kitware.github.io/glance/) is a useful reference for
  a clean standalone scientific web viewer and compact controls.
- [ParaView Visualizer](https://kitware.github.io/visualizer/docs/index.html)
  illustrates richer ParaView-backed browser operation, but shipping its full
  server stack is not the default lightweight route.
- [Trame](https://kitware.github.io/trame/) with VTK/PyVista is the preferred
  continuation of our existing optional viewer: reuse scientific readers and
  filters, keep solver dependencies and viewing dependencies separate.

These are established technical references, not evidence that one is the
most popular or that every screenshot demonstrates scientific correctness.

## User workflow and staged acceptance

The opening screen asks for a result, not a new project or a chat prompt.
Use one viewport, a compact field/time toolbar, and an optional detail panel.
Provide a clean neutral theme first. Themes never change data, legend limits,
warping, association, or scientific meaning.

| Stage | Interaction | Acceptance |
|---|---|---|
| Existing bounded prototype | Saved frame, field/component, reference warp, edges, reset, refresh | Preserve topology and node/cell association; only complete saved frames |
| First useful release | Authorized local result selection, play/pause, part visibility, legend and fixed range, screenshot | Open XDMF with sibling H5; resume and refresh retain valid selection; no duplicate frames; no double warp |
| Scientific inspection | Clip/slice, node/cell probe, same-point history, CSV | Compare selected values against source arrays; make interpolation/averaging explicit; retain IDs |
| Larger models | Bounded frame cache, surface delivery, cancellable load | Measure memory and interaction latency on representative meshes; no silent truncation of scientific exports |

This is bounded basic viewing parity, not full Abaqus/ParaView feature parity.
Do not conflate a rendered surface with changing volume elements into shells.
High-order rendering needs explicit support and source-topology checks.

## Local files and privacy

A browser file input does not automatically grant access to an XDMF file's H5
siblings. Support an explicitly authorized result directory through the local
service, or a verified host file-selection capability. Never expose a home
directory or arbitrary filesystem endpoint. Bind locally; do not create a
public tunnel. Verify relative references remain within the granted scope.

Mesh arrays stay local and out of model context. Rotation, sliders, playback,
and coloring run without LLM calls or solver execution. Send only a small view
description and requested probe values to an agent. Refresh must tolerate an
atomic index replacement while the solver publishes accepted frames.

## Codex integration without host lock-in

Use the working local browser route first. Keep result inspection tools useful
without UI. The official [OpenAI MCP UI guidance](https://developers.openai.com/plugins/build/chatgpt-ui)
recommends standard MCP Apps resources (`_meta.ui.resourceUri`) and the `ui/*`
bridge, with feature detection for host extensions. Its documented ChatGPT
iframe support is not proof that every Codex host renders the same interface.

When the host advertises compatible MCP Apps support, reuse the viewer shell;
otherwise open the local browser page. Do not depend unconditionally on
`window.openai`, or add a second solver/result protocol. Verify opening, local
file access, camera interaction, field/time selection, and selected-value
exchange in each supported host before claiming embedded compatibility.

The existing optional prototype is a base, not completion of these acceptance
items. Windows behavior, large-model memory, installed launch UX, clip/probe,
and host-specific embedding still require implementation and testing.
