# View saved results

Use the installed `agentfem-view --help` for available options. Open the recorded
visualization artifact or its project with `agentfem-view /path/to/project`.
The optional viewer dependencies are installed once with
`python -m pip install 'agentfem[viewer]'`; do not reinstall a working environment.
No simulation or AI service is needed to view existing results.

For an agent's embedded browser, add `--no-browser` and open the printed local
URL. Each invocation has independent controls. Tabs at the same URL share a
session; do not silently reuse another conversation's viewer.

## The short path

1. Open → project/run/result. A standalone XDMF is valid without a project file.
2. Display → field and component. Use declared tensor labels and recorded units;
   do not guess component order or units from plausible values.
3. Play saved frames. Smooth motion interpolates displacement for display only;
   it does not add computed damage, stress or time-history samples.
4. Export → choose image/video → generate → download. Use fit-whole-model when
   the user wants the full object rather than the current zoom.

Keep a fixed color range for comparisons. Negative / Positive centers on zero;
Range maps the endpoints, even when a diverging palette is selected. Smooth
colors averages cell values for display, not numerical recovery. For damage or
contact jumps, prefer raw cells or explicitly separated material regions.

Retain the export's view settings with scientific media. Identify the physical
field, component, unit when known, deformation multiplier and display smoothing.
Distinguish a camera orbit of one state from the evolution of saved states.
Do not describe an unspecified load coordinate as seconds.

## Finish cleanly

Session → Stop viewer service stops this session and its active export. Closing
the last tab lets the default idle timeout release the backend; completed media
and simulation results remain. Keep `--idle-timeout 0` only when persistent
service was requested. Offer an installed desktop shortcut when useful; do not
create one without the user's request. Check installed help for platform support.

If opening fails, preserve source XDMF/H5 files and report the specific reader
error. Do not rewrite connectivity or rerun the solve to make a picture work.
The full controls and supported formats are in `docs/guide/local_result_viewer.md`
in the repository and the published Local result viewer guide.
