"""Cross-snapshot comparison: what `p10b` measures, expanded into deliverables.

`p10b_drift` answers "how far apart are these snapshots" with a summary distance
per level. This package answers the questions a reader asks next — which classes
moved, whether a class missing from one snapshot is really absent, what is
characteristic of each snapshot, and what the rows actually say — and ships them
as tables, figures, a report and two workbooks.

Nothing here knows a corpus name. Everything is derived from the run's own
artifacts: the snapshots are the tags `p0` wrote, the classes are the delivered
tree, the risk layer is the run's own risk screen. The only things a user may
supply are display names, a grouping of snapshots, and which pairs to contrast
— all optional, all with working defaults.
"""

from .manifest import Contrast, SnapshotManifest, build_manifest
from .pipeline import PooledOptions, PooledResult, run_comparison
from .source import CANONICAL_SNAPSHOT_COL, DepsSource, DirSource, RunSource

__all__ = ["Contrast", "SnapshotManifest", "build_manifest", "PooledOptions",
           "PooledResult", "run_comparison", "CANONICAL_SNAPSHOT_COL",
           "DepsSource", "DirSource", "RunSource"]
