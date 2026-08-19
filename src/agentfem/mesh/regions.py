"""Named mesh region collections."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionSet:
    """Named collection of regions sharing one mesh tag object."""

    domain: object
    regions: dict[str, object]
    tags: object | None = None
    kind: str = "regions"

    def __getattr__(self, name: str):
        try:
            return self.regions[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, name: str):
        return self.regions[name]

    def __iter__(self):
        return iter(self.regions.values())

    def __len__(self) -> int:
        return len(self.regions)

    @property
    def names(self) -> tuple[str, ...]:
        """Return region names in insertion order."""

        return tuple(self.regions.keys())

    def field(self, name: str = "RegionId"):
        """Create a DG0 field visualizing this region set's cell tags."""

        if self.tags is None:
            raise ValueError("RegionSet.field requires cell tags.")
        if self.kind != "cell":
            raise ValueError("RegionSet.field currently supports cell partitions.")
        from . import tag_field

        return tag_field(self.domain, self.tags, name=name)

    def summary(self) -> dict[str, object]:
        """Return a compact region-set summary."""

        return {
            "kind": f"{self.kind}_region_set",
            "names": self.names,
            "regions": tuple(
                region.summary() if hasattr(region, "summary") else repr(region)
                for region in self.regions.values()
            ),
        }
