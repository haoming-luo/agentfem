"""Private deterministic registry for Step providers.

This module owns selection only.  Scientific acceptance predicates and
lowering functions remain with their provider catalog, while execution-context
binding remains with the public Step dispatch boundary.
"""

from __future__ import annotations


class ProviderSelectionRegistry:
    """Ordered, inspectable collection of Step-lowering providers."""

    def __init__(self):
        self._providers: dict[str, object] = {}

    def register(self, provider, *, replace: bool = False):
        name = getattr(provider, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError("Step providers must declare a non-empty name.")
        if name in self._providers and not replace:
            raise ValueError(f"Step provider {name!r} is already registered.")
        self._providers[name] = provider
        return provider

    def providers(self) -> tuple[object, ...]:
        return tuple(
            sorted(
                self._providers.values(),
                key=lambda item: (-item.priority, item.name),
            )
        )

    def candidates(self, analysis: str) -> tuple[object, ...]:
        """Return providers declaring one normalized analysis."""

        return tuple(
            provider for provider in self.providers() if analysis in provider.analyses
        )

    def resolve(self, model, request):
        """Select one provider or raise a repairable capability error."""

        candidates = self.candidates(request.analysis)
        option_rejections = []
        for provider in candidates:
            if provider.accepts(model, request):
                issues = provider.option_issues(request)
                if not issues:
                    return provider
                option_rejections.append(provider)
        if option_rejections:
            # A lower-priority provider may own another valid option vocabulary
            # for the same analysis. Fail only after all compatible candidates
            # have rejected the request.
            option_rejections[0].validate_options(request)
        registered_materials = [
            type(record.item).__name__ for record in getattr(model, "materials", ())
        ]
        material = getattr(request, "material", None)
        raise NotImplementedError(
            "No step provider accepted "
            f"analysis={request.analysis!r}, material="
            f"{type(material).__name__ if material is not None else None!r}, "
            f"registered_materials={registered_materials!r}. "
            f"Candidate providers={[item.name for item in candidates]!r}."
        )


__all__ = ()
