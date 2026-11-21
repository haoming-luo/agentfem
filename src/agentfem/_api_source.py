# SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact simple signatures from installed Python source, without importing FEM.

Only literal defaults and ordinary undecorated functions, explicit constructors
or simple dataclasses are admitted. Dynamic declarations keep live inspection.
Source text is parsed, never executed; this is not a second API registry.
"""
import ast
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _mapping(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except (TypeError, ValueError):
                return None
    return None


def _module(path):
    if path.with_suffix('.py').is_file():
        return path.with_suffix('.py')
    if (path/'__init__.py').is_file():
        return path/'__init__.py'
    return None


def _dynamic_binding(body, name):
    """Reject multiple/conditional writes in this scope; do not enter functions.

    A branch may replace a callable or its signature even when an ordinary
    definition appears earlier. Such bindings must retain runtime inspection.
    """
    bindings = 0

    def visit(node, direct=False):
        nonlocal bindings
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                bindings += 1
                return not direct
            return False
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            matches = sum((alias.asname or alias.name.split('.')[0]) == name for alias in node.names)
            bindings += matches
            return bool(matches and not direct)
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
            return True
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
            return True
        return any(visit(child) for child in ast.iter_child_nodes(node))

    return any(visit(node, direct=True) for node in body) or bindings > 1


def _binding(path, name, visited):
    if (path, name) in visited or not path.is_relative_to(ROOT):
        return None
    visited.add((path, name))
    tree = ast.parse(path.read_text(encoding='utf-8'))
    if _dynamic_binding(tree.body, name):
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return node, tree
    for key in ('_DOCUMENTED_REEXPORTS', '_LAZY_EXPORTS'):
        mapping = _mapping(tree, key) or {}
        if name in mapping:
            target = _module(path.parent.joinpath(*mapping[name].split('.')))
            return None if target is None else _binding(target, name, visited)
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level and node.module:
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    base = path.parent
                    for _ in range(node.level-1):
                        base = base.parent
                    target = _module(base.joinpath(*node.module.split('.')))
                    return None if target is None else _binding(target, alias.name, visited)
    if any(isinstance(node, ast.FunctionDef) and node.name == '__getattr__' for node in tree.body):
        return None
    if name in (_mapping(tree, '__all__') or ()):
        return None
    raise LookupError(name)


def _literal(node):
    # No eval(), attribute access, default factory, decorator or annotation call.
    return ast.literal_eval(node)


def _signature(node, *, drop_self=False):
    if node.decorator_list:
        return None
    args, parameters = node.args, []
    positional = [*args.posonlyargs, *args.args]
    defaults = [None]*(len(positional)-len(args.defaults)) + list(args.defaults)
    def parameter(arg, kind, default=None):
        return inspect.Parameter(arg.arg, kind,
            default=inspect.Parameter.empty if default is None else _literal(default),
            annotation=inspect.Parameter.empty if arg.annotation is None else ast.unparse(arg.annotation))
    try:
        for i, (arg, default) in enumerate(zip(positional, defaults, strict=True)):
            if drop_self and i == 0 and arg.arg in ('self', 'cls'):
                continue
            parameters.append(parameter(arg, inspect.Parameter.POSITIONAL_ONLY if i < len(args.posonlyargs)
                else inspect.Parameter.POSITIONAL_OR_KEYWORD, default))
        if args.vararg:
            parameters.append(parameter(args.vararg, inspect.Parameter.VAR_POSITIONAL))
        for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
            parameters.append(parameter(arg, inspect.Parameter.KEYWORD_ONLY, default))
        if args.kwarg:
            parameters.append(parameter(args.kwarg, inspect.Parameter.VAR_KEYWORD))
        return inspect.Signature(parameters, return_annotation=(inspect.Signature.empty
            if node.returns is None else ast.unparse(node.returns)))
    except (TypeError, ValueError):
        return None


def source_callable(module_name, object_name, *, method=None):
    """Return (signature, kind, doc) or None when live inspection is necessary."""
    path = _module(ROOT/module_name)
    if path is None:
        return None
    try:
        selected = _binding(path, object_name, set())
    except (OSError, SyntaxError):
        return None
    if selected is None:
        return None
    node, _ = selected
    doc = ast.get_docstring(node) or ''
    if method is not None:
        if not isinstance(node, ast.ClassDef):
            raise LookupError(method)
        if _dynamic_binding(node.body, method):
            return None
        node = next((item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == method), None)
        if node is None:
            return None  # A declared inherited/assigned method may need runtime inspection.
        signature = _signature(node, drop_self=True)
        return None if signature is None else (signature, 'model_method', ast.get_docstring(node) or '')
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        signature = _signature(node)
        return None if signature is None else (signature, 'function', doc)
    root_tree = ast.parse(path.read_text(encoding='utf-8'))
    if _mapping(root_tree, '__all__') is None and any(isinstance(item, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == '__all__' for t in item.targets) for item in root_tree.body):
        return None
    exports = _mapping(root_tree, '__all__') or ()
    documented = _mapping(root_tree, '_DOCUMENTED_REEXPORTS') or {}
    if object_name not in exports and object_name not in documented:
        raise LookupError('not an exported class')
    decorators = [d.func if isinstance(d, ast.Call) else d for d in node.decorator_list]
    if any(not isinstance(d, ast.Name) or d.id != 'dataclass' for d in decorators):
        return None
    if node.keywords or node.bases or any(
        (isinstance(item, ast.FunctionDef) and item.name == '__new__')
        or (isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__signature__' for t in item.targets))
        for item in node.body):
        return None
    init = next((item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == '__init__'), None)
    if _dynamic_binding(node.body, '__init__'):
        return None
    if init is not None:
        signature = _signature(init, drop_self=True)
        return None if signature is None else (signature, 'constructor', doc)
    if node.bases or not any(isinstance(d, ast.Name) and d.id == 'dataclass' for d in decorators):
        return None
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call):
            for option in decorator.keywords:
                if option.arg in ('init', 'kw_only'):
                    try:
                        if _literal(option.value) != (option.arg == 'init'):
                            return None
                    except (TypeError, ValueError):
                        return None
    parameters = []
    try:
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if isinstance(item.annotation, ast.Subscript) and isinstance(item.annotation.value, ast.Name) and item.annotation.value.id in ('ClassVar', 'InitVar'):
                    return None
                parameters.append(inspect.Parameter(item.target.id, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=inspect.Parameter.empty if item.value is None else _literal(item.value),
                    annotation=ast.unparse(item.annotation)))
        return inspect.Signature(parameters), 'constructor', doc
    except (TypeError, ValueError):
        return None
