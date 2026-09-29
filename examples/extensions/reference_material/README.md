<!--
SPDX-FileCopyrightText: 2026 Haoming Luo and AgentFEM contributors
SPDX-License-Identifier: Apache-2.0
-->

# AgentFEM reference material extension

This deliberately small package is a distribution-boundary acceptance fixture
and an extension-author example. It registers one material card through the
public `agentfem.extensions` entry point. The package neither vendors nor
patches AgentFEM and uses the same Model, Procedure, Result, and Verification
lifecycle as every other material.

It is not a new reference material or design allowable. Its numerical values
match the versioned AgentFEM static-cantilever Golden solely so the installed
extension path can be checked with an independent, deterministic result.
