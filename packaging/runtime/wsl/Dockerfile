FROM mambaorg/micromamba:2.3.2@sha256:ce2026639d612fbe837a35ca9d513292a8d776a24a5e6c82cccf9dddf45ad3d2

ARG AGENTFEM_VERSION
ARG MAMBA_DOCKERFILE_ACTIVATE=1

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      adduser bash ca-certificates gzip passwd sudo util-linux && \
    rm -rf /var/lib/apt/lists/*

COPY build/runtime/runtime-lock.txt /tmp/runtime-lock.txt
RUN micromamba install -y -n base --file /tmp/runtime-lock.txt && \
    micromamba clean --all --yes

COPY build/runtime/wheels/agentfem-*.whl /tmp/
RUN /opt/conda/bin/python -m pip install --no-deps --force-reinstall /tmp/agentfem-*.whl && \
    rm -f /tmp/agentfem-*.whl /tmp/runtime-lock.txt

COPY packaging/runtime/wsl/wsl-distribution.conf /etc/wsl-distribution.conf
COPY packaging/runtime/wsl/wsl.conf /etc/wsl.conf
COPY packaging/runtime/wsl/oobe.sh /usr/lib/agentfem/oobe.sh
COPY packaging/runtime/wsl/terminal-profile.json /usr/lib/agentfem/terminal-profile.json
COPY packaging/runtime/wsl/agentfem-profile.sh /etc/profile.d/agentfem.sh
COPY build/runtime/THIRD_PARTY_NOTICES.txt /usr/lib/agentfem/THIRD_PARTY_NOTICES.txt
COPY build/runtime/third-party-components.json /usr/lib/agentfem/third-party-components.json
COPY build/runtime/GMSH-LICENSE.txt /usr/lib/agentfem/GMSH-LICENSE.txt
COPY build/runtime/runtime-sbom.cdx.json /usr/lib/agentfem/runtime-sbom.cdx.json
COPY build/runtime/runtime-release.json /opt/conda/share/agentfem/runtime-release.json

RUN chmod 755 /usr/lib/agentfem/oobe.sh /etc/profile.d/agentfem.sh && \
    printf '%s\n' "$AGENTFEM_VERSION" > /usr/lib/agentfem/runtime-version && \
    mkdir -p /var/lib/agentfem

WORKDIR /root
CMD ["/bin/bash"]
