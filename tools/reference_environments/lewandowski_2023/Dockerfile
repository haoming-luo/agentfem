FROM mambaorg/micromamba@sha256:e0a99b0f17a759e14c2f967dc0ca2d3a3c1ca3c62955f4d20bba770eaaf0184d

USER root
RUN micromamba install -y -n base -c conda-forge \
      python=3.10 \
      fenics-dolfin=2019.1.0 \
      mgis=2.1 \
      mfront=4.1 \
      matplotlib-base \
      pandas \
      mumps-mpi \
    && micromamba clean --all --yes
RUN micromamba install -y -n base -c conda-forge make \
    && micromamba clean --all --yes

ENV MPLBACKEND=Agg
WORKDIR /reference
