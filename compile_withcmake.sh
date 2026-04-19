#!/bin/bash
set -euo pipefail
cd "${0%/*}" || exit  # Run from current directory (source directory) or exit

if [[ -z "${NCORES:-}" ]]; then 
    case "$(uname)" in
        Darwin)
            NCORES=$(sysctl -n hw.ncpu)
            ;;
        *)
            NCORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null)
            ;;
    esac
    [ -n "$NCORES" ] || NCORES=1
fi

rm -rf deploy
rm -rf build
mkdir build

ADDITIONAL_ARGS="-DBUILD_GPU_HEIPA=On -DGPU_HEIPA_BUILD_APPS=Off"
REQUEST_GPU_HEIPA=1
REQUEST_PYTHON_MODULE=0

for arg in "$@"; do
    if [ "$arg" == "BUILDPYTHONMODULE" ]; then
        REQUEST_PYTHON_MODULE=1
        ADDITIONAL_ARGS="$ADDITIONAL_ARGS -DBUILDPYTHONMODULE=On"
    elif [ "$arg" == "BUILD_GPU_HEIPA" ]; then
        # Backward-compatible no-op: GPU-HeiPa is enabled by default.
        :
    elif [ "$arg" == "NO_BUILD_GPU_HEIPA" ]; then
        REQUEST_GPU_HEIPA=0
        ADDITIONAL_ARGS="$ADDITIONAL_ARGS -DBUILD_GPU_HEIPA=Off"
    else
        ADDITIONAL_ARGS="$ADDITIONAL_ARGS $arg"
    fi
done

if [ "$REQUEST_GPU_HEIPA" -eq 1 ]; then
    echo
    echo "Preparing GPU-HeiPa dependencies (Kokkos/KokkosKernels, AUTO mode)..."
    ./extern/gpu_heipa/build.sh --download-kokkos=AUTO --max-threads="$NCORES" --deps-only=ON
fi

if [ "$REQUEST_PYTHON_MODULE" -eq 1 ]; then
    pybind11_location="$(pip show pybind11 | grep '^Location:' | cut -d ' ' -f 2 || true)"
    if [ -n "$pybind11_location" ]; then
        export pybind11_DIR="$pybind11_location/pybind11/share/cmake/pybind11"
    else
        echo "ERROR: BUILDPYTHONMODULE requested, but pybind11 is not installed in the active Python environment."
        exit 1
    fi
fi

cmake -B build -DCMAKE_BUILD_TYPE=Release $ADDITIONAL_ARGS
cmake --build build -j "$NCORES"

echo
echo "Copying files into 'deploy'"
mkdir deploy

# Serial
echo
echo "... executables (serial)"
for name in \
    edge_partitioning \
    evaluator \
    global_multisection \
    graphchecker \
    kaffpa \
    label_propagation \
    node_ordering \
    node_separator \
    partition_to_vertex_separator \
;
do
    cp ./build/"$name" deploy/
done
if [[ -f ./build/kaffpaE ]]; then 
    cp ./build/kaffpaE deploy/
fi

# These may or may not exist
for name in \
    fast_node_ordering \
    ilp_improve \
    ilp_exact \
;
do
    if [ -f build/"$name" ]; then
        cp ./build/"$name" deploy/
    fi
done

echo "... libraries (serial)"
mv -f ./build/libkahip_static.a deploy/libkahip.a
cp ./build/libkahip* deploy/

echo "... headers (serial)"
cp ./interface/kaHIP_interface.h deploy/


# Parallel
echo
if [ -d ./build/parallel ]
then
    echo "... executables (parallel)"

    cp ./build/parallel/parallel_src/dsp* ./deploy/distributed_edge_partitioning
    cp ./build/parallel/parallel_src/g* ./deploy
    cp ./build/parallel/parallel_src/parhip ./deploy/parhip
    if ls ./build/parallel/parallel_src/parhip*.pc >/dev/null 2>&1; then
        cp ./build/parallel/parallel_src/parhip*.pc ./deploy/
    fi
    cp ./build/parallel/parallel_src/toolbox* ./deploy/

    echo "... libraries (parallel)"
    cp ./build/parallel/parallel_src/libparhip_inter*.a deploy/libparhip.a
    mkdir deploy/parallel
    cp ./build/parallel/modified_kahip/lib*.a deploy/parallel/libkahip.a

    echo "... headers (parallel)"
    cp ./parallel/parallel_src/interface/parhip_interface.h deploy/

else
    echo "Parhip was not built - skipping deployment"
fi

# maybe adapt paths here and python version here
if [[ " $* " == *" BUILDPYTHONMODULE "* ]]; then
    cp misc/pymodule/call* deploy/
    cp ./build/kahip.cp* deploy/
fi

echo
echo "Created files in deploy/"
echo =========================
(cd deploy 2>/dev/null && ls -dF *)
echo =========================
echo
echo "Can remove old build directory"
echo

# ------------------------------------------------------------------------
