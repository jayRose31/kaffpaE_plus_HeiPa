#!/bin/bash
set -euo pipefail

cd "${0%/*}" || exit 1

if [[ -z "${NCORES:-}" ]]; then
    case "$(uname)" in
        Darwin)
            NCORES=$(sysctl -n hw.ncpu)
            ;;
        *)
            NCORES=$(getconf _NPROCESSORS_ONLN 2>/dev/null)
            ;;
    esac
    [[ -n "$NCORES" ]] || NCORES=2
fi

ADDITIONAL_ARGS="-DBUILD_GPU_HEIPA=On -DGPU_HEIPA_BUILD_APPS=Off"
REQUEST_GPU_HEIPA=1

for arg in "$@"; do
    if [ "$arg" == "BUILD_GPU_HEIPA" ]; then
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

# Configure once into build/, then compile only the kaffpaE target.
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release $ADDITIONAL_ARGS
cmake --build build --target kaffpaE -j "$NCORES"

echo
echo "Built target: kaffpaE"
echo "Binary: build/kaffpaE"
