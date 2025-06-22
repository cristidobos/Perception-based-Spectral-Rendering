#ifndef PBRT_UTIL_SAMPLING_DISTRIBUTION_H
#define PBRT_UTIL_SAMPLING_DISTRIBUTION_H

#include <pbrt/util/spectrum.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/float.h>

#include <vector>
#include <string>

namespace pbrt {

// static constexpr int NumDistributionBins = 81;
// static constexpr Float lambdaStep = 5.0;

struct WavelengthSamplingPDFs {
    Float *wavelengths;
    Float *pdfs_flat;
    Float *cdfs_flat;
    Float *mix_pdf;
};

extern WavelengthSamplingPDFs *globalSamplingTable;

void InitializeWavelengthSamplingTable(const std::string &filename);

}  // namespace pbrt

#endif // PBRT_UTIL_SAMPLING_DISTRIBUTION_H
