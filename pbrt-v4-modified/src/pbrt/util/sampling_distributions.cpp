#include <pbrt/util/sampling_distributions.h>

#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/util.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <pbrt/util/error.h>
#include <pbrt/util/print.h>
#include <pbrt/util/math.h>



#include <fstream>
#include <sstream>

namespace pbrt {

WavelengthSamplingPDFs *globalSamplingTable = nullptr;

static pstd::vector<Float> LoadWavelengthSamplingCSV(const std::string &filename);
static pstd::vector<Float> NormalizePDFs(const pstd::vector<Float> &weights, Float dx = lambdaStep);
static pstd::vector<Float> ComputeCDFs(const pstd::vector<Float> &pdfs, Float dx = lambdaStep);
static Float trapezoid(pstd::span<const Float> y, Float dx = lambdaStep);

void InitializeWavelengthSamplingTable(const std::string &filename) {
    if (globalSamplingTable)
        return;

    // pstd::vector<Float> host_weights = LoadWavelengthSamplingCSV(filename);
    pstd::vector<Float> host_weights = LoadWavelengthSamplingCSV(filename);    // Assume pdfs are already normalized using the trapezoidal rule
    pstd::vector<Float> host_pdfs = NormalizePDFs(host_weights);
    pstd::vector<Float> host_cdfs = ComputeCDFs(host_pdfs);

    // for (int w = 0; w < NSpectrumSamples; ++w) {
    //     pstd::span<const Float> pdf_row_span(&host_pdfs[w * NumDistributionBins], NumDistributionBins);
    //     Float iin = trapezoid(pdf_row_span);
    //     Printf("Calculated pdf integral value: %f\n", iin);

    //     // const Float last = host_cdfs.back();
    //     // Printf("Last element in cdf: %f\n", last);
    // }

    pstd::vector<Float> host_mix_pdf(NumDistributionBins, Float(0));
        for (int b = 0; b < NumDistributionBins; ++b)
            for (int k = 0; k < NSpectrumSamples; ++k)
                host_mix_pdf[b] += host_pdfs[k * NumDistributionBins + b];
    
    Float in = trapezoid(host_mix_pdf);
    Printf("Calculated mix integral value: %f\n", in);
    for (Float &v : host_mix_pdf) v /= in;

    // Float norm = 0;
    // for (Float v : host_mix_pdf) norm += v;
    // norm *= lambdaStep;
    // Float mix_pdf_integral = trapezoid(host_mix_pdf);
    // if (mix_pdf_integral > 0) {
    //     for (Float &v : host_mix_pdf)
    //         v /= mix_pdf_integral;
    // }
    // in = trapezoid(host_mix_pdf);
    // Printf("Calculated mix integral value: %f\n", in);

    // Normalize the pdfs and cdfs using dx. !!!!
    // inkscape for visualization, adobe illustrator (maybe), Desmos has some stuff that might be useful.

    pstd::vector<Float> host_wavelengths(NumDistributionBins);
    for (int i = 0; i < NumDistributionBins; ++i) {
        host_wavelengths[i] = Lerp(Float(i) / Float(NumDistributionBins - 1),
                                   Lambda_min, Lambda_max);
    }

    Float *gpu_wavelengths;
    Float *gpu_pdfs;
    Float *gpu_cdfs;
    Float *gpu_mix_pdf;
    WavelengthSamplingPDFs *gpu_table_struct;

    cudaMallocManaged(&gpu_wavelengths, host_wavelengths.size() * sizeof(Float));
    cudaMallocManaged(&gpu_pdfs, host_pdfs.size() * sizeof(Float));
    cudaMallocManaged(&gpu_cdfs, host_cdfs.size() * sizeof(Float));
    cudaMallocManaged(&gpu_mix_pdf,  host_mix_pdf.size() * sizeof(Float));
    cudaMallocManaged(&gpu_table_struct, sizeof(WavelengthSamplingPDFs));

    Printf("Starting upload of wavelength sampling distributions to GPU...\n");

    for (size_t i = 0; i < host_wavelengths.size(); ++i)
        gpu_wavelengths[i] = host_wavelengths[i];
    for (size_t i = 0; i < host_pdfs.size(); ++i)
        gpu_pdfs[i] = host_pdfs[i];
    for (size_t i = 0; i < host_cdfs.size(); ++i)
        gpu_cdfs[i] = host_cdfs[i];
    for (int b = 0; b < NumDistributionBins; ++b) {
        gpu_mix_pdf[b] = host_mix_pdf[b];
    }

    gpu_table_struct->wavelengths = gpu_wavelengths;
    gpu_table_struct->pdfs_flat = gpu_pdfs;
    gpu_table_struct->cdfs_flat = gpu_cdfs;
    gpu_table_struct->mix_pdf  = gpu_mix_pdf;

    globalSamplingTable = gpu_table_struct;

    Printf("Finished preparing wavelength sampling distributions on managed memory.\n");
    fflush(stdout);
}


static pstd::vector<Float> LoadWavelengthSamplingCSV(const std::string &filename) {
    std::ifstream file(filename);
    if (!file)
        ErrorExit("Failed to open sampling CSV file: %s", filename);

    pstd::vector<Float> pdfs;
    std::string line;
    int rowCount = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;

        std::stringstream ss(line);
        std::string item; 
        for (int i = 0; i < NumDistributionBins; ++i) {
            if (!std::getline(ss, item, ','))
                ErrorExit("CSV format error: Not enough columns at row %d in file %s.",
                          rowCount, filename);
            
            try {
                // pdfs.push_back(std::stof(item));
                Float value = std::stof(item);
                if (value == 0.0f) {
                    value = 1e-8;
                }
                pdfs.push_back(value);
            } catch (const std::invalid_argument &ia) {
                ErrorExit("CSV format error: Could not parse float from '%s' at row %d, column %d in file %s.",
                          item, rowCount, i, filename);
            }
        }
        rowCount++;
    }

    if (rowCount != NSpectrumSamples)
        ErrorExit("CSV row count mismatch: expected %d rows but got %d rows in file %s.",
                  NSpectrumSamples, rowCount, filename);

    return pdfs;
}

static pstd::vector<Float> NormalizePDFs(const pstd::vector<Float> &weights, Float dx) {
    pstd::vector<Float> pdfs(weights.size());
    for (int w = 0; w < NSpectrumSamples; ++w) {
        const Float *weight_row = &weights[w * NumDistributionBins];
        Float *pdf_row = &pdfs[w * NumDistributionBins];
        
        // Start with half of the first and last values.
        // Float sum = (weight_row[0] + weight_row[NumDistributionBins - 1]) / 2;
        Float sum = 0;
        // Add all the interior points.
        for (int i = 0; i <NumDistributionBins; ++i) {
            sum += weight_row[i];
        }
        
        sum *= dx;


    //     Float sum = 0;
    //     for (int i = 0; i < NumDistributionBins; ++i)
    //         sum += weight_row[i];
    //     // sum *= lambdaStep;
        if (sum > 0) {
            for (int i = 0; i < NumDistributionBins; ++i)
                pdf_row[i] = weight_row[i] / sum;
        } else {
            // degenerate case: create a uniform PDF
            for (int i = 0; i < NumDistributionBins; ++i)
                pdf_row[i] = Float(1) / Float(NumDistributionBins);
        }
    }
    return pdfs;
}

static pstd::vector<Float> ComputeCDFs(const pstd::vector<Float> &pdfs, Float dx) {
    pstd::vector<Float> cdfs(pdfs.size());
    for (int w = 0; w < NSpectrumSamples; ++w) {
        const Float *pdf_row = &pdfs[w * NumDistributionBins];
        Float *cdf_row = &cdfs[w * NumDistributionBins];
        
        cdf_row[0] = pdf_row[0];
        for (int i = 1; i < NumDistributionBins; ++i) {
            // Float trapezoid_area = (pdf_row[i - 1] + pdf_row[i]) / 2.0f * dx;
            Float area = pdf_row[i] * dx;
            cdf_row[i] = cdf_row[i - 1] + area;
            // Printf("%f\n", cdf_row[i]);
        }

        if (NumDistributionBins > 0) {
             cdf_row[NumDistributionBins - 1] = 1.0f;
        }
        // cdf_row[0] = pdf_row[0];
        // Printf("%f ", cdf_row[0]);
        // for (int i = 1; i < NumDistributionBins; ++i) {
        //     cdf_row[i] = cdf_row[i - 1] + pdf_row[i];
        //     Printf("%f ", cdf_row[i]);
        // }
        // if (cdf_row[NumDistributionBins - 1] > 0) {
        //     cdf_row[NumDistributionBins - 1] = 1; 
        // }
        // Printf("\n");
    }
    return cdfs;
}

static Float trapezoid(pstd::span<const Float> y, Float dx) {
    if (y.size() < 2)
        return 0;

    // Start with half of the first and last values.
    // Float sum = (y.front() + y.back()) / 2.0f;
    Float sum = 0;

    // Add all the interior points.
    for (size_t i = 0; i < y.size(); ++i) {
        sum += y[i];
    }

    return sum * dx;
}

    
}