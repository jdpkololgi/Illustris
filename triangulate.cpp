#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
// Include GDel3D header files
#include "/global/homes/d/dkololgi/gDel3D/GDelFlipping/src/gDel3D/GpuDelaunay.h"

void checkGPUMemory() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "GPU Memory - Free: " << free_mem / (1024*1024*1024) << " GB, "
              << "Total: " << total_mem / (1024*1024*1024) << " GB" << std::endl;
}

std::vector<Point3> load_binary_points(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    std::vector<Point3> points;

    if (!file) {
        std::cerr << "Error opening binary file: " << filename << std::endl;
        return points;
    }

    file.seekg(0, std::ios::end);
    size_t num_bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_floats = num_bytes / sizeof(float);
    if (num_floats % 3 != 0) {
        std::cerr << "Binary file size is not a multiple of 3 floats!\n";
        return points;
    }

    size_t num_points = num_floats / 3;
    std::vector<float> buffer(num_floats);
    file.read(reinterpret_cast<char*>(buffer.data()), num_bytes);

    points.reserve(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        points.push_back({buffer[3 * i], buffer[3 * i + 1], buffer[3 * i + 2]});
    }

    return points;
}

int main() {
    const std::string input_file = "galaxies.bin";
    std::vector<Point3> host_points = load_binary_points(input_file);

    if (host_points.empty()) {
        std::cerr << "No points loaded.\n";
        return 1;
    }

    std::cout << "Loaded " << host_points.size() << " points.\n";
    
    // Check GPU memory before processing
    checkGPUMemory();
    
    // Estimate memory requirements
    size_t estimated_tets = host_points.size() * 8;  // Conservative estimate
    size_t estimated_memory = estimated_tets * (sizeof(int) * 4 + sizeof(int) * 4) * 3; // Tets + Opps + Info
    std::cout << "Estimated GPU memory needed: " << estimated_memory / (1024*1024*1024) << " GB" << std::endl;
    
    // Convert to thrust host_vector as expected by gDel3D
    Point3HVec pointVec(host_points.begin(), host_points.end());

    GDelOutput output;
    GpuDel triangulator;

    try {
        triangulator.compute(pointVec, &output);
        std::cout << "Triangulation complete.\n";
        std::cout << "Generated " << output.tetVec.size() << " tetrahedra.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Triangulation failed: " << e.what() << std::endl;
        checkGPUMemory();
        return 1;
    }

    // Optionally write tetrahedra to file
    std::ofstream outbin("tets.bin", std::ios::binary);
    if (!outbin) {
    std::cerr << "Error writing to tets.bin\n";
    return 2;
    }
    outbin.write(reinterpret_cast<const char*>(output.tetVec.data()), output.tetVec.size() * sizeof(Tet));
    outbin.close();

    return 0;
}