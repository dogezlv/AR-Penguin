#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct Vertex {
    double x, y, z;
};

struct Normal {
    double nx, ny, nz;
};

struct Face {
    int v1, v2, v3;
};

int main() {
    std::ifstream inputFile("PuntosLimpios.asc");  // Cambia el nombre del archivo a tu archivo .asc

    if (!inputFile.is_open()) {
        std::cerr << "Error al abrir el archivo de entrada." << std::endl;
        return 1;
    }

    std::ofstream outputFile("Pinguino.obj");  // Cambia el nombre del archivo a tu archivo .obj

    if (!outputFile.is_open()) {
        std::cerr << "Error al abrir el archivo de salida." << std::endl;
        return 1;
    }

    std::string line;
    std::vector<Vertex> vertices;
    std::vector<Normal> normals;

    // Leer los datos y almacenarlos como vértices y normales
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        Vertex vertex;
        Normal normal;
        if (iss >> vertex.x >> vertex.y >> vertex.z >> normal.nx >> normal.ny >> normal.nz) {
            vertices.push_back(vertex);
            normals.push_back(normal);
        }
    }

    inputFile.close();

    std::vector<Face> faces;

    // Generar caras de ejemplo (triangulación simple para demostración)
    
    for (size_t i = 0; i + 2 < vertices.size(); i += 3) {
        faces.push_back({static_cast<int>(i + 1), static_cast<int>(i + 2), static_cast<int>(i + 3)});
    }

    // Escribir vértices
    for (const auto& vertex : vertices) {
        outputFile << "v " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
    }

    // Escribir normales
    for (const auto& normal : normals) {
        outputFile << "vn " << normal.nx << " " << normal.ny << " " << normal.nz << std::endl;
    }

    // Escribir caras
    for (const auto& face : faces) {
        outputFile << "f " << face.v1 << "//" << face.v1 << " " << face.v2 << "//" << face.v2 << " " << face.v3 << "//" << face.v3 << std::endl;
    }

    outputFile.close();

    std::cout << "Archivo .obj generado correctamente." << std::endl;
    return 0;
}
