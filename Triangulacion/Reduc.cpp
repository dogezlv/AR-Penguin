#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

void reducir_puntos(const std::string& input_file, const std::string& output_file, int intervalo) {
    std::ifstream infile(input_file);
    std::ofstream outfile(output_file);
    if (!infile.is_open() || !outfile.is_open()) {
        std::cerr << "No se pudo abrir el archivo." << std::endl;
        return;
    }

    std::string line;
    std::vector<std::string> header;
    std::vector<std::vector<double>> data;

    // Leer el encabezado (primeras 6 líneas)
    for (int i = 0; i < 6; ++i) {
        std::getline(infile, line);
        header.push_back(line);
    }

    // Leer los datos
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        double value;
        while (iss >> value) {
            row.push_back(value);
        }
        data.push_back(row);
    }

    // Escribir el encabezado al archivo de salida
    for (const auto& h : header) {
        outfile << h << "\n";
    }

    // Escribir los datos reducidos al archivo de salida
    for (size_t i = 0; i < data.size(); i += intervalo) {
        for (const auto& value : data[i]) {
            outfile << value << " ";
        }
        outfile << "\n";
    }

    infile.close();
    outfile.close();
}

int main() {
    std::string input_file = "pinguino.asc";
    std::string output_file = "PuntosLimpios.asc";
    int intervalo = 320; // Cambia esto al intervalo deseado

    reducir_puntos(input_file, output_file, intervalo);

    return 0;
}
