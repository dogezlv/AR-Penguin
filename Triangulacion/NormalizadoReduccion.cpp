#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <cmath>  // Para std::abs

int main() {
    std::ifstream inputFile("Pinguino.asc");  // Archivo de entrada

    if (!inputFile.is_open()) {
        std::cerr << "Error al abrir el archivo de entrada." << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> data;
    std::string line;
    double minValue = std::numeric_limits<double>::max();

    // Leer los datos y encontrar el valor mínimo en las columnas 4, 5 y 6
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        std::string value;
        std::vector<double> row;
        int count = 0;

        while (std::getline(iss, value, ' ')) {
            double doubleValue = std::stod(value);
            if (count >= 3 && count < 6) {
                row.push_back(doubleValue);
                if (doubleValue < minValue) {
                    minValue = doubleValue;
                }
            }
            count++;
        }
        if (row.size() == 3) {
            data.push_back(row);
        }
    }

    inputFile.close();

    // Calcular el desplazamiento necesario
    double offset = std::abs(minValue) + 1.0;

    // Crear el archivo de salida y escribir los datos ajustados
    std::ofstream outputFile("Puntos.asc");

    if (!outputFile.is_open()) {
        std::cerr << "Error al abrir el archivo de salida." << std::endl;
        return 1;
    }

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            outputFile << (row[i] + offset) << " ";
        }
        outputFile << std::endl;
    }

    outputFile.close();

    std::cout << "Archivo Puntos.asc generado correctamente." << std::endl;
    return 0;
}
