#include <iostream>
#include <cmath>
#include <filesystem>
#include <string>
#include <fstream>
#include "../LBM2D/src/lbm.hh"

double x_coord; 
double y_coord;

InputParameters params;

void initParams(std::string inputfile)
{

    params.addParameter<double>(x_coord, "x_coord");
    params.addParameter<double>(y_coord, "y_coord");

    params.readInput(inputfile);
}

double objectiveFn(double x, double y)
{
	double x2 = pow(x,2);
	double y2 = pow(y,2);
	double pi = 3.14159265;
	double fcomplex = sin(6*x) + sin(6*y) - pow(x - pi/2, 2) - pow(y - pi/2, 2);
	double fcrazy = 20 + x2 - 10*cos(2 * pi * x) + y2 - 10*cos(2 * pi * y);
	double fsimple = 1 - (x2 + y2);
	
	return fcrazy;
}

int main(int argc, char* argv[])
{

    // std::string inputfile = "input.txt";
    // if(argc > 1)
    // {
    //     inputfile = argv[1];
    // }

    initParams("input.txt");

    double val;

    std::cout << "x = " << x_coord << ", y = " << y_coord << std::endl;

    val = objectiveFn(x_coord, y_coord);

    std::cout << "val = " << val << std::endl;

    std::string dir = "data";

    //if(!std::filesystem::exists(dir))
    //{
    //    std::filesystem::create_directory(dir);
    //}

    //std::string dirName = "./data/data_x" + std::to_string(x_coord) + "_y" + std::to_string(y_coord);
    std::string fileName = "output.dat";


    //std::filesystem::create_directory(dirName);

    // Open the file and write val to it
    std::ofstream outFile(fileName);
    if (!outFile) 
    {
        std::cerr << "Failed to open file: " << fileName << '\n';
        return 1;
    }

    outFile << val << '\n';
    outFile.close();

    std::cout << "Value written to " << fileName << '\n';






}
