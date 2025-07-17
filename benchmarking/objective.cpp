#include <iostream>
#include <cmath>
#include <filesystem>
#include <string>
#include <fstream>
#include </home/zcandels/Projects/LBM-main/src/lbm.hh>

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
	
	return fcomplex;
}

int main()
{

    initParams("input.txt");

    double val;

    val = objectiveFn(x_coord, y_coord);

    std::cout << "val = " << val << std::endl;



    std::string dirName = "data";
    std::string fileName = dirName + "/output.dat";


    std::filesystem::create_directory(dirName);

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
