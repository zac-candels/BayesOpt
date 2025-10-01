#include "main.hh"

int main(int argc, char **argv) {
// Initialisation for MPI parallelisationm you can ignore this
#ifdef MPIPARALLEL
    mpi.init();
    initMPIBoundary<Lattice>();
#endif

    // Input file to read params from. See initParams in main.hh.
    initParams("input.txt");

    // Model that will calculate the time evolution of the order parameter (and density) (solves Cahn-Hilliard equation)
    auto binary = initBinary();

    // Will calculate the time evolution of the pressure and velocity (solves Navier-Stokes equation)
    auto pressure = initPressure();

    // Initialise the boundary labels. See initBoundary in main.hh, this is a function that will return a label for each
    // lattice node. {0,...} is a vector that contains the ids that correspond to fluid nodes.
    Geometry<Lattice>::initialiseBoundaries(initBoundary, {0});

    // Initialise the order parameter to a droplet above the posts. See initFluid in main.hh.
    OrderParameter<>::set<Lattice>(initFluid);

    // Will initialise the models. The lbm class can be used to evolve the LBM algorithm for both models by one timestep
    // with lbm.evolve();.
    Algorithm lbm(binary, pressure);

    // Class that will handle saving, in the given directory.
    SaveHandler<Lattice> saver(datadir);

    // Save binary file Header.mat with basic information for the simulation.
    saver.saveHeader(timesteps, saveInterval);

    // Main simulation loop
    for (int timestep = 0; timestep <= timesteps; timestep++) {
        // Save the desired parameters, producing a binary file for each.
        if (timestep % saveInterval == 0) {
            if (mpi.rank == 0) std::cout << "Saving at timestep " << timestep << "." << std::endl;

            saver.saveBoundaries(timestep);
            saver.saveParameter<ChemicalPotential<>>(timestep);
            saver.saveParameter<Density<>>(timestep);
            saver.saveParameter<Pressure<>>(timestep);
            saver.saveParameter<OrderParameter<>>(timestep);
            saver.saveParameter<ViscousDissipation<>>(timestep);
            saver.saveParameter<Velocity<>, Lattice::NDIM>(timestep);
        }
        
        // Will start to apply the bodyforce after equilibriumtimesteps timesteps. See AfterEquilibration in main.hh.
        AfterEquilibration(timestep, pressure);

        // Evolve by one timestep
        lbm.evolve();
    }
}
