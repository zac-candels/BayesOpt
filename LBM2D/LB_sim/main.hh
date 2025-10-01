#include <math.h>
#include <stdlib.h>

#include <lbm.hh>

int lx = 200;             // Simulation size in x direction
int ly = 200;             // Simulation size in y direction
int lz = 1;               // Simulation size in z direction
int timesteps = 5000;     // Total number of iterations
int saveInterval = 1000;  // How frequently to save order parameter, velocity etc
double radius = 25.0;     // Droplet radius
double theta = 90.0;      // Contact angle
double A = 0.003;         // Parameter for free energy functional (depends on surface tension and interface width, you can
                          // Ignore for now)
double kappa = A * 9 / 8; // Parameter for free energy functional (depends on surface tension and interface width, you can ignore for now)
double surfacetension = sqrt(2 * A * kappa) / 6;  // Surface tension in lattice units
double interfacewidth = sqrt(8 * kappa / A);      // Width of the diffuse interface between fluids
double posx = 0.0;        // Droplet position in the x direction
double posy = 0.0;        // Droplet position in the y direction
//double posz = 0.0;        // Droplet position in the z direction, uncomment for full 3D
double dens1 = 1;         // Droplet density
double dens2 = 1;         // Air density
double bodyforcex = 0;    // Magnitude of body force acting on the droplet in the x direction
std::string datadir = "data/"; // Data directory
double postfraction = 0.5; // Approximate fraction of area in the x direction taken up by posts (approximate because we have a finite number of lattice points)
double postfractionZ = 0.5; // Approximate fraction of area in the x direction taken up by posts
int equilibriumtimesteps = 0; // Number of timesteps until body force is applied
int postheight = 10;      // Height of the posts
int nbPost = 10;          // Number of posts in the x direction
int nbPostZ = 1;          // Number of posts in the z direction
double tau1 = 1;          // Relaxation time of droplet
double tau2 = 1;          // Relaxation time of droplet

// Class that will handle input parameters
InputParameters params;

// Lattice classes, contain information about the size of the simulation and the parallelisation.
#ifdef MPIPARALLEL //If you have -DM
using Lattice = LatticePropertiesRuntime<ParallelX<2> /*Class to handle parallelisation*/, 3 /*Number of physical dimensions*/>;
#else
using Lattice = LatticePropertiesRuntime<NoParallel, 3>;
#endif

// Function used to define the solid geometry
int initBoundary(const int k) {
    // x coordinate of the lattice node k
    int xx = computeXGlobal<Lattice>(k);
    // y coordinate of the lattice node k
    int yy = computeY(ly, lz, k);
    // z coordinate of the lattice node k
    int zz = computeZ(ly, lz, k);
        
    // Layer of nodes with id 1 at the bottom of the domain, we will later apply bounce back to these nodes
    if (yy <= 1) return 1;

    // Layer of nodes with id 2 at the top of the domain, we will later apply the mirror boundary condition to these nodes
    if (yy >= ly - 2) return 2;
    //if (yy == ly - 3) return 2;

    // Pillars in x and z direction with id 1, we will later apply bounce back to these nodes
    if (xx % (lx / nbPost) < postfraction * lx / nbPost && // Take the remainder after dividing the current x coordinate by the lx/nbPost.
                                                           // Postfraction of this should be a post, (1-postfraction) should be fluid.
        zz % (lz / nbPostZ) < postfractionZ * lz / nbPostZ && // Same for z
        yy < postheight) // y less than post height
    {
        return 1;
    }

    // If none of these conditions apply, return 0 to signify a fluid node
    return 0;
}

//Initialises the fluid with a bulk value of 1 for the droplet and a bulk value of 0 for air
double initFluid(const int k) {
    // x coordinate of the lattice node k
    int xx = computeXGlobal<Lattice>(k);
    // y coordinate of the lattice node k
    int yy = computeY(ly, lz, k);
    // z coordinate of the lattice node k
    //int zz = computeZ(ly, lz, k); // Uncomment for 3D

    // Radial distance from the centre of the droplet at (posx, posy)
    double rr2 = (xx - posx) * (xx - posx) + (yy - posy) * (yy - posy);
    // Double rr2 = (xx - posx) * (xx - posx) + (yy - posy) * (yy - posy) + (zz - posz) * (zz - posz); // Switch these for 3D

    // Smooth droplet
    return (0.5 - 0.5 * tanh(2 * (sqrt(rr2) - radius) / (4.0))) * // Circle with given radius
           (0.5 + 0.5 * tanh(2 * (yy - postheight) / (4.0)));
    // Sharp droplet
    // If (rr2 < radius * radius&&yy>postheight)
    //     return 1;
    // Else
    //     return 0;
}

// Function to initialise the parameters from an input file
void initParams(std::string inputfile) {
    params.addParameter<int>(lx, "lx");
    params.addParameter<int>(ly, "ly");
    params.addParameter<int>(lz, "lz");
    params.addParameter<int>(timesteps, "timesteps");
    params.addParameter<int>(saveInterval, "saveInterval");
    params.addParameter<double>(radius, "radius");
    params.addParameter<double>(theta, "theta");
    params.addParameter<double>(A, "A");
    params.addParameter<double>(kappa, "kappa");
    params.addParameter<double>(surfacetension, "surfacetension");
    params.addParameter<double>(interfacewidth, "interfacewidth");
    params.addParameter<double>(posx, "posx");
    params.addParameter<double>(posy, "posy");
    //params.addParameter<double>(posz, "posz"); // Uncomment for 3D
    params.addParameter<double>(dens1, "dens1");
    params.addParameter<double>(dens2, "dens2");
    params.addParameter<double>(tau1, "tau1");
    params.addParameter<double>(tau2, "tau2");
    params.addParameter<std::string>(datadir, "datadir");
    params.addParameter<double>(postfraction, "postfraction");
    params.addParameter<double>(postfractionZ, "postfractionZ");
    params.addParameter<double>(bodyforcex, "bodyforcex");
    params.addParameter<int>(postheight, "postheight");
    params.addParameter<int>(nbPost, "nbPost");
    params.addParameter<int>(nbPostZ, "nbPostZ");
    params.addParameter<int>(equilibriumtimesteps, "equilibriumtimesteps");

    /*
    If you want to add a parameter here, follow the format above
    params.addParameter< *parameter type* >( *parameter name in this file*, *parameter name in the input file* );
    */

    // Read the input file and initialise the parameters
    params.readInput(inputfile);

    // Initialise free energy parameters from the surface tension and interface width
    A = 12 * surfacetension / interfacewidth;
    kappa = pow(interfacewidth, 2) / 8.0 * A;

    // Initialise the lattice class with the simulation size
    Lattice::init(lx, ly, lz);

}

//
///////// Simulation details
//
using densitygradients =
    GradientsMultiStencil<Density<>, CentralXYZBounceBack, CentralQBounceBack>;
using velocitygradients =
    GradientsDirectional<Velocity<>, CentralXYZBounceBackDirectional>;

template<class TLattice>
using DefaultTraitPressureWellBalancedNew = typename DefaultTrait<TLattice,2>::template SetBoundary<BounceBack, FreeSlip>
                                                               :: template SetProcessor<std::tuple<MirrorBoundary<Density<>>,MirrorBoundary<Velocity<>,Lattice::NDIM>>,std::tuple<densitygradients,velocitygradients,ViscousStressCalculator>>
                                                               :: template SetForce< PressureForceWellBalanced<WellBalancedForce<NoTauDependence>,ChemicalForceBinaryMu>, PressureForceWellBalancedTau<SimpleForcingQ<SplitGuoPrefactor>,ChemicalForceBinaryMu>, BodyForce<> >;


// Function to create the pressure LBM model (solves navier stokes)
auto initPressure() {

    // PressureLee is the model class, and we give the lattice and traits as template parameters.
    FlowFieldPressureWellBalanced<Lattice, DefaultTraitPressureWellBalancedNew<Lattice>> pressure;

    // Boundary ids to apply the LBM model
    pressure.setCollideID({0});
    //pressure.template getBoundary<EquilibriumPL>().setNodeID(11);
    // Set relaxation times of each component
    pressure.setTaus(tau1,tau2);
    pressure.setDensities(dens1,dens2);

    pressure.template getBoundary<BounceBack>().setNodeID(1);

    // Apply the mirror boundary condition on all nodes with id 2
    pressure.template getBoundary<FreeSlip>().setNodeID(2);

    // Set magnitude of bodyforce and component to apply it to, will be zero until AfterEquilibration is called
    pressure.template getForce<BodyForce<>>().setForce({0, 0}, // 0 in the x direction, 0 in the y direction
                                                        0); // Act on 0 component (droplet)

    pressure.template getProcessor<densitygradients>().setBoundaryID({1});
    pressure.template getProcessor<velocitygradients>().setBoundaryID({1});
    pressure.template getProcessor<MirrorBoundary<Density<>>>().setNodeID(2);
    pressure.template getProcessor<MirrorBoundary<Velocity<>,Lattice::NDIM>>().setNodeID(2);                         

    // Return the model so it can be used in main.cc
    return pressure;
}
using orderparamgradients = GradientsMultiStencil<OrderParameter<>, CentralXYZBounceBack, CentralQBounceBack, LaplacianCentralWetting>;


template<int N, int Nmax, class TLattice>
using DefaultTraitWellBalancedCH2 = typename DefaultTrait<TLattice,Nmax>::template SetBoundary<BounceBack, FreeSlip>
                                                                ::template SetProcessor<std::tuple<LinearWetting,MirrorBoundary<OrderParameter<>>>,std::tuple<orderparamgradients, ChemicalPotentialCalculatorBinaryLee>,std::tuple<MirrorBoundary<ChemicalPotential<>>>> // Note the inclusion of a class to calculate the chemical potential
                                                               :: template SetForce< NoSource<N> >;

// Function to create the order parameter LBM model (solves cahn hilliard)
auto initBinary() {

    // BinaryLee is the model class, and we give the lattice and traits as template parameters.
    //BinaryWellBalanced<Lattice, typename DefaultTraitBinaryWellBalanced<Lattice>::template SetStencil<D2Q5>> binary;
    WellBalancedCH<0,2,Lattice, DefaultTraitWellBalancedCH2<0,2,Lattice>::template AddProcessor<std::tuple<GradientsMultiStencil<ChemicalPotential<0>,CentralQBounceBack,CentralXYZBounceBack>>>> binary;

    // Boundary ids to apply the LBM model
    binary.setCollideID({0,11});

    // Chemical potential calculation needs the free energy parameters
    binary.template getProcessor<ChemicalPotentialCalculatorBinaryLee>().setA(A);
    binary.template getProcessor<ChemicalPotentialCalculatorBinaryLee>().setKappa(kappa);
    
    binary.template getProcessor<ChemicalPotentialCalculatorBinaryLee>().setOmega(0.01);

    binary.setAij({-1,0});    
    
    binary.template getBoundary<BounceBack>().setNodeID(1);

    // Apply the mirror boundary condition on all nodes with id 2
    binary.template getBoundary<FreeSlip>().setNodeID(2);

    binary.template getProcessor<orderparamgradients>().setBoundaryID({1});

    binary.template getProcessor<MirrorBoundary<OrderParameter<>>>().setNodeID(2);
    binary.template getProcessor<MirrorBoundary<ChemicalPotential<>>>().setNodeID(2);

    double wettingprefactor = -2 * cos(theta * M_PI / 180.0) * sqrt(2 * A / kappa);
    
    binary.template getProcessor<orderparamgradients>().setWettingPrefactor(wettingprefactor);
    // Stabilisation parameter to stop the order parameter going below 0


    return binary;
}


//Will apply the body force after the equilibrium timesteps
template <typename T>
void AfterEquilibration(int eqsteps, T& model) {
    if (eqsteps == equilibriumtimesteps) model.template getForce<BodyForce<>>().setForce({bodyforcex, 0}, // bodyforcex in the x direction, 0 in the y direction
                                                                                         0); // Act on 0 component (droplet)
}
