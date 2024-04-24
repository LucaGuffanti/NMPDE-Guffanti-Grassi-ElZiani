#ifndef WAVE_HPP
#define WAVE_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>


#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// Main class for the wave equation.
class WaveEquation
{
public:
  static constexpr unsigned int dim = 2;

  // Function for the forcing term. When set to zero, the wave equation becomes
  // the homogeneous wave equation, for which, with the use of the crank nicolson method,
  // energy conservation is expected.
  // TODO: Forcing term parameter passed by user.
  class ForcingTerm : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  class InitialU : public Function<dim>
  {
  public:

  };

  class InitialV : public Function<dim>
  {
  public:

  };


  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  WaveEquation(const std::string  &mesh_file_name_,
       const unsigned int &r_,
       const double       &T_,
       const double       &deltat_,
       const double       &theta_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , T(T_)
    , mesh_file_name(mesh_file_name_)
    , r(r_)
    , deltat(deltat_)
    , theta(theta_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();

  // Compute the error.
  double
  compute_error(const VectorTools::NormType &norm_type);

protected:

  // Assemble the mass and stiffness matrices.
  void
  assemble_matrices();

  // Assemble the right-hand side of the problem for the first equation.
  void
  assemble_rhs_u(const double &time);

  // Assemble the right hand side of the problem for the second equation.
  void
  assemble_rhs_v(const double &time);

  // Solve the first equation of the problem for one time step.
  void solve_u();

  // Solve the second equation of the problem for one time step.
  void solve_v();

  // Output.
  void
  output(const unsigned int &time_step) const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // Forcing term.
  ForcingTerm forcing_term;

  // Initial u
  InitialU initial_u;

  // Initial v
  InitialV initial_v;

  // Current time.
  double time;

  // Final time.
  const double T;

  // Discretization. ///////////////////////////////////////////////////////////

  // Mesh file name.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Theta parameter of the theta method.
  const double theta;

  // Mesh.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula. As the problem only has Dirichlet boundary conditions, we define here
  // a quadrature over the faces of the cells.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Mass matrix M / deltat.
  TrilinosWrappers::SparseMatrix mass_matrix;

  // Stiffness matrix A.
  TrilinosWrappers::SparseMatrix stiffness_matrix;

  // Matrix on the left-hand side for the first equation.
  TrilinosWrappers::SparseMatrix lhs_matrix_u;

  // Matrix on the right-hand side for the first equation.
  TrilinosWrappers::SparseMatrix rhs_matrix_u;

  // Right-hand side vector for the first equation.
  TrilinosWrappers::MPI::Vector rhs_u;

  // Right-hand side vector for the second equation.
  TrilinosWrappers::MPI::Vector rhs_v;

  // First equation solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_u_owned;

  // First equation solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution_u;

  // Second equation solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_v_owned;

  // Second equation solution rescaled by deltat.
  TrilinosWrappers::MPI::Vector solution_v_owned_rescaled;

  TrilinosWrappers::MPI::Vector solution_u_new_owned_rescaled;

  TrilinosWrappers::MPI::Vector solution_u_old_owned_rescaled;

  // Second equation solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution_v;
};

#endif