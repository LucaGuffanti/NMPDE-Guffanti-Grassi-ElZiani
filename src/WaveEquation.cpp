#include "WaveEquation.hpp"


void
WaveEquation::setup()
{
  // Create the mesh.
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_file_name);
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the matrices" << std::endl;

    mass_matrix.reinit(sparsity);
    stiffness_matrix.reinit(sparsity);
    lhs_matrix_u.reinit(sparsity);
    rhs_matrix_u.reinit(sparsity);

    pcout << "  Initializing the right-hand side vectors for both equations" << std::endl;
    rhs_u.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    rhs_v.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    pcout << "  Initializing the solution vectors for both equations" << std::endl;
    solution_u_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution_u.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

    solution_v_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution_v.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

void
WaveEquation::assemble_matrices()
{
  pcout << "===============================================" << std::endl;
  pcout << "Assembling the system matrices" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix      = 0.0;
  stiffness_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_mass_matrix      = 0.0;
      cell_stiffness_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                            fe_values.shape_value(j, q) *
                                            fe_values.JxW(q);

                  cell_stiffness_matrix(i, j) +=
                    fe_values.shape_grad(i, q) *
                    fe_values.shape_grad(j, q) * fe_values.JxW(q);
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      mass_matrix.add(dof_indices, cell_mass_matrix);
      stiffness_matrix.add(dof_indices, cell_stiffness_matrix);
    }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);

  // We build the matrix on the left-hand side of the algebraic problem (the one
  // that we'll invert at each timestep).
  lhs_matrix_u.copy_from(mass_matrix);
  lhs_matrix_u.add(deltat * deltat * theta * theta, stiffness_matrix);

  // We build the matrix on the right-hand side (the one that multiplies the old
  // solution un).
  rhs_matrix_u.copy_from(mass_matrix);
  rhs_matrix_u.add(-deltat * deltat * theta * (1.0 - theta), stiffness_matrix);
}

void
WaveEquation::assemble_rhs_u(const double &time)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  rhs_u = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_rhs = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // We need to compute the forcing term at the current time (tn+1) and
          // at the old time (tn). deal.II Functions can be computed at a
          // specific time by calling their set_time method.

          // Compute f(tn+1)
          forcing_term.set_time(time);
          const double f_new_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          // Compute f(tn)
          forcing_term.set_time(time - deltat);
          const double f_old_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              cell_rhs(i) += deltat * deltat * theta * (theta * f_new_loc + (1.0 - theta) * f_old_loc) *
                             fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);
      rhs_u.add(dof_indices, cell_rhs);
    }

  rhs_u.compress(VectorOperation::add);

  // Add the term that comes from the old solution.
  rhs_matrix_u.vmult_add(rhs_u, solution_u_owned);

  // Rescale the solution_v_owned vector by deltat
  solution_v_owned_rescaled = solution_v_owned;
  solution_v_owned_rescaled *= deltat;

  // And add the term to the partial rhs
  mass_matrix.vmult_add(rhs_u, solution_v_owned_rescaled);

  // We apply boundary conditions to the algebraic system.
  // TODO: Generalize based on the specific problem (or make it general for the mesh)
  {
    std::map<types::global_dof_index, double> boundary_values;

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;

    Functions::ZeroFunction<dim> zero;

    for (unsigned int i = 0; i < 4; ++i)
      boundary_functions[i] = &zero;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, lhs_matrix_u, solution_u_owned, rhs_u, false);
  }
}

void
WaveEquation::assemble_rhs_v(const double &time)
{
const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  rhs_v = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_rhs = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // We need to compute the forcing term at the current time (tn+1) and
          // at the old time (tn). deal.II Functions can be computed at a
          // specific time by calling their set_time method.

          // Compute f(tn+1)
          forcing_term.set_time(time);
          const double f_new_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          // Compute f(tn)
          forcing_term.set_time(time - deltat);
          const double f_old_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              cell_rhs(i) += deltat * (theta * f_new_loc + (1.0 - theta) * f_old_loc) *
                             fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);
      rhs_v.add(dof_indices, cell_rhs);
    }

  rhs_v.compress(VectorOperation::add);

  // Add the term that comes from the old solution.
  mass_matrix.vmult_add(rhs_v, solution_v_owned);
  solution_u_new_owned_rescaled = solution_u_owned;
  solution_u_new_owned_rescaled *= -deltat * theta;

  solution_u_old_owned_rescaled *= -deltat * (1-theta);

  stiffness_matrix.vmult_add(rhs_v, solution_u_new_owned_rescaled);
  stiffness_matrix.vmult_add(rhs_v, solution_u_old_owned_rescaled);
}


void 
WaveEquation::solve_u()
{
  SolverControl solver_control_u(10000, 1e-9 * rhs_u.l2_norm());
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control_u);
  TrilinosWrappers::PreconditionSSOR      preconditioner;
  preconditioner.initialize(
    lhs_matrix_u, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  // Keep track the n-th computed solution to the first equation in order to use it in the second equation 
  solution_u_old_owned_rescaled = solution_u_owned;

  solver.solve(lhs_matrix_u, solution_u_owned, rhs_u, preconditioner);
  pcout << "First equation:  " << solver_control_u.last_step() << " CG iterations" << std::endl;

  solution_u = solution_u_owned;
}

void 
WaveEquation::solve_v()
{
  SolverControl solver_control_v(10000, 1e-9 * rhs_u.l2_norm());
  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control_v);
  TrilinosWrappers::PreconditionSSOR      preconditioner;
  preconditioner.initialize(
    mass_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  solver.solve(lhs_matrix_u, solution_u_owned, rhs_u, preconditioner);
  pcout << "Second equation:  " << solver_control_v.last_step() << " CG iterations" << std::endl;

  solution_v = solution_v_owned;
}

void
WaveEquation::output(const unsigned int &time_step) const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution_u, "u");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
    "./", "output", time_step, MPI_COMM_WORLD, 3);
}

void
WaveEquation::solve()
{
  assemble_matrices();

  pcout << "===============================================" << std::endl;

  time = 0.0;

  // Apply the initial condition.
  {
    pcout << "Applying the initial condition" << std::endl;

    initial_u.set_time(time);
    initial_v.set_time(time);

    VectorTools::interpolate(dof_handler, initial_u, solution_u_owned);
    VectorTools::interpolate(dof_handler, initial_v, solution_v_owned);
    
    solution_u = solution_u_owned;
    solution_v = solution_v_owned;

    // Output the initial solution.
    output(0);
    pcout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;

  while (time < T - 0.5 * deltat)
    {
      time += deltat;
      ++time_step;

      pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << time << ":" << std::flush;

      assemble_rhs_u(time);
      solve_u();

      assemble_rhs_v(time);
      solve_v();
      
      output(time_step);
    }
}

// double
// WaveEquation::compute_error(const VectorTools::NormType &norm_type)
// {
//   FE_SimplexP<dim> fe_linear(1);
//   MappingFE        mapping(fe_linear);

//   const QGaussSimplex<dim> quadrature_error = QGaussSimplex<dim>(r + 2);

//   exact_solution.set_time(time);

//   Vector<double> error_per_cell;
//   VectorTools::integrate_difference(mapping,
//                                     dof_handler,
//                                     solution,
//                                     exact_solution,
//                                     error_per_cell,
//                                     quadrature_error,
//                                     norm_type);

//   const double error =
//     VectorTools::compute_global_error(mesh, error_per_cell, norm_type);

//   return error;
// }