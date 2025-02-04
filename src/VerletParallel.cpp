#include "VerletParallel.hpp"

// ========================================================
// ==================== IMPLEMENTATION ====================
// ========================================================

template <unsigned int dim>
void VerletParallel<dim>::setup(const std::string &mesh_file)
{
    timer.enter_section("Setup");
    pcout << " VerletParallel - SETUP (FROM FILE) " << std::endl;
    pcout << " ======================================== " << std::endl;

    // Create the mesh. The mesh is created by reading the input file, passed as a parameter
    {
        pcout << " Creating mesh..." << std::endl;
        pcout << " Input file\t: " << mesh_file << std::endl;

        Triangulation<dim> mesh_serial;
        GridIn<dim> grid_in;
        grid_in.attach_triangulation(mesh_serial);
        std::ifstream input_file(mesh_file);
        grid_in.read_msh(input_file);

    
        GridTools::partition_triangulation(mpi_size, mesh_serial);
        const auto construction_data = TriangulationDescription::Utilities::create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
        triangulation.create_triangulation(construction_data);
        pcout << " Number of elements\t: " << triangulation.n_global_active_cells() << std::endl;
    }   

    // Build the finite element space. We use simplex elements with polinomials of degree 'degree'.
    {
        pcout << " Building the finite element space..." << std::endl;
        fe = std::make_unique<FE_SimplexP<dim>>(degree);
        pcout << " Degree of polynomials\t: " << fe->degree << std::endl;
    }

    // Build the quadrature rule. We use Gauss-Lobatto quadratures, so the number of points 
    // is the degree of the polynomial plus one.
    {
        pcout << " Building the quadrature rule..." << std::endl;
        quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);
        pcout << " Number of quadrature points\t: " << quadrature->size() << std::endl;
    }

    complete_setup();
}

template <unsigned int dim>
void VerletParallel<dim>::setup(const unsigned int& times)
{
    timer.enter_section("Setup");
    pcout << " VerletParallel - GENERATING MESH" << std::endl;
    pcout << " ======================================== " << std::endl;

    // Create the mesh. The mesh is created by reading the input file, passed as a parameter
    {
        pcout << " Creating mesh..." << std::endl;
        
        Triangulation<dim> serial_triangulation;
        GridGenerator::hyper_cube(serial_triangulation, 0, 1);
        serial_triangulation.refine_global(times);

        pcout << " Number of elements\t: " << serial_triangulation.n_active_cells() << std::endl;

        GridTools::partition_triangulation(mpi_size, serial_triangulation);
        const auto construction_data = TriangulationDescription::Utilities::create_description_from_triangulation(serial_triangulation, MPI_COMM_WORLD);
        triangulation.create_triangulation(construction_data);

    }   

    // Build the finite element space. We use rectangular elements with polinomials of degree 'degree'.
    {
        pcout << " Building the finite element space..." << std::endl;
        fe = std::make_unique<FE_Q<dim>>(degree);
        pcout << " Degree of polynomials\t: " << fe->degree << std::endl;
    }

    // Build the quadrature rule. We use Gauss-Lobatto quadratures, so the number of points 
    // is the degree of the polynomial plus one.
    {
        pcout << " Building the quadrature rule..." << std::endl;
        quadrature = std::make_unique<QGauss<dim>>(fe->degree + 1);
        pcout << " Number of quadrature points\t: " << quadrature->size() << std::endl;
    }

    complete_setup();
}

template <unsigned int dim>
void VerletParallel<dim>::complete_setup()
{

    // Re-initialize the DoF-Handler, and distribute the degrees of freedom across the triangulation (mesh).
    {
        pcout << " Creating the DoF handler..." << std::endl;
        dof_handler.reinit(triangulation);
        dof_handler.distribute_dofs(*fe);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
        pcout << " Number of degrees of freedom\t: " << dof_handler.n_dofs() << std::endl;
    }

    // Build the linear algebra terms: matrices (via the DynamicSparsityPattern object) and vectors
    {
        pcout << " Building the matrices..." << std::endl;
        TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, sparsity);
        sparsity.compress();


        mass_matrix.reinit(sparsity);
        laplace_matrix.reinit(sparsity);
        lhs.reinit(sparsity);

        pcout << " Building the vectors..." << std::endl;

        rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        forcing_terms.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        tmp.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        
        solution_u.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
        solution_u_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

        solution_v.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
        solution_v_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

        old_solution_u_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

        old_solution_v_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

        a_old_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        a_new_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    }
    timer.exit_section("Setup");
pcout << " ======================================== " << std::endl;

}

template <unsigned int dim>
void VerletParallel<dim>::assemble_matrices()
{
    timer.enter_section("Assemble Mass/Laplace");
    pcout << " VerletParallel - ASSEMBLING MATRICES." << std::endl;
    pcout << " ======================================== " << std::endl;

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int quadrature_points = quadrature->size();

    FEValues fe_values(
        *fe,
        *quadrature,
        update_values | update_gradients | update_JxW_values | update_quadrature_points
    );

    FullMatrix<double> cell_mass_matrix(dofs_per_cell);
    FullMatrix<double> cell_laplace_matrix(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    mass_matrix = 0.0;
    laplace_matrix = 0.0;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        cell_mass_matrix = 0.0;
        cell_laplace_matrix = 0.0;

        for (unsigned int q = 0; q < quadrature_points; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    cell_mass_matrix(i, j) +=
                        fe_values.shape_value(i, q) *
                        fe_values.shape_value(j, q) *
                        fe_values.JxW(q);

                    cell_laplace_matrix(i, j) +=
                        fe_values.shape_grad(i, q) *
                        fe_values.shape_grad(j, q) *
                        fe_values.JxW(q);
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        mass_matrix.add(local_dof_indices, cell_mass_matrix);
        laplace_matrix.add(local_dof_indices, cell_laplace_matrix);
    }
    mass_matrix.compress(VectorOperation::add);
    laplace_matrix.compress(VectorOperation::add);
    timer.exit_section("Assemble Mass/Laplace");
}

template <unsigned int dim>
void VerletParallel<dim>::run()
{
    timer.enter_section("Run");
    pcout << " VerletParallel - SOLVING THE PROBLEM." << std::endl;
    pcout << " ======================================== " << std::endl;

    timer.enter_subsection("Init");
    // First, initialize the solution vectors
    initial_u.set_time(0.0);
    initial_v.set_time(0.0);

    // Interpolate the initial conditions onto the u vector
    VectorTools::interpolate(
        dof_handler,
        initial_u,
        old_solution_u_owned
    );
    solution_u = old_solution_u_owned;

    // Interpolate the initial conditions onto the v vector
    VectorTools::interpolate(
        dof_handler,
        initial_v,
        old_solution_v_owned
    );
    solution_v = old_solution_v_owned;

    // Start the time loop
    time_step_number = 0;
    time = 0.0;

    output_results();
    
    lhs.copy_from(mass_matrix);


    timer.enter_section("Compute Acceleration");
    compute_acceleration(time);
    timer.exit_section("Compute Acceleration");

    a_old_owned = a_new_owned;
    timer.exit_section("Init");


    while (time < interval)
    {
        time += time_step;
        time_step_number += 1;

        pcout << "Time: " << time << std::endl;
        
        timer.enter_section("Solve U");
        solution_u_owned = old_solution_u_owned;
        tmp = old_solution_v_owned;
        tmp *= time_step;

        solution_u_owned += tmp;

        tmp = a_old_owned;
        tmp *= time_step * time_step / 2.0;

        solution_u_owned += tmp;
        solution_u = solution_u_owned;
        timer.exit_section("Solve U");


        timer.enter_section("Compute Acceleration");
        compute_acceleration(time);
        timer.exit_section("Compute Acceleration");

        timer.enter_section("Solve V");
        solution_v_owned = old_solution_v_owned;
        tmp = a_old_owned;
        tmp *= time_step / 2.0;
        solution_v_owned += tmp;

        tmp = a_new_owned;
        tmp *= time_step / 2.0;
        solution_v_owned += tmp;

        solution_v = solution_v_owned;
        timer.exit_section("Solve V");

        timer.enter_subsection("Output Results");
        output_results();
        timer.exit_section("Output Results");

        old_solution_u_owned = solution_u_owned;
        old_solution_v_owned = solution_v_owned;
        a_old_owned = a_new_owned;

        timer.enter_subsection("Energy");
        const double energy = mass_matrix.matrix_norm_square(solution_v_owned) / 2.0 + laplace_matrix.matrix_norm_square(solution_u_owned) / 2.0;
        pcout << "Energy\t: " << energy << std::endl;
        timer.exit_section("Energy");

    }
    timer.exit_section("Run");
}

template <unsigned int dim>
void VerletParallel<dim>::compute_forcing_terms(const double& time)
{

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int quadrature_points = quadrature->size();

    FEValues<dim> fe_values(
        *fe,
        *quadrature,
        update_values | update_JxW_values | update_quadrature_points
    );


    Vector<double> cell_forcing_terms(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


    forcing_terms = 0.0;
    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;
        
        fe_values.reinit(cell);
        cell_forcing_terms = 0.0;

        for (unsigned int q = 0; q < quadrature_points; ++q)
        {
            forcing_term.set_time(time);
            double forcing_term_value_new = forcing_term.value(fe_values.quadrature_point(q));
            
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                cell_forcing_terms(i) +=
                    forcing_term_value_new  *
                    fe_values.shape_value(i, q) *
                    fe_values.JxW(q); 
            }
        }

        cell->get_dof_indices(local_dof_indices);
        forcing_terms.add(local_dof_indices, cell_forcing_terms);
    }
    forcing_terms.compress(VectorOperation::add);

}

template <unsigned int dim>
void VerletParallel<dim>::compute_acceleration(const double& time)
{
    timer.enter_subsection("Forcing Terms");
    // Compute the forcing term
    compute_forcing_terms(time);
    timer.exit_section("Forcing Terms");

    // Compute the right hand side
    tmp = 0.0;
    rhs = forcing_terms;
    laplace_matrix.vmult(tmp, solution_u_owned);
    rhs -= tmp;


    // Apply the boundary conditions
    BoundaryU boundary_values_u;
    boundary_values_u.set_time(time);
    std::map<types::global_dof_index, double> boundary_values;
    for (unsigned int i = 0; i < 4; i++)
    {
        VectorTools::interpolate_boundary_values(
            dof_handler,
            i,
            boundary_values_u,
            boundary_values
        );
    }

    MatrixTools::apply_boundary_values(
        boundary_values,
        lhs,
        a_old_owned,
        rhs
    );

    // Solve the linear system
    SolverControl solver_control(1000, 1e-12);
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
    solver.solve(lhs, a_new_owned, rhs, TrilinosWrappers::PreconditionIdentity());

    pcout << " Solution A\t: " << solver_control.last_step() << " Iterations "<< std::endl;
}

template <unsigned int dim>
void VerletParallel<dim>::output_results() const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_u, "u");

  std::vector<unsigned int> partition_int(triangulation.n_active_cells());
  GridTools::get_subdomain_association(triangulation, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
    "./", "output", time_step_number, MPI_COMM_WORLD, 3);
}

template <unsigned int dim>
void VerletParallel<dim>::print_timer_data() const 
{
    timer.print_wall_time_statistics(MPI_COMM_WORLD);
}

template class VerletParallel<2>;
template class VerletParallel<3>;