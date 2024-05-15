#include "WaveParallel.hpp"

// ========================================================
// ==================== IMPLEMENTATION ====================
// ========================================================

template <unsigned int dim>
void WaveEquationParallel<dim>::setup(const std::string &mesh_file)
{

    pcout << " ======================================== " << std::endl;
    pcout << " WAVESERIAL - SETUP (FROM FILE) " << std::endl;
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

        pcout << " Number of elements\t: " << triangulation.n_active_cells() << std::endl;
    
        GridTools::partition_triangulation(mpi_size, mesh_serial);
        const auto construction_data = TriangulationDescription::Utilities::create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
        triangulation.create_triangulation(construction_data);
    }   

    // Build the finite element space. We use simplex elements with polinomials of degree 'degree'.
    {
        pcout << " Building the finite element space..." << std::endl;
        fe = std::make_unique<FE_SimplexP<dim>>(degree);
        pcout << " Degree of polynomials\t: " << fe->degree << std::endl;
        pcout << " Number of degrees of freedom\t: " << fe->dofs_per_cell << std::endl;        
    }

    complete_setup();
}

template <unsigned int dim>
void WaveEquationParallel<dim>::setup()
{
    pcout << " ======================================== " << std::endl;
    pcout << " WAVESERIAL - GENERATING MESH" << std::endl;
    pcout << " ======================================== " << std::endl;


    // Create the mesh. The mesh is created by reading the input file, passed as a parameter
    {
        pcout << " Creating mesh..." << std::endl;
        GridGenerator::hyper_cube(triangulation, -1, 1);
        triangulation.refine_global(7);

        pcout << " Number of elements\t: " << triangulation.n_active_cells() << std::endl;
    }   

    // Build the finite element space. We use rectangular elements with polinomials of degree 'degree'.
    {
        pcout << " Building the finite element space..." << std::endl;
        fe = std::make_unique<FE_Q<dim>>(degree);
        pcout << " Degree of polynomials\t: " << fe->degree << std::endl;
        pcout << " Number of degrees of freedom\t: " << fe->dofs_per_cell << std::endl;        
    }

    complete_setup();
}

template <unsigned int dim>
void WaveEquationParallel<dim>::complete_setup()
{
    

    // Build the quadrature rule. We use Gauss-Lobatto quadratures, so the number of points 
    // is the degree of the polynomial plus one.
    {
        pcout << " Building the quadrature rule..." << std::endl;
        quadrature = std::make_unique<QGauss<dim>>(fe->degree + 1);
        pcout << " Number of quadrature points\t: " << quadrature->size() << std::endl;
    }

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
        matrix_u.reinit(sparsity);
        matrix_v.reinit(sparsity);

        pcout << " Building the vectors..." << std::endl;

        rhs.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
        rhs_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        
        solution_u.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
        solution_u_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

        solution_v.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
        solution_v_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

        old_solution_u.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
        old_solution_u_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

        old_solution_v.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
        old_solution_v_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

        forcing_terms.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
        forcing_terms_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
        
        tmp.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
        tmp_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);


    }

pcout << " ======================================== " << std::endl;

}

template <unsigned int dim>
void WaveEquationParallel<dim>::assemble_matrices(const bool& builtin)
{
    // Make pcout of bools be true/false
    std::boolalpha(pcout);

    pcout << " ======================================== " << std::endl;
    pcout << " WAVESERIAL - ASSEMBLING MATRICES." << std::endl;
    pcout << " Using builtin methods\t: " << builtin << std::endl;
    pcout << " ======================================== " << std::endl;

    if (builtin)
    {
        // MatrixCreator::create_mass_matrix(
        //     dof_handler,
        //     *quadrature,
        //     mass_matrix
        // );

        // MatrixCreator::create_laplace_matrix(
        //     dof_handler,
        //     *quadrature,
        //     laplace_matrix
        // );
    }
    else
    {
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
    }
    pcout << " ======================================== " << std::endl;
}

template <unsigned int dim>
void WaveEquationParallel<dim>::run()
{

    pcout << " ======================================== " << std::endl;
    pcout << " WAVESERIAL - SOLVING THE PROBLEM." << std::endl;
    pcout << " ======================================== " << std::endl;

    // First, initialize the solution vectors
    initial_u.set_time(0.0);
    initial_v.set_time(0.0);

    // Interpolate the initial conditions onto the u vector
    VectorTools::interpolate(
        dof_handler,
        initial_u,
        old_solution_u_owned
    );
    old_solution_u = old_solution_u_owned;

    // Interpolate the initial conditions onto the v vector
    VectorTools::interpolate(
        dof_handler,
        initial_v,
        old_solution_v
    );
    old_solution_v = old_solution_v_owned;

    // Now, we start the time loop
    time_step_number = 0;
    time = 0.0;

    output_results();

    while (time < interval)
    {
        
        pcout << "TIME: " << time << std::endl;
        time += time_step;
        time_step_number = time_step_number + 1;

        compute_forcing_terms(time, false);

        assemble_u(time);

        solve_u();

        assemble_v(time);

        solve_v();
        
        old_solution_u_owned = solution_u_owned;
        old_solution_v_owned = solution_v_owned;

        old_solution_u = solution_u;
        old_solution_v = solution_v;

        output_results();
    }
}


template <unsigned int dim>
void WaveEquationParallel<dim>::assemble_u(const double& time)
{
    rhs = 0.0;
    // M*u_n
    mass_matrix.vmult(rhs, old_solution_u);

    // -delta_t^2 * theta (1-theta) * A * u_n
    laplace_matrix.vmult(tmp, old_solution_u);
    rhs.add(-time_step * time_step * theta * (1.0 - theta), tmp);

    // delta_t * M * v_n
    mass_matrix.vmult(tmp, old_solution_v);
    rhs.add(time_step, tmp);

    // delta_t^2 * theta * (theta * f_n+1 + (1-theta) * f_n)
    // With (theta * f_n+1 + (1-theta) * f_n) being stored in forcing_terms.
    tmp = forcing_terms;
    tmp *= time_step * time_step * theta;
    rhs.add(1.0, forcing_terms);

    // lhs = M + delta_t^2 * theta^2 * A
    matrix_u.copy_from(mass_matrix);
    matrix_u.add(time_step * time_step * theta * theta, laplace_matrix);

    // Boundary conditions
    BoundaryU boundary_values_u;
    boundary_values_u.set_time(time);
    std::map<types::global_dof_index, double> boundary_values;

    VectorTools::interpolate_boundary_values(
        dof_handler,
        0,
        boundary_values_u,
        boundary_values
    );

    MatrixTools::apply_boundary_values(
        boundary_values,
        matrix_u,
        solution_u_owned,
        rhs
    );

}   

template <unsigned int dim>
void WaveEquationParallel<dim>::assemble_v(const double& time)
{
    // M * v_n
    mass_matrix.vmult(rhs, old_solution_v);

    // -delta_t * theta * A * u_n
    laplace_matrix.vmult(tmp, solution_u);
    rhs.add(-time_step * theta, tmp);

    // -delta_t * (1 - theta) * A * u_n
    laplace_matrix.vmult(tmp, old_solution_u);
    rhs.add(-time_step * (1.0 - theta), tmp);

    // delta_t * (theta F_n+1 + (1.0 - theta) * F_n)
    tmp = forcing_terms;
    tmp *= time_step;
    rhs.add(1.0, forcing_terms);

    // lhs = M
    matrix_v.copy_from(mass_matrix);

    // Boundary conditions
    BoundaryV boundary_values_v;
    boundary_values_v.set_time(time);
    std::map<types::global_dof_index, double> boundary_values;

    VectorTools::interpolate_boundary_values(
        dof_handler,
        0,
        boundary_values_v,
        boundary_values
    );

    MatrixTools::apply_boundary_values(
        boundary_values,
        matrix_v,
        solution_v_owned,
        rhs
    );
}

template <unsigned int dim>
void WaveEquationParallel<dim>::compute_forcing_terms(const double& time, const bool& builtin)
{

    if (builtin)
    {
        // TODO: add builtin use of VectorTools::create_right_hand_side.
        // Attention when using the time parameter which must be set!!!
    }
    else
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
                
                forcing_term.set_time(time - time_step);
                double forcing_term_value_old = forcing_term.value(fe_values.quadrature_point(q));

                for(unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    cell_forcing_terms(i) +=
                        ( theta * forcing_term_value_new + (1.0 - theta) * forcing_term_value_old) *
                        fe_values.shape_value(i, q) *
                        fe_values.JxW(q); 
                }
            }

            cell->get_dof_indices(local_dof_indices);
            forcing_terms.add(local_dof_indices, cell_forcing_terms);
        }
        forcing_terms.compress(VectorOperation::add);
    }
}

template <unsigned int dim>
void WaveEquationParallel<dim>::solve_u()
{
    SolverControl solver_control(1000, 1e-6 * rhs.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

    solver.solve(matrix_u, solution_u_owned, rhs, TrilinosWrappers::PreconditionIdentity());
    pcout << " Solution U\t: " << solver_control.last_step() << " Iterations "<< std::endl;

    solution_u = solution_u_owned;
}

template <unsigned int dim>
void WaveEquationParallel<dim>::solve_v()
{
    SolverControl solver_control(1000, 1e-9 * rhs.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

    solver.solve(matrix_v, solution_v_owned, rhs, TrilinosWrappers::PreconditionIdentity());
    pcout << " Solution V\t: " << solver_control.last_step() << " Iterations "<< std::endl;

    solution_v = solution_v_owned;
}

template <unsigned int dim>
void WaveEquationParallel<dim>::output_results() const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution_u, "u");

  std::vector<unsigned int> partition_int(triangulation.n_active_cells());
  GridTools::get_subdomain_association(triangulation, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
    "./", "output", time_step_number, MPI_COMM_WORLD, 3);
}

 

template class WaveEquationParallel<2>;
template class WaveEquationParallel<3>;