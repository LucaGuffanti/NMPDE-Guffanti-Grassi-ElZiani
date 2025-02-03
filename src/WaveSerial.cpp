#include "WaveSerial.hpp"

// ========================================================
// ==================== IMPLEMENTATION ====================
// ========================================================

template <unsigned int dim>
void WaveEquationSerial<dim>::setup(const std::string &mesh_file)
{

    std::cout << " ======================================== " << std::endl;
    std::cout << " WAVESERIAL - SETUP (FROM FILE) " << std::endl;
    std::cout << " ======================================== " << std::endl;


    // Create the mesh. The mesh is created by reading the input file, passed as a parameter
    {
        std::cout << " Creating mesh..." << std::endl;
        std::cout << " Input file\t: " << mesh_file << std::endl;

        GridIn<dim> grid_in;
        grid_in.attach_triangulation(triangulation);
        std::ifstream input_file(mesh_file);
        grid_in.read_msh(input_file);
        std::cout << " Number of elements\t: " << triangulation.n_active_cells() << std::endl;
    }   

    // Build the finite element space. We use simplex elements with polinomials of degree 'degree'.
    {
        std::cout << " Building the finite element space..." << std::endl;
        fe = std::make_unique<FE_SimplexP<dim>>(degree);
        std::cout << " Degree of polynomials\t: " << fe->degree << std::endl;
        std::cout << " Number of degrees of freedom\t: " << fe->dofs_per_cell << std::endl;        
    }
    // Build the quadrature rule. We use Gauss-Lobatto quadratures, so the number of points 
    // is the degree of the polynomial plus one.
    {
        std::cout << " Building the quadrature rule..." << std::endl;
        quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);
        std::cout << " Number of quadrature points\t: " << quadrature->size() << std::endl;
    }

    complete_setup();
}

template <unsigned int dim>
void WaveEquationSerial<dim>::setup(const unsigned int& times)
{
    std::cout << " ======================================== " << std::endl;
    std::cout << " WAVESERIAL - GENERATING MESH" << std::endl;
    std::cout << " ======================================== " << std::endl;


    // Create the mesh. The mesh is created by reading the input file, passed as a parameter
    {
        std::cout << " Creating mesh..." << std::endl;
        GridGenerator::hyper_cube(triangulation, 0, 1);
        triangulation.refine_global(times);

        std::cout << " Number of elements\t: " << triangulation.n_active_cells() << std::endl;
    }   

    // Build the finite element space. We use rectangular elements with polinomials of degree 'degree'.
    {
        std::cout << " Building the finite element space..." << std::endl;
        fe = std::make_unique<FE_Q<dim>>(degree);
        std::cout << " Degree of polynomials\t: " << fe->degree << std::endl;
        std::cout << " Number of degrees of freedom\t: " << fe->dofs_per_cell << std::endl;        
    }
    // Build the quadrature rule. We use Gauss-Lobatto quadratures, so the number of points 
    // is the degree of the polynomial plus one.
    {
        std::cout << " Building the quadrature rule..." << std::endl;
        quadrature = std::make_unique<QGauss<dim>>(fe->degree + 1);
        std::cout << " Number of quadrature points\t: " << quadrature->size() << std::endl;
    }

    complete_setup();
}

template <unsigned int dim>
void WaveEquationSerial<dim>::complete_setup()
{
    
    // Re-initialize the DoF-Handler, and distribute the degrees of freedom across the triangulation (mesh).
    {
        std::cout << " Creating the DoF handler..." << std::endl;
        dof_handler.reinit(triangulation);
        dof_handler.distribute_dofs(*fe);
        std::cout << " Number of degrees of freedom\t: " << dof_handler.n_dofs() << std::endl;
    }

    // Build the linear algebra terms: matrices (via the DynamicSparsityPattern object) and vectors
    {
        std::cout << " Building the matrices..." << std::endl;
        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp);

        sparsity_pattern.copy_from(dsp);

        mass_matrix.reinit(sparsity_pattern);
        laplace_matrix.reinit(sparsity_pattern);
        matrix_u.reinit(sparsity_pattern);
        matrix_v.reinit(sparsity_pattern);

        std::cout << " Building the vectors..." << std::endl;
        solution_u.reinit(dof_handler.n_dofs());
        solution_v.reinit(dof_handler.n_dofs());
        old_solution_u.reinit(dof_handler.n_dofs());
        old_solution_v.reinit(dof_handler.n_dofs());
        rhs.reinit(dof_handler.n_dofs());
        forcing_terms.reinit(dof_handler.n_dofs());
        tmp.reinit(dof_handler.n_dofs());
    }

std::cout << " ======================================== " << std::endl;

}

template <unsigned int dim>
void WaveEquationSerial<dim>::assemble_matrices()
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
}

template <unsigned int dim>
void WaveEquationSerial<dim>::run()
{

    std::cout << " ======================================== " << std::endl;
    std::cout << " WAVESERIAL - SOLVING THE PROBLEM." << std::endl;
    std::cout << " ======================================== " << std::endl;

    // First, initialize the solution vectors
    initial_u.set_time(0.0);
    initial_v.set_time(0.0);

    // Interpolate the initial conditions onto the u vector
    VectorTools::interpolate(
        dof_handler,
        initial_u,
        old_solution_u
    );

    // Interpolate the initial conditions onto the v vector
    VectorTools::interpolate(
        dof_handler,
        initial_v,
        old_solution_v
    );

    // Now, we start the time loop
    time_step_number = 0;
    time = 0.0;

    output_results();

    while (time < interval)
    {
        
        std::cout << "TIME: " << time << std::endl;
        time += time_step;
        time_step_number = time_step_number + 1;

        compute_forcing_terms(time);

        assemble_u(time);

        solve_u();

        assemble_v(time);

        solve_v();

        old_solution_u = solution_u;
        old_solution_v = solution_v;

        output_results();
        const double energy = mass_matrix.matrix_norm_square(solution_v) / 2.0 + laplace_matrix.matrix_norm_square(solution_u) / 2.0;
        std::cout << "Energy\t: " << energy << std::endl;
    }
}


template <unsigned int dim>
void WaveEquationSerial<dim>::assemble_u(const double& time)
{
    rhs = 0.0;
    //cout << " Init:" << old_solution_u.l2_norm() << "    ";
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
    rhs.add(1.0, tmp);

    // lhs = M + delta_t^2 * theta^2 * A
    matrix_u.copy_from(mass_matrix);
    matrix_u.add(time_step * time_step * theta * theta, laplace_matrix);

    // Boundary conditions
    BoundaryU boundary_values_u;
    boundary_values_u.set_time(time);
    std::map<types::global_dof_index, double> boundary_values;
    for (unsigned int i=0; i <4 ; i++){
        VectorTools::interpolate_boundary_values(
            dof_handler,
            i,
            boundary_values_u,
            boundary_values
        );
    }
    MatrixTools::apply_boundary_values(
        boundary_values,
        matrix_u,
        solution_u,
        rhs
    );

}   

template <unsigned int dim>
void WaveEquationSerial<dim>::assemble_v(const double& time)
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
    rhs.add(1.0, tmp);

    // lhs = M
    matrix_v.copy_from(mass_matrix);

    // Boundary conditions
    BoundaryV boundary_values_v;
    boundary_values_v.set_time(time);
    std::map<types::global_dof_index, double> boundary_values;

    for (unsigned int i=0; i <4 ; i++){
        VectorTools::interpolate_boundary_values(
            dof_handler,
            i,
            boundary_values_v,
            boundary_values
        );
    }

    MatrixTools::apply_boundary_values(
        boundary_values,
        matrix_v,
        solution_v,
        rhs
    );
    //cout <<"rhs pst bound: "<< rhs.l2_norm()<< std::endl;
}

template <unsigned int dim>
void WaveEquationSerial<dim>::compute_forcing_terms(const double& time)
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
}

template <unsigned int dim>
void WaveEquationSerial<dim>::solve_u()
{
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);

    solver.solve(matrix_u, solution_u, rhs, PreconditionIdentity());
    std::cout << " Solution U\t: " << solver_control.last_step() << " Iterations "<< std::endl;
}

template <unsigned int dim>
void WaveEquationSerial<dim>::solve_v()
{
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);

    solver.solve(matrix_v, solution_v, rhs, PreconditionIdentity());
    std::cout << " Solution V\t: " << solver_control.last_step() << " Iterations "<< std::endl;
}

 
  template <unsigned int dim>
  void WaveEquationSerial<dim>::output_results() const
  {
    DataOut<dim> data_out;
 
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution_u, "U");
    data_out.add_data_vector(solution_v, "V");
 
    data_out.build_patches();
 
    const std::string filename =
      "solution-" + Utilities::int_to_string(time_step_number, 3) + ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);
  }
 

template class WaveEquationSerial<2>;
template class WaveEquationSerial<3>;