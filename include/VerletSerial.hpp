#ifndef VERLET_SERIAL_HPP
#define VERLET_SERIAL_HPP


// ==================== INCLUDES ====================

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

// Triangulation and Grid Generation
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

// Finite Element space
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_q.h>

// Degrees of freedom
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

// Linear Algebra
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>

// Numeric techniques
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

// STL libraries
#include <iostream>
#include <fstream>
#include <memory>

// ==================================================

using namespace dealii;

template <unsigned int dim>
class VerletSerial
{
public:
    /**
     * @brief Constructs the triangulation, finite element space, DoF handler and linear algebra,
     * utilising a mesh file passed as input.
     * @param mesh_filename Path to the mesh file
     */
    void setup(const std::string& mesh_filename);

    /**
     * @brief Constructs the triangulation, finite element space, DoF handler and linear algebra,
     * utilising deal.ii mesh generation infrastructure.
     * @param times Number of times the mesh will be refined
     */
    void setup(const unsigned int& times);

    /**
     * @brief Completes the construction of the problem by assembling objects that do not directly depend
     * on whether the triangulation is built from a file or not.
     */
    void complete_setup();

    /**
     * @brief Constructs the mass matrix and laplace matrix for the problem. 
     * 
     */
    void assemble_matrices();

    /**
     * @brief Runs the solver by applying the Verlet integration scheme at each time step
     */
    void run();

    VerletSerial
    (
        const unsigned int& degree_,
        const double& interval_,
        const double& time_step_
    )
    :   degree (degree_)
    ,   interval (interval_)
    ,   time_step (time_step_)
    {}

    /**
     * @brief Class describing the initialization value of the height of each point in the
     * computational domain.
     * 
     * @tparam dim 
     */
    class InitialU : public Function<dim>
    {
    public:
        InitialU(){}

        virtual double value(const Point<dim>& /*p*/, const unsigned int /*component*/ = 0) const override
        {
            return 0.0;
        }

    };

    /**
     * @brief Class describing the initialization value of the velocity of each point in the computational
     * domain.
     * 
     * @tparam dim number of physical dimensions of the problem
     */
    class InitialV : public Function<dim>
    {
    public:
        InitialV(){}

        virtual double value(const Point<dim>& /*p*/, const unsigned int /*component*/ = 0) const override
        {
            return 0.0;
        }

    };

    /**
     * @brief Class describing the boundary values related to position. 
     * 
     */
    class BoundaryU : public Function<dim>
    {
    public:
        BoundaryU(){}

        virtual double value(const Point<dim>& /*p*/, const unsigned int /*component*/ = 0) const override
        {
            return 0.0;
        }

    };

    /**
     * @brief Class describing the boundary values related to velocity. They can be extracted by computing
     * the derivative of the function describin the boundary values for position.
     * 
     */
    class BoundaryV : public Function<dim>
    {
    public:
        BoundaryV(){}

        virtual double value(const Point<dim>& /*p*/, const unsigned int /*component*/ = 0) const override
        {
            return 0.0;
        }

    };

    /**
     * @brief Class describing the forcing term.
     * 
     * @tparam dim 
     */
    class ForcingTerm : public Function<dim>
    {
    public:
        ForcingTerm(){}

        virtual double value(const Point<dim>& p, const unsigned int /*component*/ = 0) const override
        {
            if (this->get_time() <= 0.5 && ((p[0]-0.5)*(p[0]-0.5) + (p[1]-0.5)*(p[1]-0.5)) <= 0.0025)
                return 3.0;
            return 0;
        }

    };

protected:

    /**
     * @brief Computes the forcing terms that will be used in the right hand side
     * of the each equation. This function is called once per iteration as the forcing term
     * remains equal, entering the equation effectively scaled by a constant.
     * 
     * @param time time of the current iteration
     */
    void compute_forcing_terms(const double& time);

    /**
     * @brief Computes the acceleration term for the Verlet algorithm
     * 
     * @param time time of the current iteration
     */
    void compute_acceleration(const double& time);


    /**
     * @brief Outputs the result of the computation to a file
     * 
     */
    void output_results() const;
    

protected:
    // =========================================
    // PROPERTIES OF THE PROBLEM
    // =========================================
    const unsigned int degree;
    const double interval;
    const double time_step;
    double time;

    unsigned int time_step_number;

    InitialU initial_u;
    InitialV initial_v;

    BoundaryU boundary_u;
    BoundaryV boundary_v;

    ForcingTerm forcing_term;

    // =========================================
    // TRIANGULATION
    // =========================================
    Triangulation<dim> triangulation;

    // =========================================
    // FINITE ELEMENT SPACE
    // =========================================
    std::unique_ptr<FiniteElement<dim>> fe;

    // =========================================
    // DEGREES OF FREEDOM
    // =========================================
    DoFHandler<dim> dof_handler;

    // =========================================
    // QUADRATURE
    // =========================================
    std::unique_ptr<Quadrature<dim>> quadrature;

    // =========================================
    // LINEAR ALGEBRA
    // =========================================
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> lhs;

    SparsityPattern sparsity_pattern;
    
    Vector<double> rhs;
    Vector<double> solution_u;
    Vector<double> solution_v;
    Vector<double> old_solution_u;
    Vector<double> old_solution_v;
    Vector<double> forcing_terms;
    Vector<double> a_old;
    Vector<double> a_new;

    Vector<double> tmp;

};

#endif // VERLET_SERIAL_HPP
