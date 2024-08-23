#ifndef WAVE_EQUATION_SERIAL_HPP
#define WAVE_EQUATION_SERIAL_HPP


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
class WaveEquationSerial
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
     */
    void setup();

    /**
     * @brief Completes the construction of the problem by assembling objects that do not directly depend
     * on whether the triangulation is built from a file or not.
     */
    void complete_setup();

    /**
     * @brief Constructs the mass matrix and laplace matrix for the problem. 
     * 
     * @param builtin Whether to use the builtin methods rather then computing the matrices manually.
     */
    void assemble_matrices(const bool& builtin = false);

    /**
     * @brief Runs the solver by iteratively computing the right hand side of the position equation
     * (u), solving the first system of equations (u_n+1) with the conjugate gradient method,
     * computing the right hand side of the velocity equation with the newly produced u_n+1,
     * and solving it by applying the conjugate gradient method. 
     */
    void run();

    WaveEquationSerial(
        const unsigned int& degree_,
        const double& interval_,
        const double& time_step_, 
        const double& theta_
    )
    :   degree (degree_)
    ,   interval (interval_)
    ,   time_step (time_step_)
    ,   theta (theta_)
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

        virtual double value(const Point<dim>& p, const unsigned int component = 0) const override
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

        virtual double value(const Point<dim>& p, const unsigned int component = 0) const override
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

        virtual double value(const Point<dim>& p, const unsigned int component = 0) const override
        {
            if ((this->get_time() <= 0.5) && (p[0] < 0) && (p[1] < 1. / 3) &&
            (p[1] > -1. / 3))
            //if (this->get_time() <= 1.0 && p[0] == 0)
            {
                //return std::sin(3*this->get_time()) * std::exp(-this->get_time());
                return std::sin(10*this->get_time());
            }
            else
                return 0;
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

        virtual double value(const Point<dim>& p, const unsigned int component = 0) const override
        {
            if ((this->get_time() <= 0.5) && (p[0] < 0) && (p[1] < 1. / 3) &&
            (p[1] > -1. / 3))
            {
                //return 3*std::cos(3*this->get_time()) * std::exp(-this->get_time()) - std::sin(3*this->get_time()) * std::exp(-this->get_time());
                return 10*std::cos(10*this->get_time());
            }
            else
                return 0;
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

        virtual double value(const Point<dim>& p, const unsigned int component = 0) const override
        {
            return 0;
        }

    };

protected:

    /**
     * @brief Builds the right hand side of the first equation
     * 
     * @param time time of the simulation at the given iteration, used for computing the value of parameters
     */
    void assemble_u(const double& time);

    /**
     * @brief Builds the right hand side of the second equation
     * 
     * @param time time of the simulation at the given iteration, used for computing the value of parameters
     */
    void assemble_v(const double& time);
 
    /**
     * @brief Computes the forcing terms that will be used in the right hand side
     * of the each equation. This function is called once per iteration as the forcing term
     * remains equal, entering the equation effectively scaled by a constant.
     * 
     */
    void compute_forcing_terms(const double& time, const bool& builtin = false);

    /**
     * @brief Solves the position equation with the conjugate gradient method
     * 
     */
    void solve_u();

    /**
     * @brief Solves the velocity equation with the conjugate gradient method
     * 
     */
    void solve_v();

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
    const double theta;
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
    SparseMatrix<double> matrix_u;
    SparseMatrix<double> matrix_v;
    
    SparsityPattern sparsity_pattern;
    
    Vector<double> rhs;
    Vector<double> solution_u;
    Vector<double> solution_v;
    Vector<double> old_solution_u;
    Vector<double> old_solution_v;
    Vector<double> forcing_terms;

    Vector<double> tmp;

};




#endif // WAVE_EQUATION_SERIAL_HPP
