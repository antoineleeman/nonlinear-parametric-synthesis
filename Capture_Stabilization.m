% -----------------------------------------------------------------------------
% File: Capture_Stabilization.m
% Author: Antoine Leeman (aleeman@ethz.ch)
% Date: 09th September 2023
% License: MIT
% Reference:
%{
@inproceedings{leeman2023_CDC,
title={Robust Optimal Control for Nonlinear Systems with Parametric Uncertainties via System Level Synthesis},
author={Leeman, Antoine P. and Sieber, Jerome and Bennani, Samir and Zeilinger, Melanie N.},
booktitle = {Proc. of the 62nd IEEE Conf. on Decision and Control (CDC)},
doi={10.1109/CDC49753.2023.10383271},
pages={4784-4791},
year={2023}}
%}
% Link: https://arxiv.org/abs/2304.00752
% -----------------------------------------------------------------------------

classdef Capture_Stabilization
    
    properties
        nx = 6; % number of states
        nu = 2; % number of inputs
        ni = 2; % number of constraints
        nw = 6;
        
        x0;
        xf;
        E;
        
        T_max;
        w_max;
        
        F_u;
        b_u;
        F_x;
        b_x;
        
        N = 10;
        T = 5;
        dt;
        
        Q_cost;
        R_cost;
        
        theta_v = [-0.01, 0.01];        
        mu;
        I = 1;
        m = 1;
    end
    
    methods
        % state : r_x, r_x_dot, r_y, r_y_dot, \theta, \theta_dot, m
        function obj = Capture_Stabilization()
            
            
            obj.x0 = [0.7;.7;0.5;.5;.5;.5];
            obj.xf = [0;0;0;0;0;0];
            
            T_max= 1;
            w_max= 1;
            obj.T_max = T_max;
            obj.w_max = w_max;
            
            F_u = [eye(2); -eye(2)];
            b_u = T_max*[ones(4,1)];
            F_x = [eye(obj.nx);
                -eye(obj.nx)];
            b_x = ones(2*obj.nx,1);
            
            obj.F_u = F_u;
            obj.b_u = b_u;
            obj.F_x = F_x;
            obj.b_x = b_x;
            
            obj.ni = length(b_u) + length(b_x); % total number of constraints
            
            obj.Q_cost = eye(obj.nx);
            obj.R_cost = eye(obj.nu);
            E = zeros(obj.nw,obj.nx);
            E(1,5)= 0.001;E(2,4)= 0.001;E(3,6)=0.001;

            obj.E = E';
            obj.dt = obj.T/obj.N;
                        
            obj.mu = obj.dt*[1.3720 1.3559 0 3.9704 3.9066 0]; %todo: call compute_mu here
            % note: using m.compute_mu(10000), we need 69.6 seconds to obtain a tighter value
            % obj.mu = [0.2845 0.2821 0 1.4272 1.4246 0];
            
        end
          
        function x_p = ddyn(obj,x,u,integrator) % discretization of the dynamical system
            if nargin < 4
                integrator = 'multi';
            end
            h = obj.dt;
            switch integrator
                case 'single'
                    x_p = x + h*ode(obj,x,u);
                case 'multi'
                    step = 10;
                    for i = 1:step
                        x = x + h/step*ode(obj,x,u);
                    end
                    x_p = x;
                case 'rk4'
                    k_1 = ode(obj,x,u);
                    k_2 = ode(obj,x+0.5*h*k_1,u);
                    k_3 = ode(obj,x+0.5*h*k_2,u);
                    k_4 = ode(obj,x+h*k_3,u);
                    x_p = x + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h;
                otherwise
                    error('unrecognised integrator');
            end
        end
        
        
        function dt = ode(obj,x,u)
            I = obj.I;
            m = obj.m;
            r = x(1:2);
            theta = x(3);
            r_dot = x(4:5);
            theta_dot = x(6);
            T = u;
            r_theta = [cos(theta), -sin(theta); sin(theta), cos(theta)];
            dt = [r_dot;
                theta_dot;
                r_theta*T/m;
                T(1)*I;
                ];
        end
        
                
        function A = A(obj,x,u) %state matric of the discrete time linearized dynamics
            import casadi.*
            x_fun = SX.sym('x',obj.nx);
            u_fun = SX.sym('u',obj.nu);
            var_fun = [x_fun;u_fun];
            A = jacobian(obj.ddyn(x_fun,u_fun), x_fun);
            A_fun = casadi.Function('A_fun',{var_fun},{A});
            A = A_fun([x;u]);
        end
        
        
        function B = B(obj,x,u) %input matrix of the discrete time linearized dynamics
            import casadi.*
            x_fun = SX.sym('x',obj.nx);
            u_fun = SX.sym('u',obj.nu);
            var_fun = [x_fun;u_fun];
            B = jacobian(obj.ddyn(x_fun,u_fun), u_fun);
            B_fun = casadi.Function('B_fun',{var_fun},{B});
            B = B_fun([x;u]);
        end
        
        function dt = ode_dtheta(obj,x,u)
            r = x(1:2);
            theta = x(3);
            r_dot = x(4:5);
            theta_dot = x(6);
            m = obj.m;
            T = u;           
            dt = [zeros(5,1);
                T(1);
                ]; 
        end
        
        function x_p = ddyn_theta(obj,x,u,integrator)
            if nargin < 4
                integrator = 'multi';%todo: should always be the same as the integrator used in ddyn
            end
            h = obj.dt;
            switch integrator
                case 'single'
                    x_p = x + h*ode_dtheta(obj,x,u);
                case 'multi'
                    step = 10;
                    for i = 1:step
                        x = x + h/step*ode_dtheta(obj,x,u);
                    end
                    x_p = x;
                case 'rk4'
                    k_1 = ode_dtheta(obj,x,u);
                    k_2 = ode_dtheta(obj,x+0.5*h*k_1,u);
                    k_3 = ode_dtheta(obj,x+0.5*h*k_2,u);
                    k_4 = ode_dtheta(obj,x+h*k_3,u);
                    x_p = x + (1/6)*(k_1+2*k_2+2*k_3+k_4)*h;
                otherwise
                    error('unrecognised integrator');
            end
        end
        
                
        function A = A_theta(obj,x,u) %state matric of the discrete time linearized dynamics
            import casadi.*
            x_fun = SX.sym('x',obj.nx);
            u_fun = SX.sym('u',obj.nu);
            var_fun = [x_fun;u_fun];
            A = jacobian(obj.ddyn_theta(x_fun,u_fun), x_fun);
            A_fun = casadi.Function('A_fun',{var_fun},{A});
            A = A_fun([x;u]);
        end
        
        
        function B = B_theta(obj,x,u) %input matrix of the discrete time linearized dynamics
            import casadi.*
            x_fun = SX.sym('x',obj.nx);
            u_fun = SX.sym('u',obj.nu);
            var_fun = [x_fun;u_fun];
            B = jacobian(obj.ddyn_theta(x_fun,u_fun), u_fun);
            B_fun = casadi.Function('B_fun',{var_fun},{B});
            B = B_fun([x;u]);
        end
         
        function [max_mu] = compute_mu(obj,n_points)
            % estimation of the value of mu, via sampling
            rng(0,'twister');
            nx = obj.nx;
            nu = obj.nu;
            tic
            M = [ones(1,6), ones(1,2)*obj.T_max];
            max_mu = zeros(1,nx);
            parfor i = 1:n_points % assume symmetrical constraints
            %for i = 1:n_points % slower alternative if parfor is not available
                eval = M.*(2*rand(1,nx+nu)-1);
                max_mu = max(max_mu,obj.eval_mu(eval));
            end
            toc;            
        end
        
        function mu = eval_mu(obj,xu)
            import casadi.*
            x_fun = SX.sym('x',obj.nx);
            u_fun = SX.sym('u',obj.nu);
            var_fun = [x_fun;u_fun];
            % ! Assume f_theta is linear ! todo: add hessian of f_theta
            H = jacobian(jacobian(obj.ddyn(x_fun,u_fun,'multi'), var_fun), var_fun); 
            H_fun = casadi.Function('H_fun',{var_fun},{H});
            H = permute(reshape(full(H_fun(xu)),[obj.nx,obj.nx+obj.nu,obj.nx+obj.nu]),[3,2,1]);
            d = size(H);
            for i = 1:d(3)
                mu(i) = 0.5*sum(sum(abs(H(:,:,i))));
            end
        end
           
        
    end
end

