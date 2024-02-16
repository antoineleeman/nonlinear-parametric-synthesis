% -----------------------------------------------------------------------------
% File: main.m
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

%%
clear all;
close all;
clc;

import casadi.*

m = Capture_Stabilization();
N = m.N; % number of control intervals
gamma_max = 0.2;

nx = m.nx;
nu = m.nu;
ni = m.ni; % number of constraints
nw = m.nw; % size of the noise matrix E
C = kron(eye(m.N),eye(nx));
D = kron(eye(m.N),eye(nu));

sls = SLS(N,nx,nu);

Z = MX.sym('state',nx,N+1);
V = MX.sym('input',nu,N+1);

Phi_x = MX.sym('state_response',nx,nx,N*(N+1)/2);
Phi_u = MX.sym('input_response',nu,nx,N*(N+1)/2);

Sigma = cellmat(1,N*(N-1)/2,nx,nx);
d = MX.sym('diag_Sigma',N*nx);
tau = MX.sym('lin_tube',N-1);

n_eq = 0;
n_ineq = 0;

g_eq = [];
g_ineq = [];

var_slack = [];

f = sls.getObjectiveNominal(m,Z,V);

[n_dyn,g_dyn] = sls.getConstraintsDynamic(m,Z,V);
n_eq = n_eq + n_dyn;
g_eq = [g_eq ; g_dyn];

[n_cons,g_cons, slack_cons, ~] = sls.getLinearConstraints(m,Z,V, cellmat(N*(N+1)/2,nx,nx),cellmat(N*(N+1)/2,nx,nu));
n_ineq = n_ineq + n_cons;
g_ineq = [g_ineq ; g_cons];
var_slack = [var_slack;slack_cons];

[y_nom, n_y_nom] = sls.getVariablesNominal(Z,V);
y = [y_nom;slack_cons];

nlp = struct('x',y, 'f',f, 'g',[g_eq;g_ineq]);
solver = nlpsol('solver', 'ipopt', nlp);
lbg = [zeros(n_eq,1);-inf(n_ineq,1)];
ubg = zeros(n_eq +n_ineq,1 );

res = solver('lbg',lbg,'ubg',ubg);
x0_nom = full(res.x);
v_sol_init = full(res.x((N+1)*nx+1:n_y_nom));
v_init = sls.vecToNominalInput(m,v_sol_init);

g_ineq = [];
n_ineq = 0;
y = y_nom;
n=0;
var_slack = [];

[n_cons,g_cons, slack_cons, n_slack_cons] = sls.getLinearConstraints(m,Z,V, Phi_x,Phi_u);
n_ineq = n_ineq + n_cons;
g_ineq = [g_ineq ; g_cons];
var_slack = [var_slack;slack_cons];

[n_filter,g_filter,n_slack_filter, slack_filter] = sls.getConstraintFilter(m,Z,V,Phi_x,Phi_u,d,tau);
n_ineq = n_ineq + n_filter;
g_ineq = [g_ineq ; g_filter];
var_slack = [var_slack;slack_filter];


[n_map, g_map] = sls.getNonlinearMapConstraints(m, Z,V, Phi_x,Phi_u, Sigma, d);
n_eq = n_eq + n_map;
g_eq = [g_eq ; g_map];

[n_ineq_tube,g_ineq_tube, n_slack_tube, slack_tube] = sls.getConstraintTube(m,tau, Phi_x, Phi_u);
n_ineq = n_ineq + n_ineq_tube;
g_ineq = [g_ineq ; g_ineq_tube];
var_slack = [var_slack;slack_tube];

[y_contr,n_y_contr] = sls.getVariablesResponses(m,Phi_x,Phi_u);
[y_tube, n_y_tube] = sls.getVariablesTube(tau);
[y_filter, n_y_filter] = sls.getVariablesFilter_onlydiag(m, d);

[ inf_norm, n_ineq_inf,g_ineq_inf, var_slack_inf,n_var_slack_inf ] = sls.mat_inf_norm([C * sls.v3_to_R(Phi_x);D*sls.v3_to_M(Phi_u)]);
n_ineq = n_ineq + n_ineq_inf +1;
g_ineq = [g_ineq ; g_ineq_inf;[inf_norm-gamma_max]; ];
var_slack = [var_slack; var_slack_inf;inf_norm];


y = [y; y_contr;y_tube;y_filter;var_slack];
alpha = 1e-6;

f = f+ alpha*y'*y;
nlp = struct('x',y, 'f',f, 'g',[g_eq;g_ineq]);
solver = nlpsol('solver', 'ipopt', nlp);
lbg = [zeros(n_eq,1);-inf(n_ineq,1)];
ubg = zeros(n_eq +n_ineq,1 );

x0 = zeros(n_slack_cons+n_slack_tube+ n_slack_filter+n_var_slack_inf+n_y_filter+ n_y_tube+n_y_contr + n_y_nom,1);
res = solver('x0',x0,'lbg',lbg,'ubg',ubg);
n=0;
z_sol_v = full(res.x(n+1:(N+1)*nx));
v_sol_v = full(res.x((N+1)*nx+1:n_y_nom));
n = n+n_y_nom;
Phi_sol_v = res.x(n+1 : n +n_y_contr);
n = n+n_y_contr;
tube_v = full(res.x(n+1 : n+n_y_tube));
n = n+n_y_tube;
filter_v = full(res.x(n+1 : n+n_y_filter));

z_sol = sls.vecToNominalState(m,z_sol_v);
v_sol = sls.vecToNominalInput(m,v_sol_v);
[Phi_x_sol,Phi_u_sol] = sls.vecToResponse(m,Phi_sol_v);
[Sigma_sol_v, d_sol] = sls.vecToSigma(m,filter_v);

R_sol = full(sls.v3_to_R(Phi_x_sol));
M_sol = full(sls.v3_to_M(Phi_u_sol));

Sigma_sol = full(sls.v3_to_Sigma(Sigma_sol_v,d_sol));

% save('CDC_paper_Leeman_2023')

%%
% clear all;
% close all;
% clc;
% load('CDC_paper_Leeman_2023.mat')

I = eye(nx);
tubes_x = reshape(vecnorm( kron(eye(m.N),I ) * R_sol,1,2), [nx,m.N]);
tubes_x = [zeros(nx,1), tubes_x];

I = eye(nu);
tubes_u = reshape(vecnorm( kron(eye(m.N),I) * M_sol,1,2), [nu,m.N]);
tubes_u = [zeros(nu,1), tubes_u];

K = M_sol*inv(R_sol);
theta_true = 0.01;

x_cl = zeros(nx,m.N+1);
u_cl = zeros(nu,m.N+1);

x_cl(:,1) = m.x0;
for k = 1 : m.N
    Delta_x = reshape(x_cl(:,2:end) -z_sol(:,2:end), [N*nx,1]);
    u_cl = reshape([zeros(nu,1);K*Delta_x] + v_sol_v,[nu,N+1]);
    x_cl(:,k+1)= m.ddyn(x_cl(:,k),u_cl(:,k)) + m.ddyn_theta(x_cl(:,k),u_cl(:,k))*theta_true + m.E*(2*(rand(m.nw,1)>0.5)-1);
end

x_cl_init = zeros(nx,m.N+1);
u_cl_init = zeros(nu,m.N+1);

x_cl_init(:,1) = m.x0;
for k = 1: m.N
    u_cl_init(:,k) = v_init(:,k);
    x_cl_init(:,k+1)= m.ddyn(x_cl_init(:,k),u_cl_init(:,k)) + m.ddyn_theta(x_cl_init(:,k),u_cl_init(:,k))*theta_true + m.E*(2*(ones(m.nw,1))-1);
end

%%
figure(1);
clf
colormap = [0.0504    0.0298    0.5280
    0.4934    0.0115    0.6580
    0.7964    0.2780    0.4713
    0.9722    0.5817    0.2541
    0.9400    0.9752    0.1313];

color1 = colormap(1,:);
color2 = colormap(3,:);
color3 = colormap(2,:);
color4 = colormap(4,:);

labelFontSize = 6;
tickFontSize = 8;
legendfontsize = 10;
alpha_line = 0.9;
alpha_area = 0.5;

highlightColor = [hex2dec('7F') hex2dec('00') hex2dec('FF')] / 255;
highlightAlpha = 0.2;

subplot(1,4,1);
hold on;

upper_y = z_sol(1,:)+tubes_x(1,:);
lower_y = z_sol(1,:)-tubes_x(1,:);
%plot(0:m.dt:m.T, upper_y,'Color',color1, 'LineWidth', 1, 'MarkerSize', 10);
%plot(0:m.dt:m.T, lower_y, 'Color',color1, 'LineWidth', 1, 'MarkerSize', 10);
h_rx = fill([0:m.dt:m.T, fliplr(0:m.dt:m.T)], [upper_y, fliplr(lower_y)],[0.8, 0.8, 0.8], 'FaceColor',color1, 'EdgeColor', 'none', 'FaceAlpha', alpha_area);

upper_y = z_sol(2,:)+tubes_x(2,:);
lower_y = z_sol(2,:)-tubes_x(2,:);
%plot(0:m.dt:m.T, upper_y, 'Color',color2, 'LineWidth', 1, 'MarkerSize', 10);
%plot(0:m.dt:m.T, lower_y,  'Color',color2, 'LineWidth', 1, 'MarkerSize', 10);
h_ry = fill([0:m.dt:m.T, fliplr(0:m.dt:m.T)], [upper_y, fliplr(lower_y)], [0.8, 0.8, 0.8],'FaceColor',color2, 'EdgeColor', 'none', 'FaceAlpha', alpha_area);

plot(0:m.dt:m.T, x_cl(2,:),'k--', 'LineWidth', 1.1, 'MarkerSize', 10);
plot(0:m.dt:m.T, x_cl(1,:),'k--', 'LineWidth', 1.1, 'MarkerSize', 10);
plot(0:m.dt:m.T, x_cl_init(2,:),'k:', 'LineWidth', 1.1, 'MarkerSize', 10);
plot(0:m.dt:m.T, x_cl_init(1,:),'k:', 'LineWidth', 1.1, 'MarkerSize', 10);

h_x = plot(0:m.dt:m.T, z_sol(1,:),'color',color1,'LineStyle','-', 'LineWidth', 1.5, 'MarkerSize', 10);
h_y = plot(0:m.dt:m.T, z_sol(2,:),'color',color2,'LineStyle','-', 'LineWidth', 1.5, 'MarkerSize', 10);


yline(1, '-k', 'LineWidth', 1.5);
yline(-1, '-k', 'LineWidth', 1.5);
lgd1 = legend([h_x, h_y, h_rx, h_ry], {'$p_x^\star$', '$p_y^\star$','$\mathcal{R}_{p_x^\star}$','$\mathcal{R}_{p_y^\star}$'}, 'Interpreter', 'latex', 'Location', 'best','fontsize',legendfontsize,'Position',[0.1712 0.1888 0.0763 0.2338]);
set(lgd1, 'Box', 'off', 'Color', 'none');

xlim([0, 5]);
ylim([-1, 1.1]);
ylabel('states', 'FontSize', labelFontSize);
xlabel('time [-]', 'FontSize', labelFontSize);
set(gca, 'FontSize', tickFontSize);
grid on;

subplot(1,4,2);
hold on;

upper_y = z_sol(3,:)+tubes_x(3,:);
lower_y = z_sol(3,:)-tubes_x(3,:);
%plot(0:m.dt:m.T, upper_y,'Color',color1, 'LineWidth', 1, 'MarkerSize', 10);
%plot(0:m.dt:m.T, lower_y, 'Color',color1, 'LineWidth', 1, 'MarkerSize', 10);
rx = fill([0:m.dt:m.T, fliplr(0:m.dt:m.T)], [upper_y, fliplr(lower_y)],[0.8, 0.8, 0.8], 'FaceColor',color1, 'EdgeColor', 'none', 'FaceAlpha', 0.5);

upper_y = z_sol(6,:)+tubes_x(6,:);
lower_y = z_sol(6,:)-tubes_x(6,:);
%plot(0:m.dt:m.T, upper_y,'Color',color2, 'LineWidth', 1, 'MarkerSize', 10);
%plot(0:m.dt:m.T, lower_y, 'Color',color2, 'LineWidth', 1, 'MarkerSize', 10);
ry = fill([0:m.dt:m.T, fliplr(0:m.dt:m.T)], [upper_y, fliplr(lower_y)],[0.8, 0.8, 0.8], 'FaceColor',color2, 'EdgeColor', 'none', 'FaceAlpha', 0.5);


h_x = plot(0:m.dt:m.T, z_sol(3,:),'color',color1,'LineStyle','-', 'LineWidth', 1.5, 'MarkerSize', 10);
h_y = plot(0:m.dt:m.T, z_sol(6,:),'color',color2,'LineStyle','-', 'LineWidth', 1.5, 'MarkerSize', 10);
plot(0:m.dt:m.T, x_cl(3,:),'k--','LineWidth', 1.1, 'MarkerSize', 10);
plot(0:m.dt:m.T, x_cl(6,:),'k--', 'LineWidth', 1.1, 'MarkerSize', 10);
plot(0:m.dt:m.T, x_cl_init(3,:),'k:','LineWidth', 1.1, 'MarkerSize', 10);
plot(0:m.dt:m.T, x_cl_init(6,:),'k:', 'LineWidth', 1.1, 'MarkerSize', 10);

yline(1, '-k', 'LineWidth', 1.5);
yline(-1, '-k', 'LineWidth', 1.5);
lgd2 = legend([h_x, h_y, rx, ry], {'$\psi^\star$', '$\dot \psi^\star$','$\mathcal{R}_{\psi^\star}$','$\mathcal{R}_{\dot \psi^\star}$'}, 'Interpreter', 'latex', 'Location', 'best','fontsize',legendfontsize,'Position',[0.3925 0.6182 0.0777 0.2379]);
set(lgd2, 'Box', 'off', 'Color', 'none');
xlim([0, 5]);
ylim([-1, 1.1]);
xlabel('time [-]', 'FontSize', labelFontSize);
set(gca,'YTickLabel', [], 'FontSize', tickFontSize);

grid on;
subplot(1,4,3);

hold on
grid on;
upper_y = z_sol(4,:)+tubes_x(4,:);
lower_y = z_sol(4,:)-tubes_x(4,:);
%plot(0:m.dt:m.T, upper_y,'Color',color1, 'LineWidth', 1, 'MarkerSize', 10);
%plot(0:m.dt:m.T, lower_y, 'Color',color1, 'LineWidth', 1, 'MarkerSize', 10);
rx = fill([0:m.dt:m.T, fliplr(0:m.dt:m.T)], [upper_y, fliplr(lower_y)],[0.8, 0.8, 0.8], 'FaceColor',color1, 'EdgeColor', 'none', 'FaceAlpha', 0.5);

upper_y = z_sol(5,:)+tubes_x(5,:);
lower_y = z_sol(5,:)-tubes_x(5,:);
%plot(0:m.dt:m.T, upper_y,'Color',color2, 'LineWidth', 1, 'MarkerSize', 10);
%plot(0:m.dt:m.T, lower_y, 'Color',color2, 'LineWidth', 1, 'MarkerSize', 10);
ry = fill([0:m.dt:m.T, fliplr(0:m.dt:m.T)], [upper_y, fliplr(lower_y)],[0.8, 0.8, 0.8], 'FaceColor',color2, 'EdgeColor', 'none', 'FaceAlpha', 0.5);


plot(0:m.dt:m.T, x_cl(4,:),'k--', 'LineWidth', 1.5, 'MarkerSize', 10);
plot(0:m.dt:m.T, x_cl(5,:),'k--','LineWidth', 1.5, 'MarkerSize', 10);
plot(0:m.dt:m.T, x_cl_init(4,:),'k:', 'LineWidth', 1.5, 'MarkerSize', 10);
plot(0:m.dt:m.T, x_cl_init(5,:),'k:','LineWidth', 1.5, 'MarkerSize', 10);

h_x = plot(0:m.dt:m.T, z_sol(4,:),'color',color1,'LineStyle','-', 'LineWidth', 1.1, 'MarkerSize', 10);
h_y = plot(0:m.dt:m.T, z_sol(5,:),'color',color2,'LineStyle','-', 'LineWidth', 1.1, 'MarkerSize', 10);

yline(1, '-k', 'LineWidth', 1.5);
yline(-1, '-k', 'LineWidth', 1.5);
xlim([0, 5]);
ylim([-1, 1.1]);
lgd3 = legend([h_x, h_y, rx, ry], {'${\dot p_x}^\star$', '${\dot p_y}^\star$', '$\mathcal{R}_{{\dot p_x}^\star}$', '$\mathcal{R}_{{\dot p_y}^\star}$'}, 'Interpreter', 'latex', 'Location', 'best','fontsize',legendfontsize,'Position',[0.5982 0.5970 0.0763 0.2338]);
set(lgd3, 'Box', 'off', 'Color', 'none');
xlabel('time [-]', 'FontSize', labelFontSize);
set(gca,'YTickLabel', [], 'FontSize', tickFontSize);


subplot(1,4,4);
hold on
grid on;

upper_y = v_sol(1,:)+tubes_u(1,:);
lower_y = v_sol(1,:)-tubes_u(1,:);
for i = 1:length(upper_y)-1
    rectangle('Position', [(i-1)*m.dt, lower_y(i), m.dt, upper_y(i)-lower_y(i)], 'FaceColor', [color3, 0.5], 'EdgeColor', 'none');
end

upper_y = v_sol(2,:)+tubes_u(2,:);
lower_y = v_sol(2,:)-tubes_u(2,:);
for i = 1:length(upper_y)-1
    rectangle('Position', [(i-1)*m.dt, lower_y(i), m.dt, upper_y(i)-lower_y(i)], 'FaceColor', [color4, 0.5], 'EdgeColor', 'none');
end

upper_y = v_sol(1,:)+tubes_u(1,:);
lower_y = v_sol(1,:)-tubes_u(1,:);
%stairs(0:m.dt:m.T, [upper_y(1:end-1),upper_y(end-1)],'Color',[color3,alpha_line], 'LineWidth', 1, 'MarkerSize', 10);
%stairs(0:m.dt:m.T, [lower_y(1:end-1),lower_y(end-1)], 'Color',[color3,alpha_line], 'LineWidth', 1, 'MarkerSize', 10);

upper_y = v_sol(2,:)+tubes_u(2,:);
lower_y = v_sol(2,:)-tubes_u(2,:);
%stairs(0:m.dt:m.T, [upper_y(1:end-1),upper_y(end-1)],'Color',[color4,alpha_line], 'LineWidth', 1, 'MarkerSize', 10);
%stairs(0:m.dt:m.T, [lower_y(1:end-1),lower_y(end-1)], 'Color',[color4,alpha_line], 'LineWidth', 1, 'MarkerSize', 10);

stairs(0:m.dt:m.T, [u_cl(2,1:end-1), u_cl(2,end-1)],'k--','LineWidth', 1.1, 'MarkerSize', 10);
stairs(0:m.dt:m.T, [u_cl(1,1:end-1), u_cl(1,end-1)],'k--','LineWidth', 1.1, 'MarkerSize', 10);
stairs(0:m.dt:m.T, [u_cl_init(2,1:end-1), u_cl_init(2,end-1)],'k:', 'LineWidth', 1.1, 'MarkerSize', 10);
stairs(0:m.dt:m.T, [u_cl_init(1,1:end-1), u_cl_init(1,end-1)],'k:','LineWidth', 1.1, 'MarkerSize', 10);

h_x = stairs(0:m.dt:m.T, [v_sol(1,1:end-1), v_sol(1,end-1)],'color',color3, 'LineWidth', 1.5, 'MarkerSize', 10);
h_y = stairs(0:m.dt:m.T, [v_sol(2,1:end-1), v_sol(2,end-1)],'color',color4, 'LineWidth', 1.5, 'MarkerSize', 10);

yline(m.T_max, '-k', 'LineWidth', 1.5);
yline(-m.T_max, '-k', 'LineWidth', 1.5);
xlim([0, 5]);
ylim([-1, 1.1]);
rx = fill(nan, nan,[0.8, 0.8, 0.8], 'FaceColor',color3, 'EdgeColor', 'none', 'FaceAlpha', 0.5);
ry = fill(nan, nan,[0.8, 0.8, 0.8], 'FaceColor',color4, 'EdgeColor', 'none', 'FaceAlpha', 0.5);


lgd4 = legend([h_x, h_y, rx, ry], {'$v_x^\star$', '$v_y^\star$','$\mathcal{R}_{v_x^\star}$','$\mathcal{R}_{v_y^\star}$'}, 'Interpreter', 'latex', 'Location', 'southeast','fontsize',legendfontsize, 'Position', [0.8313 0.1049 0.0762 0.2338]);
set(lgd4, 'Box', 'off', 'Color', 'none');
xlabel('time [-]', 'FontSize', labelFontSize);

ylabel('inputs', 'FontSize', labelFontSize);
set(gca, 'FontSize', tickFontSize);

ax = axes('Position', [0 0 1 1], 'Visible', 'off');
dummy_plot = nan(1, 3);
hold on;

dummy_plot(1) = plot(ax, nan,'k:','LineWidth', 1.1, 'MarkerSize', 10);
dummy_plot(2) = plot(ax, nan,'k--','LineWidth', 1.1, 'MarkerSize', 10);
dummy_plot(3) = plot(ax, nan, '-k', 'LineWidth', 1.5);

color_labels = { 'non-robust', 'sample','$\mathcal{C}$'};
hold off;
lgd = legend(ax, color_labels, 'Location', 'southeastoutside','Interpreter','latex','fontsize',legendfontsize,'Position',[0.9068 0.1746 0.1061 0.1402]);

set(lgd,'Box', 'off', 'Color', 'none');

width_cm = 25; % Width in centimeters
height_cm = 8; % Height in centimeters
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'Units', 'centimeters');
set(gcf, 'Position', [0 0 width_cm height_cm]);

filename = 'fig1.pdf';
exportgraphics(gcf, filename, 'ContentType', 'vector', 'Resolution', 1200);
saveas(gcf, 'fig1.png');
%%
figure(2);

custom_colors = [    0.0504    0.0298    0.5280
    0.6107    0.0902    0.6200
    0.9283    0.4730    0.3261
    0.9400    0.9752    0.1313
    ];

custom_colors(end,:) = [];
subplot_titles = {'$p_x^\star$', '$p_y^\star$', '$\psi^\star$', '${\dot p_x}^\star$', '${\dot p_y}^\star$', '$\dot \psi^\star$'};

labelFontSize = 10;
tickFontSize = 10;
B1 = abs(reshape( kron([0;tube_v].^2,m.mu'),[m.nx,m.N])); % contribution of lin error
B2 = abs(reshape( kron(ones(m.N,1),vecnorm(m.E,1,2)),[m.nx,m.N])); % contribution of add. noise
B3 = abs(reshape(d_sol,[m.nx,m.N])); % total tube

B_stack = cat(3,B2,B3-B2-B1,B1);
figure(3);
clf;

for i = 1:6
    s = subplot(1,6,i);
    b = bar(squeeze(B_stack(i,:,:)), 'stacked');
    hold on;
    for j = 1:size(custom_colors,1)
        b(j).FaceColor = custom_colors(j,:);
    end
    grid on;
    ylim([0.0005, 0.1]);
    
    xlabel('step $k$', 'FontSize', labelFontSize,'Interpreter','latex');
    title(subplot_titles{i},'Interpreter','latex', 'FontSize', labelFontSize);
end
color_labels = {'$\mathcal{W}$', '${\Theta}^\star$', '$\tau^{\star 2} \mu$'};
subplot(1,6,1);
ylabel('$\sigma_{i,k}^\star$', 'FontSize', labelFontSize,'Interpreter','latex');
    set(gca, 'YScale', 'log', 'XTickLabel', [],'FontSize', tickFontSize);
    
for i=2:6
    subplot(1,6,i);    
    set(gca, 'YTickLabel', [], 'XTickLabel', [],'YScale', 'log','FontSize', tickFontSize);
end
    
ax = axes('Position', [0 0 1 1], 'Visible', 'off');
dummy_bars = nan(1, numel(color_labels));
hold on;
for i = 1:numel(color_labels)
    dummy_bars(i) = bar(ax, nan, 'FaceColor', custom_colors(i, :));
end
hold off;
lgd = legend(ax, color_labels, 'Location', 'eastoutside','Interpreter','latex');
set(lgd, 'Box', 'off', 'Color', 'none');
ax.XAxis.Color = 'none';
ax.YAxis.Color = 'none';

width_cm = 25; % Width in centimeters
height_cm = 5; % Height in centimeters
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'Units', 'centimeters');
set(gcf, 'Position', [0 0 width_cm height_cm]);

filename = 'fig2.pdf';
exportgraphics(gcf, filename, 'ContentType', 'vector', 'Resolution', 1200);
