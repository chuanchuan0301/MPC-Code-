clear; clc; close all;
script_dir = fileparts(mfilename('fullpath'));
if ~isempty(script_dir), cd(script_dir); end

q_pos = 0.08;
q_rate = 0.08;
R_weight = 0.04;
kappa = 1.2;

N = 10;
T_sim = 50;
Ts = 0.17;

mu_vdp = [0.10; 0.18; 0.25];
rho = 1;

tau_base = 0.08;
xi_base = 0.005;
sigma_val = 0.005;
lambda_val = 0.95;
beta_et_val = 0.3;
gamma_et_val = 0.10;
alpha_i_val = 0.9;

epsilon_base = 0.1;
epsilon_xi = 0.25;
eps_factor_val = 0.01;
x_max_val = 3;
u_max_val = 3;

params.M = 3;
params.n = 2;
params.m = 1;
params.Ts = Ts;
params.N = N * ones(params.M, 1);
params.rho = rho;
params.mu = mu_vdp;
params.alpha_i = alpha_i_val;

%% 耦合 ====================
params.N_in = cell(params.M, 1);
params.N_in{1} = [2];
params.N_in{2} = [1, 3];
params.N_in{3} = [2];

params.N_out = cell(params.M, 1);
params.N_out{1} = [2];
params.N_out{2} = [1, 3];
params.N_out{3} = [2];

%% ==================== 离散时间线性化====================
for i = 1:params.M
    params.A{i} = [1, Ts; -Ts, 1 + mu_vdp(i) * Ts];
    params.B{i} = [0; Ts];
end

%% ==================== LQR / 终端权矩阵 ===================
params.Q = cell(params.M, 1);
params.R = cell(params.M, 1);
params.P = cell(params.M, 1);
params.K = cell(params.M, 1);
params.Q_star = cell(params.M, 1);
params.A_cl = cell(params.M, 1);

params.kappa = kappa;
fprintf('LQR / terminal matrices computed from (A,B,Q,R) and Lyapunov equation:\n');
for i = 1:params.M
    params.Q{i} = diag([q_pos, q_rate]);
    params.R{i} = R_weight * eye(params.m);
    [K_lqr, ~, ~] = dlqr(params.A{i}, params.B{i}, params.Q{i}, params.R{i});
    params.K{i} = -K_lqr;
    params.A_cl{i} = params.A{i} + params.B{i} * params.K{i};
    params.Q_star{i} = params.Q{i} + params.K{i}' * params.R{i} * params.K{i};
    params.P{i} = dlyap(params.A_cl{i}', params.kappa * params.Q_star{i});
    eig_cl = eig(params.A_cl{i});
    fprintf('  Subsystem %d (mu=%.2f): K = [%.4f, %.4f], |lambda|_max = %.4f (%s)\n', ...
        i, mu_vdp(i), params.K{i}(1), params.K{i}(2), max(abs(eig_cl)), ...
        iif(max(abs(eig_cl)) < 1, 'stable', 'UNSTABLE'));
    fprintf('    P = [%.4f %.4f; %.4f %.4f]\n', ...
        params.P{i}(1,1), params.P{i}(1,2), params.P{i}(2,1), params.P{i}(2,2));
end
fprintf('\n');

params.x_max = x_max_val * ones(params.M, 1);
[params.L_z, params.L_h, params.L_hij] = compute_lipschitz_constants(params);
fprintf('Lipschitz constants (max spectral norm of Jacobian on |x|<=%.1f):\n', x_max_val);
for i = 1:params.M
    fprintf('  L_z(%d)=%.4f, bar L_h(%d)=%.4f\n', i, params.L_z(i), i, params.L_h(i));
end
fprintf('  L_h12=%.4f, L_h21=%.4f, L_h23=%.4f, L_h32=%.4f\n\n', ...
    params.L_hij(1,2), params.L_hij(2,1), params.L_hij(2,3), params.L_hij(3,2));
print_KP_lipschitz_report(params);

%% ==================== Lipschitz 常数与触发参数 ====================
params.Gamma = [1; 2; 1];   

params.zeta = x_max_val * ones(params.M, 1);
params.u_max = u_max_val * ones(params.M, 1);
params.u_min = -u_max_val * ones(params.M, 1);
params.xi = xi_base * ones(params.M, 1);
params.lambda = lambda_val;
params.eps_factor = eps_factor_val;
params.tau = tau_base * ones(params.M, 1);
params.beta_et = beta_et_val * ones(params.M, 1);
params.gamma_et = gamma_et_val * ones(params.M, 1);
params.sigma = sigma_val;
params.epsilon = epsilon_base * ones(params.M, 1);
params.epsilon_xi = epsilon_xi;

params.lam_P_min = zeros(params.M, 1);
params.lam_P_max = zeros(params.M, 1);
params.chi = zeros(params.M, 1);
params.Xi_factor = zeros(params.M, 1);

for i = 1:params.M
    lam_Q_min = min(eig(params.Q{i}));
    lam_Q_max = max(eig(params.Q{i}));
    params.lam_P_min(i) = min(eig(params.P{i}));
    params.lam_P_max(i) = max(eig(params.P{i}));
    lam_Qstar_max = max(eig(params.Q_star{i}));
 

    params.chi(i) = 2 * params.zeta(i) * lam_Q_max / params.lam_P_min(i);


    params.Xi_factor(i) = lam_Q_min / params.lam_P_max(i) ...
        - params.gamma_et(i)^2 * lam_Qstar_max / params.lam_P_min(i);
    params.Xi_factor(i) = max(params.Xi_factor(i), 0);
end

fprintf('Triggering Lipschitz / threshold ingredients:\n');
for i = 1:params.M
    fprintf('  Sub %d: Lz=%.4f, Lh=%.4f, Gamma=%d, chi=%.4f, Xi_factor=%.4f\n', ...
        i, params.L_z(i), params.L_h(i), params.Gamma(i), ...
        params.chi(i), params.Xi_factor(i));
end
fprintf('\n');

%% ==================== 初始条件 ====================
x0 = {[1.8; 0]; [-1.3; 0]; [1.2; 0]};

fprintf('Initial States:\n');
for i = 1:params.M
    fprintf('  Subsystem %d: x0 = [%.2f, %.2f]^T, ||x0||_P = %.4f, Omega: ||x||_P <= %.4f\n', ...
        i, x0{i}(1), x0{i}(2), p_norm(x0{i}, params.P{i}), params.epsilon(i));
end
fprintf('\n');
disturbances = generate_disturbances(params, T_sim, 2024);



[x_et, u_et, trigger_et, conv_time_et, mpc_cpu_time_et, fb_cpu_time_et, ...
    trigger_reason_et, T_omega_et] = ...
    simulate_event_triggered_mpc(x0, T_sim, params, disturbances, true);

fprintf('  Dis-MPC triggers: [%d, %d, %d] / %d steps\n', ...
    length(trigger_et{1}), length(trigger_et{2}), length(trigger_et{3}), T_sim);
print_trigger_reasons(trigger_reason_et, params.M);

et_mpc_total = 0; et_fb_total = 0;
for i = 1:params.M
    et_mpc_total = et_mpc_total + sum(mpc_cpu_time_et{i});
    et_fb_total = et_fb_total + sum(fb_cpu_time_et{i});
end


[x_dist, u_dist, trigger_dist, conv_time_dist, mpc_cpu_time_dist, fb_cpu_time_dist, ...
    trigger_reason_dist, T_omega_dist] = ...
    simulate_event_triggered_mpc(x0, T_sim, params, disturbances, false);

fprintf('  Dec-MPC triggers: [%d, %d, %d] / %d steps\n', ...
    length(trigger_dist{1}), length(trigger_dist{2}), length(trigger_dist{3}), T_sim);
print_trigger_reasons(trigger_reason_dist, params.M);



dist_mpc_total = 0; dist_fb_total = 0;
for i = 1:params.M
    dist_mpc_total = dist_mpc_total + sum(mpc_cpu_time_dist{i});
    dist_fb_total = dist_fb_total + sum(fb_cpu_time_dist{i});
end


[RMSEx_et, Tbar_omega_et, Ju_et] = calculate_paper_metrics( ...
    x_et, u_et, T_omega_et, params, T_sim);
[RMSEx_dist, Tbar_omega_dist, Ju_dist] = calculate_paper_metrics( ...
    x_dist, u_dist, T_omega_dist, params, T_sim);

fprintf('  Tbar_Omega Dis-MPC=%.4f s, Dec-MPC=%.4f s\n', Tbar_omega_et, Tbar_omega_dist);
for i = 1:params.M
    fprintf('  Subsystem %d: Dis-MPC T_Omega=%.4f s (step %s), Dec-MPC T_Omega=%.4f s (step %s)\n', ...
        i, T_omega_et(i) * Ts, fmt_step(T_omega_et(i)), ...
        T_omega_dist(i) * Ts, fmt_step(T_omega_dist(i)));
end
fprintf('\n');




set(0, 'DefaultFigureVisible', 'on');
time_steps = 0:T_sim;
t_steps = 0:T_sim-1;

dist_colors = { ...
    [0.000, 0.447, 0.698], ...
    [0.800, 0.200, 0.200], ...
    [0.000, 0.600, 0.300]};
decent_colors = { ...
    [0.900, 0.600, 0.000], ...
    [0.400, 0.760, 0.200], ...
    [0.700, 0.400, 0.800]};

fs_tick   = 15;
fs_label  = 17;
left = 0.14; width = 0.24; gap = 0.05; bottom = 0.15; height = 0.75;


fprintf('Generating plots...\n');

fig1 = figure('Name', 'State_x1_Comparison', 'NumberTitle', 'off', ...
    'Position', [200, 200, 900, 500], 'Color', 'w', 'Visible', 'on');

for idx = 1:3
    ax = subplot(1, 3, idx);
    set(ax, 'Position', [left + (idx - 1) * (width + gap), bottom, width, height]);
    hold on; grid on; box on;
    [tx, y_dis] = smooth_series(time_steps, x_et{idx}(1, :));
    [~, y_dec] = smooth_series(time_steps, x_dist{idx}(1, :));
    plot(tx, y_dis, '-', 'Color', dist_colors{idx}, ...
        'LineWidth', 2.5, 'DisplayName', 'Distributed-MPC');
    plot(tx, y_dec, '-', 'Color', decent_colors{idx}, ...
        'LineWidth', 2.5, 'DisplayName', 'Decentralized-MPC');
    xlabel('Step', 'FontSize', 17);
    set(gca, 'FontSize', 15, 'GridLineStyle', ':', 'GridAlpha', 0.4);
    legend('FontSize', 12, 'Location', 'best');
    xlim([0, T_sim]);
end
han = axes(fig1, 'visible', 'off');
han.YLabel.Visible = 'on';
ylabel(han, 'State $x_1$', 'Interpreter', 'latex', 'FontSize', 17);


fig2 = figure('Name', 'State_x2_Comparison', 'NumberTitle', 'off', ...
    'Position', [200, 200, 900, 500], 'Color', 'w', 'Visible', 'on');

for idx = 1:params.M
    ax = subplot(1, 3, idx);
    set(ax, 'Position', [left + (idx - 1) * (width + gap), bottom, width, height]);
    hold on; grid on; box on;
    [tx, y_dis] = smooth_series(time_steps, x_et{idx}(2, :));
    [~, y_dec] = smooth_series(time_steps, x_dist{idx}(2, :));
    plot(tx, y_dis, '-', 'Color', dist_colors{idx}, ...
        'LineWidth', 2.5, 'DisplayName', 'Distributed-MPC');
    plot(tx, y_dec, '-', 'Color', decent_colors{idx}, ...
        'LineWidth', 2.5, 'DisplayName', 'Decentralized-MPC');
    xlabel('Step', 'FontSize', fs_label);
    set(gca, 'FontSize', fs_tick, 'GridLineStyle', ':', 'GridAlpha', 0.4);
    legend('FontSize', 12, 'Location', 'best');
    xlim([0, T_sim]);
end
han2 = axes(fig2, 'visible', 'off');
han2.YLabel.Visible = 'on';
ylabel(han2, 'State $x_2$', 'Interpreter', 'latex', 'FontSize', fs_label);


fig3 = figure('Name', 'ControlInput_Comparison', 'NumberTitle', 'off', ...
    'Position', [200, 200, 900, 500], 'Color', 'w', 'Visible', 'on');

for idx = 1:params.M
    ax = subplot(1, 3, idx);
    set(ax, 'Position', [left + (idx - 1) * (width + gap), bottom, width, height]);
    hold on; grid on; box on;
    u_et_plot = u_et{idx};     if size(u_et_plot, 1) > 1, u_et_plot = u_et_plot'; end
    u_dist_plot = u_dist{idx}; if size(u_dist_plot, 1) > 1, u_dist_plot = u_dist_plot'; end
    plot(t_steps, u_et_plot, '-', 'Color', dist_colors{idx}, ...
        'LineWidth', 2.0, 'DisplayName', 'Distributed-MPC');
    plot(t_steps, u_dist_plot, '-', 'Color', decent_colors{idx}, ...
        'LineWidth', 2.0, 'DisplayName', 'Decentralized-MPC');
    yline(0, ':', 'Color', [0.5 0.5 0.5], 'HandleVisibility', 'off');
    xlabel('Step', 'FontSize', fs_label);
    set(gca, 'FontSize', fs_tick, 'GridLineStyle', ':', 'GridAlpha', 0.4);
    legend('FontSize', 12, 'Location', 'best');
    xlim([0, T_sim - 1]);
    um = max([max(abs(u_et_plot)), max(abs(u_dist_plot)), 0.15]);
    ylim([-1.15 * um, 1.15 * um]);
end
han3 = axes(fig3, 'visible', 'off');
han3.YLabel.Visible = 'on';
ylabel(han3, 'Control input $u$', 'Interpreter', 'latex', 'FontSize', fs_label);


fig6 = figure('Name', 'Trigger_Instants_Comparison', 'NumberTitle', 'off', ...
    'Position', [200, 200, 900, 500], 'Color', 'w', 'Visible', 'on');
c_dis = [0.000, 0.447, 0.698];
c_dec = [0.835, 0.369, 0.000];
left_m = 0.13; right_m = 0.04; top_m = 0.04; bot_m = 0.14;
ax_w = 1 - left_m - right_m;
gap_v = 0.02;
ax_h = (1 - top_m - bot_m - (params.M - 1) * gap_v) / params.M;
for si = 1:params.M
    ax = axes('Position', [left_m, bot_m + (params.M - si) * (ax_h + gap_v), ax_w, ax_h], ...
        'Parent', fig6);
    hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
    set(ax, 'GridLineStyle', ':', 'GridAlpha', 0.25);
    if ~isempty(trigger_et{si})
        stem(ax, trigger_et{si}, 0.72 * ones(size(trigger_et{si})), ...
            'o', 'Color', c_dis, 'MarkerSize', 5, 'MarkerFaceColor', c_dis, ...
            'LineWidth', 1.2, 'DisplayName', 'Distributed ET-MPC');
    end
    if ~isempty(trigger_dist{si})
        stem(ax, trigger_dist{si}, 0.28 * ones(size(trigger_dist{si})), ...
            'd', 'Color', c_dec, 'MarkerSize', 5, 'MarkerFaceColor', c_dec, ...
            'LineWidth', 1.2, 'DisplayName', 'Decentralized ET-MPC');
    end
    xlim(ax, [0, T_sim]); ylim(ax, [0, 1]);
    set(ax, 'YTick', [0.28, 0.72], 'YTickLabel', {}, ...
        'FontSize', fs_tick, 'TickDir', 'in', 'LineWidth', 0.8);
    if si == params.M
        xlabel(ax, ' Step ', 'FontSize', fs_label, 'Interpreter', 'latex');
    else
        set(ax, 'XTickLabel', []);
    end
    ylabel(ax, sprintf('$k_%d^r$', si), 'FontSize', fs_label, 'Interpreter', 'latex');
    text(ax, T_sim * 0.985, 0.90, sprintf('Subsystem %d', si), ...
        'HorizontalAlignment', 'right', 'FontSize', fs_tick - 1, 'Color', [0 0 0]);
    if si == 1
        legend(ax, 'Location', 'northeast', 'FontSize', 14, 'EdgeColor', [0.75 0.75 0.75]);
    end
end
drawnow;



function [x_traj, u_traj, trigger_times, conv_time, mpc_cpu_time, fb_cpu_time, ...
          trigger_reason, T_omega] = ...
    simulate_event_triggered_mpc(x0, T_sim, params, disturbances, is_distributed)


    M = params.M;
    n = params.n;
    m = params.m;

    x_traj = cell(M, 1);
    u_traj = cell(M, 1);
    trigger_times = cell(M, 1);
    trigger_reason = cell(M, 1);
    conv_time = cell(M, 1);
    mpc_cpu_time = cell(M, 1);
    fb_cpu_time = cell(M, 1);
    u_opt_seq = cell(M, 1);
    x_star = cell(M, 1);
    k_r = -ones(M, 1);
    T_omega = T_sim * ones(M, 1);

    for i = 1:M
        Ni = params.N(i);
        x_traj{i} = zeros(n, T_sim + 1);
        x_traj{i}(:, 1) = x0{i};
        u_traj{i} = zeros(m, T_sim);
        trigger_times{i} = [];
        trigger_reason{i} = [];
        conv_time{i} = [];
        mpc_cpu_time{i} = [];
        fb_cpu_time{i} = [];
        u_opt_seq{i} = zeros(m, Ni);
        x_star{i} = repmat(x0{i}, 1, Ni + 1);
    end

    for k = 0:T_sim-1
        x_curr = cell(M, 1);
        in_omega = false(M, 1);
        for i = 1:M
            x_curr{i} = x_traj{i}(:, k + 1);
            in_omega(i) = (p_norm(x_curr{i}, params.P{i}) <= params.epsilon(i));
            if in_omega(i) && T_omega(i) == T_sim
                T_omega(i) = k;
                conv_time{i} = k * params.Ts;
            end
        end

        triggered = false(M, 1);
        reason = zeros(M, 1);
        for i = 1:M
            if in_omega(i)
                continue;
            end
            [triggered(i), reason(i)] = check_trigger(k, i, x_curr{i}, x_star{i}, ...
                k_r(i), params, x_traj{i});
        end

        for i = 1:M
            if ~triggered(i)
                continue;
            end
            trigger_times{i} = [trigger_times{i}, k];
            trigger_reason{i} = [trigger_reason{i}, reason(i)];

            xa = cell(M, 1);
            if is_distributed
                for j = params.N_in{i}
                    xa{j} = construct_assumed_state(j, k, params.N(i), params, x_star, k_r);
                end
            end
            x_bar = [];
            if k_r(i) >= 0
                x_bar = construct_shifted_prev(x_star{i}, k_r(i), k, params.N(i), i, params);
            end

            tic;
            [u_opt_seq{i}, x_star{i}] = solve_mpc(x_curr{i}, i, params.N(i), ...
                params, xa, x_bar, is_distributed, u_opt_seq{i});
            mpc_cpu_time{i} = [mpc_cpu_time{i}, toc];
            k_r(i) = k;
        end


        for i = 1:M
            if in_omega(i)
                tic;
                u_traj{i}(:, k + 1) = params.K{i} * x_curr{i};
                fb_cpu_time{i} = [fb_cpu_time{i}, toc];
            else
                l_curr = k - k_r(i);
                if k_r(i) >= 0 && l_curr >= 0 && l_curr < size(u_opt_seq{i}, 2)
                    u_traj{i}(:, k + 1) = u_opt_seq{i}(:, l_curr + 1);
                else
                    tic;
                    u_traj{i}(:, k + 1) = params.K{i} * x_curr{i};
                    fb_cpu_time{i} = [fb_cpu_time{i}, toc];
                end
            end
            u_traj{i}(:, k + 1) = max(params.u_min(i), ...
                min(params.u_max(i), u_traj{i}(:, k + 1)));
        end


        % 分布式耦合系数 params.rho；分散式耦合系数 0.1
        x_nb = x_curr;
        rho_plant = params.rho;
        if ~is_distributed
            rho_plant = 0.1;
        end
        for i = 1:M
            x_next = subsystem_dynamics(x_curr{i}, u_traj{i}(:, k + 1), i, ...
                x_nb, params, true, rho_plant);
            x_traj{i}(:, k + 2) = x_next + disturbances{i}(:, k + 1);
        end
    end
end

function [triggered, reason] = check_trigger(k, i, x_curr, x_star_i, k_r_i, params, x_hist)


    triggered = false;
    reason = 0;
    Ni = params.N(i);

    if k == 0
        triggered = true;
        reason = 1;
        return;
    end
    if k_r_i < 0
        triggered = true;
        reason = 1;
        return;
    end

    eta = k - k_r_i;
    if eta >= Ni
        triggered = true;
        reason = 3;
        return;
    end

    x_pred = x_star_i(:, min(eta + 1, size(x_star_i, 2)));
    e_now = p_norm(x_curr - x_pred, params.P{i});
    delta_i = compute_threshold(k, params, i);
    if e_now >= delta_i
        triggered = true;
        reason = 2;
        return;
    end


    acc = 0;
    nstar = size(x_star_i, 2);
    for s = 1:eta
        xs = x_hist(:, min(k_r_i + s + 1, size(x_hist, 2)));
        xp = x_star_i(:, min(s + 1, nstar));
        acc = acc + p_norm(xs - xp, params.P{i});
    end
    Xi_i = params.epsilon_xi^2 * max(eta, 1) * params.Xi_factor(i);
    if Xi_i > 0 && params.chi(i) * acc > Xi_i
        triggered = true;
        reason = 4;
    end
end

function delta = compute_threshold(k, params, i)


    sqmin = sqrt(params.lam_P_min(i));
    sqmax = sqrt(params.lam_P_max(i));
    tau_j = params.tau(i);
    num = sqmin * (params.tau(i) - params.xi(i)) ...
        - sqmax * params.Gamma(i) * params.L_h(i) ...
        * (tau_j + params.N(i) * params.sigma);
    den = sqmax * params.L_z(i) + 1e-12;
    adaptive = 1 - params.beta_et(i) * exp(-params.lambda * k);
    delta = max(1e-4, adaptive * num / den);
end

function xa = construct_assumed_state(j, k, N_i, params, x_star, k_r)


    n = params.n;
    xa = zeros(n, N_i + 1);
    Nj = params.N(j);
    if k_r(j) < 0 || isempty(x_star{j})
        xa = repmat(x_star{j}(:, 1), 1, N_i + 1);
        return;
    end
    shift = k - k_r(j);
    for l = 0:N_i
        src = shift + l + 1;
        if src >= 1 && src <= size(x_star{j}, 2) && l <= Nj - shift
            xa(:, l + 1) = x_star{j}(:, src);
        else
            if l == 0
                xa(:, 1) = x_star{j}(:, end);
            else
                u_fb = saturate_u(params.K{j} * xa(:, l), params, j);
                xa(:, l + 1) = subsystem_dynamics(xa(:, l), u_fb, j, {}, params, false);
            end
        end
    end
end

function x_bar = construct_shifted_prev(x_star_prev, k_r_prev, k, Ni, i, params)


    n = params.n;
    x_bar = zeros(n, Ni + 1);
    eta = k - k_r_prev;
    for l = 0:Ni
        src = eta + l + 1;
        if l <= Ni - eta && src >= 1 && src <= size(x_star_prev, 2)
            x_bar(:, l + 1) = x_star_prev(:, src);
        else
            if l == 0
                x_bar(:, 1) = x_star_prev(:, end);
            else
                u_fb = saturate_u(params.K{i} * x_bar(:, l), params, i);
                x_bar(:, l + 1) = subsystem_dynamics(x_bar(:, l), u_fb, i, {}, params, false);
            end
        end
    end
end

function [u_opt, x_opt] = solve_mpc(x0, i, Ni, params, xa, x_bar, is_distributed, u_prev)
    m = params.m;
    u0 = repmat(params.K{i} * x0, Ni, 1);
    if nargin >= 8 && ~isempty(u_prev) && numel(u_prev) == m * Ni
        up = reshape(u_prev, m, Ni);
        u_tail = saturate_u(params.K{i} * x0, params, i);
        u0 = [up(:, 2:end), u_tail];
        u0 = u0(:);
    end
    lb = repmat(params.u_min(i), Ni, 1);
    ub = repmat(params.u_max(i), Ni, 1);
    opts = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
        'MaxIterations', 60, 'MaxFunctionEvaluations', 1200, ...
        'OptimalityTolerance', 1e-4, 'ConstraintTolerance', 1e-4);

    obj = @(u) mpc_cost(u, x0, i, Ni, params, xa, is_distributed);


    nlc_relax = @(u) mpc_constraints(u, x0, i, Ni, params, xa, x_bar, is_distributed, false);
    nlc_full = @(u) mpc_constraints(u, x0, i, Ni, params, xa, x_bar, is_distributed, true);

    u_opt = [];
    try
        [u_try, ~, eflag] = fmincon(obj, u0, [], [], [], [], lb, ub, nlc_relax, opts);
        if eflag > 0 && ~isempty(u_try) && ~any(isnan(u_try))
            u_opt = u_try;
        end
    catch
    end
    if ~isempty(u_opt) && ~isempty(x_bar)
        try
            [u_try, ~, eflag] = fmincon(obj, u_opt, [], [], [], [], lb, ub, nlc_full, opts);
            if eflag > 0 && ~isempty(u_try) && ~any(isnan(u_try))
                u_opt = u_try;
            end
        catch
        end
    end
    if isempty(u_opt)
        try
            [u_try, ~, eflag] = fmincon(obj, u0, [], [], [], [], lb, ub, [], opts);
            if eflag > 0 && ~isempty(u_try) && ~any(isnan(u_try))
                u_opt = u_try;
            end
        catch
        end
    end
    if isempty(u_opt) || any(isnan(u_opt))
        u_opt = u0;
    end
    u_opt = reshape(u_opt, m, Ni);
    x_opt = predict_horizon(u_opt, x0, i, Ni, params, xa, is_distributed);
end

function cost = mpc_cost(u_seq, x0, i, Ni, params, xa, is_distributed)
    u_seq = reshape(u_seq, params.m, Ni);
    x_pred = predict_horizon(u_seq, x0, i, Ni, params, xa, is_distributed);
    cost = 0;
    for l = 1:Ni
        xl = x_pred(:, l);
        ul = u_seq(:, l);
        cost = cost + xl' * params.Q{i} * xl + ul' * params.R{i} * ul;
    end
    xN = x_pred(:, Ni + 1);
    cost = cost + xN' * params.P{i} * xN;
end

function [c, ceq] = mpc_constraints(u_seq, x0, i, Ni, params, xa, x_bar, ...
        is_distributed, use_robust)
    u_seq = reshape(u_seq, params.m, Ni);
    x_pred = predict_horizon(u_seq, x0, i, Ni, params, xa, is_distributed);
    sq_lam = sqrt(params.lambda);
    c = [];
    for l = 0:Ni-1
        eps_l = params.eps_factor * (1 - sq_lam^l) / (1 - sq_lam + 1e-12);
        tight = (1 - eps_l) * params.x_max(i);
        c = [c; abs(x_pred(1, l + 1)) - tight; abs(x_pred(2, l + 1)) - tight]; %#ok<AGROW>
    end
    c = [c; p_norm(x_pred(:, Ni + 1), params.P{i}) - 0.85 * params.epsilon(i)];
    if use_robust && ~isempty(x_bar)
        nbar = size(x_bar, 2);
        for l = 0:Ni
            xb = x_bar(:, min(l + 1, nbar));
            c = [c; p_norm(x_pred(:, l + 1) - xb, params.P{i}) - params.sigma]; %#ok<AGROW>
        end
    end
    ceq = [];
end

function x_pred = predict_horizon(u_seq, x0, i, Ni, params, xa, is_distributed)
    n = params.n;
    x_pred = zeros(n, Ni + 1);
    x_pred(:, 1) = x0;
    for l = 1:Ni
        x_nb = pack_neighbors_at(xa, l, params.M);
        u = u_seq(:, min(l, size(u_seq, 2)));
        x_pred(:, l + 1) = subsystem_dynamics(x_pred(:, l), u, i, x_nb, ...
            params, is_distributed);
    end
end

function x_nb = pack_neighbors_at(xa, l, M)
    x_nb = cell(M, 1);
    for j = 1:M
        if ~isempty(xa) && numel(xa) >= j && ~isempty(xa{j})
            x_nb{j} = xa{j}(:, min(l, size(xa{j}, 2)));
        end
    end
end

function x_next = subsystem_dynamics(x, u, i, x_neighbors, params, use_coupling, rho_use)
    if nargin < 7
        rho_use = params.rho;
    end
    Ts = params.Ts;
    mu = params.mu(i);
    dx1 = x(2);
    dx2 = mu * (1 - x(1)^2) * x(2) - x(1) + u;
    if use_coupling
        dx2 = dx2 + coupling_term(i, x, x_neighbors, rho_use);
    end
    x_next = x + Ts * [dx1; dx2];
end

function c = coupling_term(i, x, x_neighbors, rho)
    c = 0;
    switch i
        case 1
            if numel(x_neighbors) >= 2 && ~isempty(x_neighbors{2})
                xj = x_neighbors{2};
                c = rho * (0.057 * xj(1) * xj(2) + 0.1 * (x(2) - xj(2)));
            end
        case 2
            c1 = 0; c3 = 0;
            if numel(x_neighbors) >= 1 && ~isempty(x_neighbors{1})
                x1s = x_neighbors{1};
                c1 = 0.1 * (x(2) - x1s(2));
            end
            if numel(x_neighbors) >= 3 && ~isempty(x_neighbors{3})
                x3s = x_neighbors{3};
                c3 = 0.1 * (x(2) - x3s(2));
            end
            c = rho * (c1 + c3);
        case 3
            if numel(x_neighbors) >= 2 && ~isempty(x_neighbors{2})
                xj = x_neighbors{2};
                c = rho * (0.057 * xj(1) * xj(2) + 0.1 * (x(2) - xj(2)));
            end
    end
end

function [L_z, L_h, L_hij] = compute_lipschitz_constants(params)


    M = params.M;
    ngrid = 41;
    xs = linspace(-params.x_max(1), params.x_max(1), ngrid);
    L_z = zeros(M, 1);
    L_h = zeros(M, 1);
    L_hij = zeros(M, M);

    for i = 1:M
        Lmax = 0;
        mu = params.mu(i);
        Ts = params.Ts;
        for a = 1:ngrid
            for b = 1:ngrid
                x1 = xs(a);
                x2 = xs(b);
                Jz = [1, Ts; ...
                      -Ts * (2 * mu * x1 * x2 + 1), 1 + Ts * mu * (1 - x1^2)];
                Lmax = max(Lmax, norm(Jz, 2));
            end
        end
        L_z(i) = Lmax;
    end

    for i = 1:M
        for jj = 1:numel(params.N_in{i})
            j = params.N_in{i}(jj);
            L_hij(i, j) = max_coupling_jacobian_norm(i, j, params, xs);
        end
        nb = params.N_in{i};
        if isempty(nb)
            L_h(i) = 0;
        else
            L_h(i) = max(L_hij(i, nb));
        end
    end
end

function Lmax = max_coupling_jacobian_norm(i, j, params, xs)
    ngrid = numel(xs);
    Lmax = 0;
    x_own = [0; 0];
    x_nb = cell(params.M, 1);
    for a = 1:ngrid
        for b = 1:ngrid
            xj = [xs(a); xs(b)];
            x_nb{j} = xj;
            J = jacobian_coupling_wrt_neighbor(i, j, x_own, x_nb, params);
            Lmax = max(Lmax, norm(J, 2));
        end
    end
end

function J = jacobian_coupling_wrt_neighbor(i, j, x, x_nb, params)
    eps_fd = 1e-6;
    J = zeros(params.n, params.n);
    xj0 = x_nb{j};
    c0 = coupling_increment(i, x, x_nb, params);
    for col = 1:params.n
        x_nb_p = x_nb;
        xjp = xj0;
        xjp(col) = xjp(col) + eps_fd;
        x_nb_p{j} = xjp;
        cp = coupling_increment(i, x, x_nb_p, params);
        J(:, col) = (cp - c0) / eps_fd;
    end
end

function dx = coupling_increment(i, x, x_neighbors, params)
    dx = params.Ts * [0; coupling_term(i, x, x_neighbors, params.rho)];
end

function d = generate_disturbances(params, T_sim, seed)
    rng(seed);
    d = cell(params.M, 1);
    for i = 1:params.M
        d{i} = zeros(params.n, T_sim);
        P = params.P{i};
        for k = 1:T_sim
            v = randn(params.n, 1);
            vn = p_norm(v, P);
            if vn < 1e-12
                d{i}(:, k) = zeros(params.n, 1);
            else
                d{i}(:, k) = (params.xi(i) * rand() / vn) * v;
            end
        end
    end
end

function [RMSEx, Tbar_omega, Ju] = calculate_paper_metrics(x_traj, u_traj, T_omega, params, T_sim)
    M = params.M;
    acc = 0;
    for k = 0:T_sim-1
        for i = 1:M
            acc = acc + sum(x_traj{i}(:, k + 1).^2);
        end
    end
    RMSEx = sqrt(acc / (M * T_sim));
    Tbar_omega = mean(T_omega) * params.Ts;
    Ju = 0;
    for i = 1:M
        k_end = min(max(T_omega(i), 1), T_sim);
        for k = 1:k_end
            uk = u_traj{i}(:, k);
            Ju = Ju + uk' * params.R{i} * uk;
        end
    end
end

function [xi, yi] = smooth_series(x, y)
    x = x(:)';
    y = y(:)';
    n = numel(x);
    if n < 2
        xi = x; yi = y; return;
    end
    xi = linspace(x(1), x(end), max(10 * n, 400));
    yi = pchip(x, y, xi);
end

function print_KP_lipschitz_report(params)
    fprintf('==============================================================\n');
    fprintf(' Computed K, P (LQR + Lyapunov) and Lipschitz constants\n');
    fprintf('==============================================================\n');
    for i = 1:params.M
        fprintf('Subsystem %d:\n', i);
        fprintf('  K_%d = [%.4f, %.4f]\n', i, params.K{i}(1), params.K{i}(2));
        fprintf('  P_%d = [ %.4f, %.4f ;\n', i, params.P{i}(1,1), params.P{i}(1,2));
        fprintf('          %.4f, %.4f ]\n', params.P{i}(2,1), params.P{i}(2,2));
        fprintf('  L_z_%d = %.4f,  bar(L_h)_%d = %.4f\n', i, params.L_z(i), i, params.L_h(i));
    end
    fprintf('  L_h12 = %.4f, L_h21 = %.4f, L_h23 = %.4f, L_h32 = %.4f\n', ...
        params.L_hij(1,2), params.L_hij(2,1), params.L_hij(2,3), params.L_hij(3,2));
    fprintf('==============================================================\n\n');
end

function nrm = p_norm(x, P)
    nrm = sqrt(max(x' * P * x, 0));
end

function u = saturate_u(u, params, i)
    u = max(params.u_min(i), min(params.u_max(i), u));
end

function print_trigger_reasons(trigger_reason, M)
    for i = 1:M
        n_init = sum(trigger_reason{i} == 1);
        n_dev = sum(trigger_reason{i} == 2);
        n_to = sum(trigger_reason{i} == 3);
        n_acc = sum(trigger_reason{i} == 4);
        fprintf('  Subsystem %d: Init=%d, Instant(16)=%d, Horizon=%d, Accum(17)=%d\n', ...
            i, n_init, n_dev, n_to, n_acc);
    end
end

function s = fmt_step(k)
    if isinf(k)
        s = 'never';
    else
        s = sprintf('%d', k);
    end
end

function result = iif(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end
