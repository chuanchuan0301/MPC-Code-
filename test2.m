clear; clc; close all;
script_dir = fileparts(mfilename('fullpath'));
if ~isempty(script_dir), cd(script_dir); end


Q_weight = 0.08;
R_weight = 0.01;
N = 10;
T_sim = 100;
Ts = 0.17;
mu_vdp = [0.10; 0.18; 0.25];
rho = 1;

tau_base = 0.08;
xi_base = 0.008;
sigma_val = 0.005;
lambda_val = 0.95;
beta_et_val = 0.3;
gamma_et_val = 0.15;
alpha_i_val = 0.9;
epsilon_base = 0.08;
epsilon_xi = 0.16;
TT_period = 4;
eps_factor_val = 0.01;
x_max_val = 3;
u_max_val = 3;
ell_bits = 8;


params.M = 3;
params.n = 2;
params.m = 1;
params.Ts = Ts;
params.N = N * ones(params.M, 1);
params.rho = rho;
params.mu = mu_vdp;
params.alpha_i = alpha_i_val;
params.ell = ell_bits;

params.N_in = cell(params.M, 1);
params.N_in{1} = 2;
params.N_in{2} = [1, 3];
params.N_in{3} = 2;
params.N_out = cell(params.M, 1);
params.N_out{1} = 2;
params.N_out{2} = [1, 3];
params.N_out{3} = 2;

for i = 1:params.M
    params.A{i} = [1, Ts; -Ts, 1 + mu_vdp(i) * Ts];
    params.B{i} = [0; Ts];
end

params.Q = cell(params.M, 1);
params.R = cell(params.M, 1);
params.P = cell(params.M, 1);
params.K = cell(params.M, 1);
params.Q_star = cell(params.M, 1);
params.A_cl = cell(params.M, 1);

kappa = 1.2;
params.kappa = kappa;
for i = 1:params.M
    params.Q{i} = Q_weight * eye(params.n);
    params.R{i} = R_weight * eye(params.m);
    [K_lqr, ~, ~] = dlqr(params.A{i}, params.B{i}, params.Q{i}, params.R{i});
    params.K{i} = -K_lqr;
    params.A_cl{i} = params.A{i} + params.B{i} * params.K{i};
    params.Q_star{i} = params.Q{i} + params.K{i}' * params.R{i} * params.K{i};
    params.P{i} = dlyap(params.A_cl{i}', params.kappa * params.Q_star{i});
end

params.x_max = x_max_val * ones(params.M, 1);
[params.L_z, params.L_h, params.L_hij] = compute_lipschitz_constants(params);
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
params.TT_period = TT_period;

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
    L_Qi = 2 * params.zeta(i) * sqrt(lam_Q_max);
    params.chi(i) = L_Qi * lam_Q_max / params.lam_P_min(i);
    params.Xi_factor(i) = max(0, lam_Q_min / params.lam_P_max(i) ...
        - params.gamma_et(i)^2 * lam_Qstar_max / params.lam_P_min(i));
end

x0 = {[1.8; 0]; [-1.3; 0]; [1.2; 0]};
disturbances_et = generate_disturbances(params, T_sim, 2024);
disturbances_tt = disturbances_et;

[x_et, u_et, trigger_et, ~, mpc_et, fb_et, reason_et, T_omega_et] = ...
    simulate_cloud_dmpc(x0, T_sim, params, disturbances_et, true, false);
[x_tt, u_tt, trigger_tt, ~, mpc_tt, fb_tt, ~, T_omega_tt] = ...
    simulate_cloud_dmpc(x0, T_sim, params, disturbances_tt, true, true);


n_et = cellfun(@numel, trigger_et);
n_tt = cellfun(@numel, trigger_tt);
avg_trig_et = mean(n_et);
avg_trig_tt = mean(n_tt);
n_solve_et = max(sum(n_et), 1);
n_solve_tt = max(sum(n_tt), 1);
avg_cpu_et = 1000 * (sum(cellfun(@sum, mpc_et)) + sum(cellfun(@sum, fb_et))) / n_solve_et;
avg_cpu_tt = 1000 * (sum(cellfun(@sum, mpc_tt)) + sum(cellfun(@sum, fb_tt))) / n_solve_tt;

ss_et = 0; ss_tt = 0;
for i = 1:params.M
    ss_et = ss_et + norm(x_et{i}(:, end));
    ss_tt = ss_tt + norm(x_tt{i}(:, end));
end
ss_et = ss_et / params.M;
ss_tt = ss_tt / params.M;

T_horizon = T_sim * Ts;
S_packet = (params.n * (N + 1) + params.m * N) * params.ell;  % bits
B_et = sum(n_et) * S_packet;
B_tt = sum(n_tt) * S_packet;
Rbar_et = (B_et / T_horizon) / 1000;
Rbar_tt = (B_tt / T_horizon) / 1000;

fprintf('Trigger counts ET = [%s], avg = %.3f (paper ~7.3)\n', num2str(n_et'), avg_trig_et);
fprintf('Trigger counts TT = [%s], avg = %.3f (paper 25)\n', num2str(n_tt'), avg_trig_tt);
fprintf('ET total %d, TT total %d (paper 22 vs 75)\n', sum(n_et), sum(n_tt));
for i = 1:params.M
    rr = reason_et{i};
    fprintf('  ET subsystem %d: init=%d, inst(16)=%d, horizon=%d, acc(17)=%d, T_omega=%d\n', ...
        i, sum(rr==1), sum(rr==2), sum(rr==3), sum(rr==4), T_omega_et(i));
end




ET_results.x_traj = x_et;
ET_results.u_traj = u_et;
ET_results.trigger_times = trigger_et;
TT_results.x_traj = x_tt;
TT_results.u_traj = u_tt;
TT_results.trigger_times = trigger_tt;


set(0, 'DefaultFigureVisible', 'on');
steps = 0:T_sim;
steps_u = 0:T_sim-1;

et_colors = {[0.000, 0.447, 0.698], [0.800, 0.200, 0.200], [0.000, 0.600, 0.300]};
tt_colors = {[0.900, 0.600, 0.000], [0.400, 0.760, 0.200], [0.700, 0.400, 0.800]};
lw_et = 2.0; lw_tt = 2.0;
fs_label = 17; fs_tick = 15; fs_legend = 14;

zoom1_center = [min(85, 0.85 * T_sim), 0]; zoom1_mag = 3; zoom1_inset = [0.52, 0.36, 0.2, 0.15];
zoom2_center = [min(85, 0.85 * T_sim), 0]; zoom2_mag = 3; zoom2_inset = [0.52, 0.36, 0.2, 0.15];


fig1 = figure('Name', 'State x1', 'NumberTitle', 'off', 'Color', 'w', 'Visible', 'on', ...
    'Position', [50, 50, 900, 500]);
ax1_main = axes('Parent', fig1, 'Position', [0.13, 0.14, 0.83, 0.82]);
hold(ax1_main, 'on'); box(ax1_main, 'on'); grid(ax1_main, 'on');
for i = 1:params.M
    [tx, y_et] = smooth_series(steps, x_et{i}(1, :));
    plot(ax1_main, tx, y_et, '-', 'Color', et_colors{i}, ...
        'LineWidth', lw_et, 'DisplayName', sprintf('Subsystem %d ET', i));
end
for i = 1:params.M
    [tx, y_tt] = smooth_series(steps, x_tt{i}(1, :));
    plot(ax1_main, tx, y_tt, '-', 'Color', tt_colors{i}, ...
        'LineWidth', lw_tt, 'DisplayName', sprintf('Subsystem %d TT', i));
end
yline(ax1_main, 0, 'Color', [0.5 0.5 0.5], 'LineStyle', ':', 'LineWidth', 0.8, ...
    'HandleVisibility', 'off');
xlabel(ax1_main, 'Step', 'FontSize', fs_label, 'Interpreter', 'latex');
ylabel(ax1_main, 'State $x_1$', 'FontSize', fs_label, 'Interpreter', 'latex');
xlim(ax1_main, [0, T_sim]);
set(ax1_main, 'GridLineStyle', ':', 'GridAlpha', 0.25, 'FontSize', fs_tick);
legend(ax1_main, 'Location', 'northeast', 'FontSize', fs_legend, 'NumColumns', 2);
Y1 = cell(1, 2 * params.M);
for i = 1:params.M
    Y1{i} = x_et{i}(1, :);
    Y1{params.M + i} = x_tt{i}(1, :);
end
[zx1, zy1] = calc_zoom_range_data(steps, Y1, zoom1_center(1), zoom1_mag, T_sim);
rectangle(ax1_main, 'Position', [zx1(1), zy1(1), diff(zx1), diff(zy1)], ...
    'EdgeColor', [0.15 0.15 0.15], 'LineWidth', 1.3);
ax1_inset = axes(fig1, 'Position', zoom1_inset);
hold(ax1_inset, 'on'); box(ax1_inset, 'on'); grid(ax1_inset, 'on');
for i = 1:params.M
    [tx, y_et] = smooth_series(steps, x_et{i}(1, :));
    plot(ax1_inset, tx, y_et, '-', 'Color', et_colors{i}, 'LineWidth', lw_et);
end
for i = 1:params.M
    [tx, y_tt] = smooth_series(steps, x_tt{i}(1, :));
    plot(ax1_inset, tx, y_tt, '-', 'Color', tt_colors{i}, 'LineWidth', lw_tt);
end
xlim(ax1_inset, zx1); ylim(ax1_inset, zy1);
set(ax1_inset, 'GridLineStyle', ':', 'FontSize', 8, 'Color', [0.96 0.96 0.98]);


fig2 = figure('Name', 'State x2', 'NumberTitle', 'off', 'Color', 'w', 'Visible', 'on', ...
    'Position', [100, 100, 900, 500]);
ax2_main = axes('Parent', fig2, 'Position', [0.13, 0.14, 0.83, 0.82]);
hold(ax2_main, 'on'); box(ax2_main, 'on'); grid(ax2_main, 'on');
for i = 1:params.M
    [tx, y_et] = smooth_series(steps, x_et{i}(2, :));
    plot(ax2_main, tx, y_et, '-', 'Color', et_colors{i}, ...
        'LineWidth', lw_et, 'DisplayName', sprintf('Subsystem %d ET', i));
end
for i = 1:params.M
    [tx, y_tt] = smooth_series(steps, x_tt{i}(2, :));
    plot(ax2_main, tx, y_tt, '-', 'Color', tt_colors{i}, ...
        'LineWidth', lw_tt, 'DisplayName', sprintf('Subsystem %d TT', i));
end
yline(ax2_main, 0, 'Color', [0.5 0.5 0.5], 'LineStyle', ':', 'LineWidth', 0.8, ...
    'HandleVisibility', 'off');
xlabel(ax2_main, 'Step', 'FontSize', fs_label, 'Interpreter', 'latex');
ylabel(ax2_main, 'State $x_2$', 'FontSize', fs_label, 'Interpreter', 'latex');
xlim(ax2_main, [0, T_sim]);
set(ax2_main, 'GridLineStyle', ':', 'GridAlpha', 0.25, 'FontSize', fs_tick);
legend(ax2_main, 'Location', 'northeast', 'FontSize', fs_legend, 'NumColumns', 2);
Y2 = cell(1, 2 * params.M);
for i = 1:params.M
    Y2{i} = x_et{i}(2, :);
    Y2{params.M + i} = x_tt{i}(2, :);
end
[zx2, zy2] = calc_zoom_range_data(steps, Y2, zoom2_center(1), zoom2_mag, T_sim);
rectangle(ax2_main, 'Position', [zx2(1), zy2(1), diff(zx2), diff(zy2)], ...
    'EdgeColor', [0.15 0.15 0.15], 'LineWidth', 1.3);
ax2_inset = axes(fig2, 'Position', zoom2_inset);
hold(ax2_inset, 'on'); box(ax2_inset, 'on'); grid(ax2_inset, 'on');
for i = 1:params.M
    [tx, y_et] = smooth_series(steps, x_et{i}(2, :));
    plot(ax2_inset, tx, y_et, '-', 'Color', et_colors{i}, 'LineWidth', lw_et);
end
for i = 1:params.M
    [tx, y_tt] = smooth_series(steps, x_tt{i}(2, :));
    plot(ax2_inset, tx, y_tt, '-', 'Color', tt_colors{i}, 'LineWidth', lw_tt);
end
xlim(ax2_inset, zx2); ylim(ax2_inset, zy2);
set(ax2_inset, 'GridLineStyle', ':', 'FontSize', 8, 'Color', [0.96 0.96 0.98]);


fig3 = figure('Name', 'Control Input u', 'NumberTitle', 'off', 'Color', 'w', 'Visible', 'on', ...
    'Position', [150, 150, 900, 500]);
ax3_main = axes('Parent', fig3, 'Position', [0.13, 0.14, 0.83, 0.82]);
hold(ax3_main, 'on'); box(ax3_main, 'on'); grid(ax3_main, 'on');
for i = 1:params.M
    [tu, y_et] = smooth_series(steps_u, u_et{i}(:).');
    plot(ax3_main, tu, y_et, '-', 'Color', et_colors{i}, ...
        'LineWidth', lw_et, 'DisplayName', sprintf('Subsystem %d ET', i));
end
for i = 1:params.M
    [tu, y_tt] = smooth_series(steps_u, u_tt{i}(:).');
    plot(ax3_main, tu, y_tt, '-', 'Color', tt_colors{i}, ...
        'LineWidth', lw_tt, 'DisplayName', sprintf('Subsystem %d TT', i));
end
yline(ax3_main, 0, 'Color', [0.5 0.5 0.5], 'LineStyle', ':', 'LineWidth', 0.8, ...
    'HandleVisibility', 'off');
xlabel(ax3_main, 'Step', 'FontSize', fs_label, 'Interpreter', 'latex');
ylabel(ax3_main, 'Control input $u$', 'FontSize', fs_label, 'Interpreter', 'latex');
xlim(ax3_main, [0, T_sim]);
um = 0.15;
for i = 1:params.M
    um = max([um, max(abs(u_et{i}(:))), max(abs(u_tt{i}(:)))]);
end
ylim(ax3_main, [-1.15 * um, 1.15 * um]);
set(ax3_main, 'GridLineStyle', ':', 'GridAlpha', 0.25, 'FontSize', fs_tick);
legend(ax3_main, 'Location', 'northeast', 'FontSize', fs_legend, 'NumColumns', 2);


fig4 = figure('Name', 'Trigger Instants', 'NumberTitle', 'off', 'Color', 'w', 'Visible', 'on', ...
    'Position', [200, 200, 900, 500]);
c_et = [0.000, 0.447, 0.698];
c_tt = [0.835, 0.369, 0.000];
left_m = 0.13; right_m = 0.04; top_m = 0.04; bot_m = 0.14;
ax_w = 1 - left_m - right_m;
gap_v = 0.02;
ax_h = (1 - top_m - bot_m - (params.M - 1) * gap_v) / params.M;
for si = 1:params.M
    ax = axes('Position', [left_m, bot_m + (params.M - si) * (ax_h + gap_v), ax_w, ax_h], ...
        'Parent', fig4);
    hold(ax, 'on'); box(ax, 'on'); grid(ax, 'on');
    set(ax, 'GridLineStyle', ':', 'GridAlpha', 0.25);
    if ~isempty(trigger_et{si})
        stem(ax, trigger_et{si}, 0.72 * ones(size(trigger_et{si})), ...
            'o', 'Color', c_et, 'MarkerSize', 5, 'MarkerFaceColor', c_et, ...
            'LineWidth', 1.2, 'DisplayName', 'Event-Triggered');
    end
    if ~isempty(trigger_tt{si})
        stem(ax, trigger_tt{si}, 0.28 * ones(size(trigger_tt{si})), ...
            'd', 'Color', c_tt, 'MarkerSize', 5, 'MarkerFaceColor', c_tt, ...
            'LineWidth', 1.2, 'DisplayName', 'Time-Triggered');
    end
    xlim(ax, [0, T_sim]); ylim(ax, [0, 1]);
    set(ax, 'YTick', [0.28, 0.72], 'YTickLabel', {}, ...
        'FontSize', fs_tick, 'TickDir', 'in', 'LineWidth', 0.8, ...
        'GridLineStyle', ':', 'GridAlpha', 0.3);
    if si == params.M
        xlabel(ax, ' Step ', 'FontSize', fs_label, 'Interpreter', 'latex');
    else
        set(ax, 'XTickLabel', []);
    end
    ylabel(ax, sprintf('$k_%d^r$', si), 'FontSize', fs_label, 'Interpreter', 'latex');
    text(ax, T_sim * 0.985, 0.90, sprintf('Subsystem %d', si), ...
        'HorizontalAlignment', 'right', 'FontSize', fs_tick - 1, 'Color', [0 0 0]);
    if si == 1
        legend(ax, 'Location', 'northeast', 'FontSize', fs_legend, ...
            'EdgeColor', [0.75 0.75 0.75]);
    end
end
drawnow;


function [x_traj, u_traj, trigger_times, conv_time, mpc_cpu_time, fb_cpu_time, ...
          trigger_reason, T_omega] = ...
    simulate_cloud_dmpc(x0, T_sim, params, disturbances, is_distributed, time_triggered)

    M = params.M; n = params.n; m = params.m;
    x_traj = cell(M, 1); u_traj = cell(M, 1);
    trigger_times = cell(M, 1); trigger_reason = cell(M, 1);
    conv_time = cell(M, 1); mpc_cpu_time = cell(M, 1); fb_cpu_time = cell(M, 1);
    u_opt_seq = cell(M, 1); x_star = cell(M, 1);
    k_r = -ones(M, 1); T_omega = T_sim * ones(M, 1);
    stay_omega = false(M, 1);

    for i = 1:M
        Ni = params.N(i);
        x_traj{i} = zeros(n, T_sim + 1); x_traj{i}(:, 1) = x0{i};
        u_traj{i} = zeros(m, T_sim);
        trigger_times{i} = []; trigger_reason{i} = [];
        conv_time{i} = []; mpc_cpu_time{i} = []; fb_cpu_time{i} = [];
        u_opt_seq{i} = zeros(m, Ni);
        x_star{i} = repmat(x0{i}, 1, Ni + 1);
    end

    for k = 0:T_sim-1
        x_curr = cell(M, 1);
        in_omega = false(M, 1);
        for i = 1:M
            x_curr{i} = x_traj{i}(:, k + 1);
            nx = p_norm(x_curr{i}, params.P{i});
            in_omega(i) = (nx <= params.epsilon(i)) || ...
                (stay_omega(i) && nx <= 1.15 * params.epsilon(i));
            stay_omega(i) = in_omega(i);
            if in_omega(i) && T_omega(i) == T_sim
                T_omega(i) = k;
                conv_time{i} = k * params.Ts;
            end
        end

        triggered = false(M, 1);
        reason = zeros(M, 1);
        for i = 1:M
            if time_triggered
                triggered(i) = (mod(k, params.TT_period) == 0);
                if triggered(i)
                    reason(i) = 5;
                end
            else
                if in_omega(i)
                    continue;
                end
                [triggered(i), reason(i)] = check_trigger(k, i, x_curr{i}, ...
                    x_star{i}, k_r(i), params, x_traj{i});
            end
        end

        for i = 1:M
            if ~triggered(i), continue; end
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
            eta_w = 1;
            if k_r(i) >= 0
                eta_w = max(k - k_r(i), 1);
            end
            tic;
            [u_opt_seq{i}, x_star{i}] = solve_mpc(x_curr{i}, i, params.N(i), ...
                params, xa, x_bar, is_distributed, u_opt_seq{i}, eta_w);
            mpc_cpu_time{i} = [mpc_cpu_time{i}, toc];
            k_r(i) = k;
        end

        for i = 1:M
            use_local_fb = in_omega(i) && ~time_triggered;
            if use_local_fb
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
            u_traj{i}(:, k + 1) = max(params.u_min(i), min(params.u_max(i), u_traj{i}(:, k + 1)));
        end

        x_nb = x_curr;
        for i = 1:M
            x_next = subsystem_dynamics(x_curr{i}, u_traj{i}(:, k + 1), i, x_nb, params, true);
            x_traj{i}(:, k + 2) = x_next + disturbances{i}(:, k + 1);
        end
    end
end

function [triggered, reason] = check_trigger(k, i, x_curr, x_star_i, k_r_i, params, x_hist)
    triggered = false; reason = 0; Ni = params.N(i);
    if k == 0 || k_r_i < 0
        triggered = true; reason = 1; return;
    end
    eta = k - k_r_i;
    if eta >= Ni
        triggered = true; reason = 3; return;
    end
    x_pred = x_star_i(:, min(eta + 1, size(x_star_i, 2)));
    e_now = p_norm(x_curr - x_pred, params.P{i});
    if e_now >= compute_threshold(k, params, i)
        triggered = true; reason = 2; return;
    end
    acc = 0; nstar = size(x_star_i, 2);
    for s = 1:eta
        xs = x_hist(:, min(k_r_i + s + 1, size(x_hist, 2)));
        xp = x_star_i(:, min(s + 1, nstar));
        acc = acc + p_norm(xs - xp, params.P{i});
    end
    Xi_i = params.epsilon_xi^2 * max(eta, 1) * params.Xi_factor(i);
    if Xi_i > 0 && params.chi(i) * acc > Xi_i
        triggered = true; reason = 4;
    end
end

function delta = compute_threshold(k, params, i)
    sqmin = sqrt(params.lam_P_min(i));
    sqmax = sqrt(params.lam_P_max(i));
    num = sqmin * (params.tau(i) - params.xi(i)) ...
        - sqmax * params.Gamma(i) * params.L_h(i) * (params.tau(i) + params.N(i) * params.sigma);
    den = sqmax * params.L_z(i) + 1e-12;
    adaptive = 1 - params.beta_et(i) * exp(-params.lambda * k);
    if num <= 0
        num = 0.15 * sqmin * max(params.tau(i) - params.xi(i), 1e-3);
    end
    delta = max(5e-4, adaptive * num / den);
end

function xa = construct_assumed_state(j, k, N_i, params, x_star, k_r)
    n = params.n; xa = zeros(n, N_i + 1); Nj = params.N(j);
    if k_r(j) < 0 || isempty(x_star{j})
        xa = repmat(x_star{j}(:, 1), 1, N_i + 1); return;
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
    n = params.n; x_bar = zeros(n, Ni + 1); eta = k - k_r_prev;
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

function [u_opt, x_opt] = solve_mpc(x0, i, Ni, params, xa, x_bar, is_distributed, u_prev, eta)
    m = params.m;
    u_fb = saturate_u(params.K{i} * x0, params, i);
    u0 = repmat(u_fb, Ni, 1);
    if nargin >= 8 && ~isempty(u_prev) && numel(u_prev) == m * Ni
        up = reshape(u_prev, m, Ni);
        if nargin < 9 || isempty(eta) || eta < 1
            eta = 1;
        end
        shift = min(round(eta), Ni);
        if shift >= Ni
            u0 = repmat(u_fb, 1, Ni);
        else
            u0 = [up(:, shift+1:end), repmat(u_fb, 1, shift)];
        end
        u0 = u0(:);
    end
    lb = repmat(params.u_min(i), Ni, 1);
    ub = repmat(params.u_max(i), Ni, 1);
    opts = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', ...
        'MaxIterations', 60, 'MaxFunctionEvaluations', 1200, ...
        'OptimalityTolerance', 1e-4, 'ConstraintTolerance', 1e-4);
    obj = @(u) mpc_cost(u, x0, i, Ni, params, xa, is_distributed);
    nlc_relax = @(u) mpc_constraints(u, x0, i, Ni, params, xa, x_bar, is_distributed, false);
    u_opt = [];
    try
        [u_try, ~, eflag] = fmincon(obj, u0, [], [], [], [], lb, ub, nlc_relax, opts);
        if eflag > 0 && ~isempty(u_try) && ~any(isnan(u_try)), u_opt = u_try; end
    catch
    end
    if isempty(u_opt)
        try
            [u_try, ~, eflag] = fmincon(obj, u0, [], [], [], [], lb, ub, [], opts);
            if eflag > 0 && ~isempty(u_try) && ~any(isnan(u_try)), u_opt = u_try; end
        catch
        end
    end
    if isempty(u_opt) || any(isnan(u_opt)), u_opt = u0; end
    u_opt = reshape(u_opt, m, Ni);
    x_opt = predict_horizon(u_opt, x0, i, Ni, params, xa, is_distributed);
end

function cost = mpc_cost(u_seq, x0, i, Ni, params, xa, is_distributed)
    u_seq = reshape(u_seq, params.m, Ni);
    x_pred = predict_horizon(u_seq, x0, i, Ni, params, xa, is_distributed);
    cost = 0;
    for l = 1:Ni
        cost = cost + x_pred(:, l)' * params.Q{i} * x_pred(:, l) + u_seq(:, l)' * params.R{i} * u_seq(:, l);
    end
    cost = cost + x_pred(:, Ni + 1)' * params.P{i} * x_pred(:, Ni + 1);
end

function [c, ceq] = mpc_constraints(u_seq, x0, i, Ni, params, xa, x_bar, is_distributed, use_robust)
    u_seq = reshape(u_seq, params.m, Ni);
    x_pred = predict_horizon(u_seq, x0, i, Ni, params, xa, is_distributed);
    sq_lam = sqrt(params.lambda); c = [];
    for l = 0:Ni-1
        eps_l = params.eps_factor * (1 - sq_lam^l) / (1 - sq_lam + 1e-12);
        tight = (1 - eps_l) * params.x_max(i);
        c = [c; abs(x_pred(1, l + 1)) - tight; abs(x_pred(2, l + 1)) - tight]; 
    end
    c = [c; p_norm(x_pred(:, Ni + 1), params.P{i}) - 0.85 * params.epsilon(i)];
    if use_robust && ~isempty(x_bar)
        for l = 0:Ni
            xb = x_bar(:, min(l + 1, size(x_bar, 2)));
            c = [c; p_norm(x_pred(:, l + 1) - xb, params.P{i}) - params.sigma]; 
        end
    end
    ceq = [];
end

function x_pred = predict_horizon(u_seq, x0, i, Ni, params, xa, is_distributed)
    x_pred = zeros(params.n, Ni + 1); x_pred(:, 1) = x0;
    for l = 1:Ni
        x_nb = pack_neighbors_at(xa, l, params.M);
        x_pred(:, l + 1) = subsystem_dynamics(x_pred(:, l), u_seq(:, min(l, size(u_seq, 2))), ...
            i, x_nb, params, is_distributed);
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

function x_next = subsystem_dynamics(x, u, i, x_neighbors, params, use_coupling)
    Ts = params.Ts; mu = params.mu(i);
    dx1 = x(2);
    dx2 = mu * (1 - x(1)^2) * x(2) - x(1) + u;
    if use_coupling
        dx2 = dx2 + coupling_term(i, x, x_neighbors, params.rho);
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
                c1 = 0.1 * (x(2) - x_neighbors{1}(2));
            end
            if numel(x_neighbors) >= 3 && ~isempty(x_neighbors{3})
                c3 = 0.1 * (x(2) - x_neighbors{3}(2));
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
                x1 = xs(a); x2 = xs(b);
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
        if ~isempty(nb)
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
            x_nb{j} = [xs(a); xs(b)];
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
        xjp = xj0; xjp(col) = xjp(col) + eps_fd;
        x_nb_p{j} = xjp;
        J(:, col) = (coupling_increment(i, x, x_nb_p, params) - c0) / eps_fd;
    end
end

function dx = coupling_increment(i, x, x_neighbors, params)
    dx = params.Ts * [0; coupling_term(i, x, x_neighbors, params.rho)];
end

function d = generate_disturbances(params, T_sim, seed)
    rng(seed); d = cell(params.M, 1);
    for i = 1:params.M
        d{i} = zeros(params.n, T_sim); P = params.P{i};
        for k = 1:T_sim
            v = randn(params.n, 1); vn = p_norm(v, P);
            if vn < 1e-12
                d{i}(:, k) = 0;
            else
                d{i}(:, k) = (params.xi(i) * rand() / vn) * v;
            end
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


function nrm = p_norm(x, P)
    nrm = sqrt(max(x' * P * x, 0));
end

function u = saturate_u(u, params, i)
    u = max(params.u_min(i), min(params.u_max(i), u));
end


function [xrange, yrange] = calc_zoom_range_data(x, Ycells, cx, mag, x_total)
    x_half = x_total / mag / 2;
    cx = min(max(cx, x_half), max(x_total - x_half, x_half));
    xrange = [max(cx - x_half, 0), min(cx + x_half, x_total)];
    if xrange(2) <= xrange(1)
        xrange = [0, x_total];
    end
    ymin = inf;
    ymax = -inf;
    x = x(:)';
    for k = 1:numel(Ycells)
        y = Ycells{k}(:)';
        mask = (x >= xrange(1)) & (x <= xrange(2));
        if any(mask)
            ymin = min(ymin, min(y(mask)));
            ymax = max(ymax, max(y(mask)));
        end
    end
    if ~isfinite(ymin) || ~isfinite(ymax) || ymax <= ymin
        yrange = [-0.2, 0.2];
        return;
    end
    pad = max(0.08 * (ymax - ymin), 0.02);
    yrange = [ymin - pad, ymax + pad];
end
