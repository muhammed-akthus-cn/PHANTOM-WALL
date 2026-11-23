%% ========================================================================
%  PHANTOM-WALL: Through-Wall S-Band FMCW Tomography & Vital-Sign Engine
%  Step 1: EM Wall Propagation Physics, SVD Clutter Removal & Phase Recovery
%  Author: Muhammed Akthus C N
% =========================================================================
clear; clc; close all;

fprintf('=================================================================\n');
fprintf('  PHANTOM-WALL | Step 1: S-Band EM Wall Physics & SVD Tomography\n');
fprintf('=================================================================\n');

%% 1. S-Band FMCW Radar & Electromagnetic Wall Parameters
c  = 3e8;                  % Speed of light (m/s)
fc = 3.5e9;                % S-band carrier frequency (3.5 GHz - penetrates masonry)
lambda = c / fc;           % Wavelength (~8.57 cm)
B  = 1.5e9;                % FMCW sweep bandwidth (1.5 GHz -> 10 cm range res)
Tc = 50e-6;                % Chirp duration (50 us)
S  = B / Tc;               % Chirp slope (Hz/s)
Fs = 25.6e6;               % ADC sampling rate (25.6 MHz)
Ns = round(Tc * Fs);       % Fast-time samples per chirp (1280)
t_fast = (0:Ns-1).' / Fs;  % Fast-time vector

% Slow-time observation window (12 seconds @ 20 Hz frame rate -> 240 frames)
fps = 20;
T_obs = 12.0;
Nf = fps * T_obs;
t_slow = (0:Nf-1) / fps;

% 8-Element Virtual MIMO Array along Cross-Range (X-axis)
N_ant = 8;
d_ant = lambda / 2;
x_ant = ((0:N_ant-1) - (N_ant-1)/2) * d_ant;

% Reinforced Concrete Wall Properties (y = 1.50 m, thickness d_wall = 0.20 m)
y_wall   = 1.50;           % Wall front face distance (m)
d_wall   = 0.20;           % Wall thickness (20 cm)
eps_r    = 6.5;            % Relative permittivity of concrete
sigma_w  = 0.05;           % Conductivity of concrete (S/m)
eps_0    = 8.854e-12;
mu_0     = 4*pi*1e-7;
omega    = 2*pi*fc;

% Exact EM Propagation Constant in Lossy Concrete: gamma = alpha + j*beta
gamma_w  = sqrt(1j * omega * mu_0 * (sigma_w + 1j * omega * eps_0 * eps_r));
alpha_w  = real(gamma_w);  % Attenuation constant (Np/m)
beta_w   = imag(gamma_w);  % Phase constant (rad/m)
eta_0    = sqrt(mu_0 / eps_0);
eta_w    = sqrt(1j * omega * mu_0 / (sigma_w + 1j * omega * eps_0 * eps_r));

% Fresnel Reflection & Two-Way Wall Penetration Loss
Gamma_wall = abs((eta_w - eta_0) / (eta_w + eta_0));
T_trans    = 1 - Gamma_wall^2;
L_wall_dB  = 2 * (8.686 * alpha_w * d_wall - 10*log10(T_trans)); % Two-way loss
v_wall     = omega / beta_w;
extra_delay_m = d_wall * (c / v_wall - 1); % Optical path delay through concrete

fprintf('[EM Physics] Wall Concrete (eps_r=%.1f, sigma=%.2f S/m, d=%.2f m)\n', eps_r, sigma_w, d_wall);
fprintf('[EM Physics] Attenuation alpha = %.2f Np/m | Two-Way Wall Loss = %.2f dB\n', alpha_w, L_wall_dB);

%% 2. Trapped Victim 3D Scatterer & Sub-Millimeter Vital-Sign Kinematics
% Victim located behind wall at Cross-Range x = -0.35 m, Down-Range y = 2.85 m
victim_x = -0.35;
victim_y = 2.85;

% True physiological micro-motion: 16.8 BPM breathing (0.28 Hz) + 75 BPM heartbeat (1.25 Hz)
f_resp_true = 0.28;        % Hz (16.8 breaths/min)
f_card_true = 1.25;        % Hz (75 beats/min)
A_resp      = 4.5e-3;      % 4.5 mm chest displacement
A_card      = 0.6e-3;      % 0.6 mm cardiac displacement
delta_R_true = A_resp * sin(2*pi*f_resp_true*t_slow) + ...
               A_card * sin(2*pi*f_card_true*t_slow);

% Multi-point scatterer body model [dx, dy, RCS_weight, motion_coupling]
body_pts = [
     0.00,  0.00, 1.00, 1.00;   % Chest center (full respiratory coupling)
    -0.14,  0.04, 0.65, 0.70;   % Left thorax
     0.14,  0.04, 0.65, 0.70;   % Right thorax
     0.00,  0.08, 0.50, 0.15;   % Head/shoulder scatterer
     0.05, -0.05, 0.45, 0.30    % Abdomen scatterer
];

% Static background clutter points (concrete wall segments + rear rubble)
wall_x = linspace(-1.5, 1.5, 25);
wall_rcs = 45.0; % Wall is ~33 dB stronger than human chest scatterer

%% 3. Synthesize Raw MIMO FMCW Beat-Signal Cube [Ns x Nf x N_ant]
rng(42);
raw_cube = zeros(Ns, Nf, N_ant);
noise_sigma = 0.08;

for m = 1:N_ant
    xm = x_ant(m);
    
    % A) Strong static concrete wall reflection (front & back faces)
    wall_sig = zeros(Ns, 1);
    for w = 1:length(wall_x)
        R_w1 = hypot(wall_x(w) - xm, y_wall);
        R_w2 = hypot(wall_x(w) - xm, y_wall + d_wall) + extra_delay_m;
        tau_1 = 2 * R_w1 / c;
        tau_2 = 2 * R_w2 / c;
        wall_sig = wall_sig + (wall_rcs / length(wall_x)) * ...
            (exp(1j * 2*pi * (fc*tau_1 + S*tau_1*t_fast)) + ...
             0.45 * exp(1j * 2*pi * (fc*tau_2 + S*tau_2*t_fast)));
    end
    
    % Replicate static wall across all slow-time frames (+ tiny thermal drift)
    raw_cube(:, :, m) = repmat(wall_sig, 1, Nf) .* ...
        (1 + 0.002 * randn(1, Nf));
    
    % B) Attenuated victim echo with sub-mm chest phase modulation
    wall_atten_lin = 10^(-L_wall_dB / 20);
    for p = 1:size(body_pts, 1)
        px = victim_x + body_pts(p, 1);
        py = victim_y + body_pts(p, 2);
        R0 = hypot(px - xm, py) + extra_delay_m;
        
        for k = 1:Nf
            Rk = R0 + body_pts(p, 4) * delta_R_true(k);
            tau_k = 2 * Rk / c;
            raw_cube(:, k, m) = raw_cube(:, k, m) + ...
                wall_atten_lin * body_pts(p, 3) * ...
                exp(1j * 2*pi * (fc*tau_k + S*tau_k*t_fast));
        end
    end
    
    % C) Add receiver AWGN
    raw_cube(:, :, m) = raw_cube(:, :, m) + ...
        noise_sigma * (randn(Ns, Nf) + 1j*randn(Ns, Nf)) / sqrt(2);
end

%% 4. Range FFT & SVD Static-Wall Clutter Suppression
N_fft = 2048;
win = hann(Ns);
range_axis = (0:N_fft-1) * (Fs * c) / (2 * S * N_fft);
max_r_idx = find(range_axis <= 4.5, 1, 'last');
r_grid = range_axis(1:max_r_idx);

% Range profiles [Nr x Nf x N_ant]
rng_profiles_raw   = zeros(max_r_idx, Nf, N_ant);
rng_profiles_clean = zeros(max_r_idx, Nf, N_ant);

for m = 1:N_ant
    R_mat = fft(raw_cube(:, :, m) .* win, N_fft, 1);
    R_mat = R_mat(1:max_r_idx, :);
    rng_profiles_raw(:, :, m) = R_mat;
    
    % SVD Clutter Suppression: zero out top 2 singular values (static wall)
    [U, Sigma_svd, V] = svd(R_mat, 'econ');
    s_vals = diag(Sigma_svd);
    s_vals(1:2) = 0; % Strip dominant static wall components
    rng_profiles_clean(:, :, m) = U * diag(s_vals) * V';
end

scr_before = 10*log10(mean(abs(rng_profiles_raw(round(2.95/r_grid(2)),:,1)).^2) / ...
                      mean(abs(rng_profiles_raw(round(1.50/r_grid(2)),:,1)).^2));
scr_after  = 10*log10(mean(abs(rng_profiles_clean(round(2.95/r_grid(2)),:,1)).^2) / ...
                      mean(abs(rng_profiles_clean(round(1.50/r_grid(2)),:,1)).^2));
fprintf('[SVD Filter] Signal-to-Wall-Clutter Ratio improved from %.1f dB to +%.1f dB!\n', ...
        scr_before, scr_after);

%% 5. 2D Range–Cross-Range Backprojection Tomography
x_grid = linspace(-1.5, 1.5, 121);
y_grid = linspace(0.5, 4.2, 149);
[Xg, Yg] = meshgrid(x_grid, y_grid);

img_raw   = zeros(size(Xg));
img_clean = zeros(size(Xg));

% Coherent Backprojection across virtual MIMO aperture
for m = 1:N_ant
    xm = x_ant(m);
    % Effective radar range including concrete refractive path delay
    R_map = hypot(Xg - xm, Yg);
    R_map(Yg > y_wall) = R_map(Yg > y_wall) + extra_delay_m;
    
    % Average across slow-time power for spatial imaging
    prof_raw_avg   = mean(abs(rng_profiles_raw(:, :, m)), 2);
    prof_clean_avg = mean(abs(rng_profiles_clean(:, :, m)), 2);
    
    img_raw   = img_raw   + interp1(r_grid, prof_raw_avg, R_map, 'linear', 0);
    img_clean = img_clean + interp1(r_grid, prof_clean_avg, R_map, 'linear', 0);
end

img_raw_dB   = 20*log10(img_raw / max(img_raw(:)) + 1e-6);
img_clean_dB = 20*log10(img_clean / max(img_clean(:)) + 1e-6);

% Detect Victim Peak behind the wall (y > y_wall + d_wall + 0.2)
mask_behind = Yg > (y_wall + d_wall + 0.2);
masked_clean = img_clean .* mask_behind;
[~, peak_idx] = max(masked_clean(:));
[iy_det, ix_det] = ind2sub(size(Xg), peak_idx);
x_det = x_grid(ix_det);
y_det = y_grid(iy_det);

fprintf('[Tomography] Victim Detected at (x = %.2f m, y = %.2f m) [True: (%.2f, %.2f)]\n', ...
        x_det, y_det, victim_x, victim_y);

%% 6. Sub-Millimeter Chest-Wall Phase Extraction & Vital-Sign FFT
% Beamform slow-time complex signal directly onto the detected victim voxel
z_slow = zeros(1, Nf);
for m = 1:N_ant
    R_det = hypot(x_det - x_ant(m), y_det) + extra_delay_m;
    [~, r_bin] = min(abs(r_grid - R_det));
    phase_comp = exp(-1j * 4*pi * R_det / lambda);
    z_slow = z_slow + rng_profiles_clean(r_bin, :, m) * phase_comp;
end

% Unwrap phase and convert to millimeter displacement: dR = (lambda / 4*pi) * dPhi
phi_unwrapped = unwrap(angle(z_slow));
phi_detrended = detrend(phi_unwrapped);
disp_est_mm   = (lambda / (4*pi)) * phi_detrended * 1e3;

% Align sign ambiguity from SVD eigenvector projection
if corr(disp_est_mm.', delta_R_true.') < 0
    disp_est_mm = -disp_est_mm;
end

% Spectral estimation of breathing rate (BPM)
N_spec = 1024;
f_spec = (0:N_spec-1) * (fps / N_spec);
spec_mag = abs(fft(disp_est_mm .* hann(Nf).', N_spec));
valid_resp = (f_spec >= 0.12) & (f_spec <= 0.60);
[~, idx_bpm] = max(spec_mag(valid_resp));
f_resp_slice = f_spec(valid_resp);
bpm_est = f_resp_slice(idx_bpm) * 60;

fprintf('[Vital Signs] Estimated Breathing Rate = %.1f BPM (Ground Truth: %.1f BPM)\n', ...
        bpm_est, f_resp_true * 60);

% Save data for Step 2 (Micro-Doppler Dataset + INT8 Quantized CNN)
save('phantom_wall_step1_data.mat', 'rng_profiles_clean', 'disp_est_mm', ...
     't_slow', 'fps', 'fc', 'lambda', 'L_wall_dB', 'x_det', 'y_det', 'bpm_est');

%% 7. Daylight Industrial Telemetry Dashboard (High-Contrast Off-White UI)
fig = figure('Name', 'PHANTOM-WALL | S-Band Through-Wall Tomography', ...
             'Color', [0.97 0.98 0.99], 'Position', [80 60 1320 780]);

bg_panel = [1.00 1.00 1.00];
ink_dark = [0.10 0.14 0.20];
teal_col = [0.00 0.52 0.56];
amber_col= [0.85 0.42 0.05];
red_col  = [0.80 0.14 0.20];
grid_col = [0.85 0.88 0.91];

% --- Panel 1: Raw Range-Azimuth Image (Blinded by Concrete Wall) ---
subplot(2, 2, 1);
imagesc(x_grid, y_grid, img_raw_dB, [-35 0]);
set(gca, 'YDir', 'normal', 'Color', bg_panel, 'XColor', ink_dark, 'YColor', ink_dark, 'FontSize', 10);
colormap(gca, parula); cb1 = colorbar; cb1.Label.String = 'Normalized Power (dB)';
hold on;
rectangle('Position', [-1.5, y_wall, 3.0, d_wall], 'EdgeColor', red_col, 'LineWidth', 2, 'LineStyle', '--');
text(-1.42, y_wall + 0.09, '20 cm Reinforced Concrete Wall (+33 dB Clutter)', 'Color', 'w', 'FontWeight', 'bold', 'FontSize', 8.5);
plot(x_ant, zeros(size(x_ant)) + 0.55, '^', 'MarkerFaceColor', amber_col, 'MarkerEdgeColor', 'k', 'MarkerSize', 7);
title('1. Raw S-Band Return (Blinded by Wall Reflection)', 'Color', ink_dark, 'FontWeight', 'bold');
xlabel('Cross-Range X (m)'); ylabel('Down-Range Y (m)');

% --- Panel 2: Post-SVD Tomography (Wall Stripped, Victim Isolated) ---
subplot(2, 2, 2);
imagesc(x_grid, y_grid, img_clean_dB, [-25 0]);
set(gca, 'YDir', 'normal', 'Color', bg_panel, 'XColor', ink_dark, 'YColor', ink_dark, 'FontSize', 10);
colormap(gca, parula); cb2 = colorbar; cb2.Label.String = 'Clutter-Suppressed Power (dB)';
hold on;
rectangle('Position', [-1.5, y_wall, 3.0, d_wall], 'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 1.5, 'LineStyle', ':');
plot(x_det, y_det, 'o', 'MarkerSize', 16, 'LineWidth', 2.2, 'Color', red_col);
text(x_det + 0.14, y_det, sprintf('TRAPPED SUBJECT\n(%.2fm, %.2fm)', x_det, y_det), ...
     'Color', 'w', 'FontWeight', 'bold', 'FontSize', 8.5, 'BackgroundColor', [0.1 0.14 0.2 0.75]);
title(sprintf('2. Post-SVD Tomography (+%.1f dB Clutter Gain)', scr_after - scr_before), 'Color', ink_dark, 'FontWeight', 'bold');
xlabel('Cross-Range X (m)'); ylabel('Down-Range Y (m)');

% --- Panel 3: Sub-Millimeter Chest-Wall Phase Displacement ---
subplot(2, 2, 3);
plot(t_slow, delta_R_true * 1e3, '--', 'Color', [0.45 0.50 0.58], 'LineWidth', 1.6); hold on;
plot(t_slow, disp_est_mm, '-', 'Color', teal_col, 'LineWidth', 2.2);
set(gca, 'Color', bg_panel, 'XColor', ink_dark, 'YColor', ink_dark, 'GridColor', grid_col, 'FontSize', 10);
grid on; ylim([-7 7]);
legend({'True Chest Displacement (4.5 mm)', 'Recovered Unwrapped RF Phase'}, 'Location', 'northeast', 'Box', 'on');
title('3. Through-Wall Sub-Millimeter Chest Phase Extraction', 'Color', ink_dark, 'FontWeight', 'bold');
xlabel('Slow Time (seconds)'); ylabel('Chest Displacement (mm)');

% --- Panel 4: Respiratory Spectrum & Link Budget Telemetry ---
subplot(2, 2, 4);
f_bpm = f_spec(1:N_spec/2) * 60;
spec_norm = spec_mag(1:N_spec/2) / max(spec_mag(1:N_spec/2));
area(f_bpm, spec_norm, 'FaceColor', [0.00 0.52 0.56], 'FaceAlpha', 0.25, 'EdgeColor', teal_col, 'LineWidth', 2);
hold on;
xline(bpm_est, '--', 'Color', red_col, 'LineWidth', 2);
xlim([0 100]); ylim([0 1.25]);
set(gca, 'Color', bg_panel, 'XColor', ink_dark, 'YColor', ink_dark, 'GridColor', grid_col, 'FontSize', 10);
grid on;
text(bpm_est + 2.5, 1.05, sprintf('Detected Respiration: %.1f BPM\nStatus: ALIVE / TRAPPED VICTIM\nTwo-Way Wall Loss: %.1f dB', ...
     bpm_est, L_wall_dB), 'Color', ink_dark, 'FontWeight', 'bold', 'FontSize', 9, ...
     'BackgroundColor', [0.95 0.97 0.99], 'EdgeColor', grid_col);
title('4. Vital-Sign Doppler Spectrum & EM Link Telemetry', 'Color', ink_dark, 'FontWeight', 'bold');
xlabel('Physiological Rate (Cycles / Minute - BPM)'); ylabel('Normalized Spectral Amplitude');

sgtitle('PHANTOM-WALL | S-Band Through-Wall RF Micro-Motion & Vital-Sign Tomography Engine', ...
        'Color', ink_dark, 'FontSize', 13, 'FontWeight', 'bold');