%% ========================================================================
%  PHANTOM-WALL: Through-Wall S-Band FMCW Tomography & Vital-Sign Engine
%  Step 2: Coherent Focusing, Micro-Doppler AI, INT8 Quantization & C++ Export
%  Author: Muhammed Akthus C N
% =========================================================================
clear; clc; close all;

fprintf('=================================================================\n');
fprintf('  PHANTOM-WALL | Step 2: Coherent Refinement, INT8 AI & C++ Export\n');
fprintf('=================================================================\n');

if ~isfile('phantom_wall_step1_data.mat')
    error('Run step1_phantom_wall_physics.m first to generate phantom_wall_step1_data.mat');
end
load('phantom_wall_step1_data.mat'); % rng_profiles_clean, disp_est_mm, t_slow, fps, fc, lambda, L_wall_dB

%% 1. Coherent Phase-Compensated Focusing & Glitch-Free Phase Recovery
[Nr, Nf, N_ant] = size(rng_profiles_clean);
c = 3e8; B = 1.5e9; Tc = 50e-6; S = B / Tc; Fs = 25.6e6; N_fft = 2048;
range_axis = (0:N_fft-1) * (Fs * c) / (2 * S * N_fft);
r_grid = range_axis(1:Nr);
d_ant = lambda / 2;
x_ant = ((0:N_ant-1) - (N_ant-1)/2) * d_ant;

% Wall delay parameters matching Step 1
y_wall = 1.50; d_wall = 0.20; eps_r = 6.5; sigma_w = 0.05;
eps_0 = 8.854e-12; mu_0 = 4*pi*1e-7; omega = 2*pi*fc;
gamma_w = sqrt(1j * omega * mu_0 * (sigma_w + 1j * omega * eps_0 * eps_r));
v_wall = omega / imag(gamma_w);
extra_delay_m = d_wall * (c / v_wall - 1);

% Coherent Complex Backprojection across X-Y grid
x_grid = linspace(-1.5, 1.5, 121);
y_grid = linspace(0.5, 4.2, 149);
[Xg, Yg] = meshgrid(x_grid, y_grid);
img_coherent = zeros(size(Xg));

for k = 1:Nf
    frame_slice = zeros(size(Xg));
    for m = 1:N_ant
        R_map = hypot(Xg - x_ant(m), Yg);
        R_map(Yg > y_wall) = R_map(Yg > y_wall) + extra_delay_m;
        prof_k = interp1(r_grid, rng_profiles_clean(:, k, m), R_map, 'linear', 0);
        % Coherent matched-filter phase alignment
        frame_slice = frame_slice + prof_k .* exp(1j * 4*pi * R_map / lambda);
    end
    img_coherent = img_coherent + abs(frame_slice);
end

mask_behind = Yg > (y_wall + d_wall + 0.2);
[~, pk_idx] = max((img_coherent .* mask_behind), [], 'all');
[iy_c, ix_c] = ind2sub(size(Xg), pk_idx);
x_coh = x_grid(ix_c);
y_coh = y_grid(iy_c);

% Coherent beamforming on refined peak + complex I/Q smoothing (eliminates phase jumps)
z_coh = zeros(1, Nf);
for m = 1:N_ant
    R_det = hypot(x_coh - x_ant(m), y_coh) + extra_delay_m;
    [~, r_bin] = min(abs(r_grid - R_det));
    z_coh = z_coh + rng_profiles_clean(r_bin, :, m) * exp(1j * 4*pi * R_det / lambda);
end
z_smooth = movmean(z_coh, 5);
phi_clean = detrend(unwrap(angle(z_smooth)));
disp_clean_mm = (lambda / (4*pi)) * phi_clean * 1e3;
delta_R_ref = 4.5 * sin(2*pi*0.28*t_slow) + 0.6 * sin(2*pi*1.25*t_slow);
if corr(disp_clean_mm.', delta_R_ref.') < 0
    disp_clean_mm = -disp_clean_mm;
end

N_spec = 1024;
f_spec = (0:N_spec-1) * (fps / N_spec);
spec_mag = abs(fft(disp_clean_mm .* hann(Nf).', N_spec));
valid_resp = (f_spec >= 0.12) & (f_spec <= 0.60);
[~, idx_bpm] = max(spec_mag(valid_resp));
f_slice = f_spec(valid_resp);
bpm_refined = f_slice(idx_bpm) * 60;

fprintf('[Coherent Focus] Refined Victim Position: (%.2f m, %.2f m) [True: (-0.35, 2.85)]\n', x_coh, y_coh);
fprintf('[Coherent Phase] Refined Breathing Rate : %.1f BPM (Ground Truth: 16.8 BPM)\n', bpm_refined);

%% 2. Generate 4-Class Disaster Triage Micro-Doppler Dataset (400 Samples)
% Classes:
%   1: Empty Rubble / Static Clutter
%   2: Trapped Victim - Normal Breathing (12-20 BPM, 3.5-5.5 mm)
%   3: Trapped Victim - Distressed / Shallow Breathing (24-36 BPM, 1.2-2.2 mm)
%   4: Conscious Victim - Active Limb / Torso Motion (15-45 mm irregular)
rng(2026);
num_per_class = 100;
num_classes = 4;
N_total = num_per_class * num_classes;
class_names = {'Empty Rubble', 'Normal Breathing', 'Shallow/Distressed', 'Limb Motion'};

% Feature extraction: 16 normalized micro-Doppler spectral & kinematic features per sample
X_data = zeros(16, N_total);
Y_labels = zeros(1, N_total);
sample_specs = cell(1, 4); % Save 1 representative STFT spectrogram per class for visualization

win_stft = 32; hop_stft = 4; nfft_stft = 64;
idx = 0;

for c_id = 1:num_classes
    for n = 1:num_per_class
        idx = idx + 1;
        Y_labels(idx) = c_id;
        
        switch c_id
            case 1 % Empty Rubble (residual thermal + vibration noise)
                disp_sim = 0.25 * randn(1, Nf);
            case 2 % Normal Breathing
                f_b = (12 + 8*rand()) / 60;
                a_b = 3.5 + 2.0*rand();
                disp_sim = a_b * sin(2*pi*f_b*t_slow + 2*pi*rand()) + ...
                           0.5 * sin(2*pi*1.2*t_slow) + 0.35*randn(1, Nf);
            case 3 % Distressed / Shallow Rapid Breathing
                f_b = (24 + 12*rand()) / 60;
                a_b = 1.2 + 1.0*rand();
                disp_sim = a_b * sin(2*pi*f_b*t_slow + 2*pi*rand()) + ...
                           0.4 * sin(2*pi*1.6*t_slow) + 0.35*randn(1, Nf);
            case 4 % Conscious Victim Moving Limbs / Tapping
                f_m = 0.6 + 1.4*rand();
                a_m = 14.0 + 18.0*rand();
                burst = exp(-((t_slow - (2 + 8*rand())).^2) / (2 * 1.1^2));
                disp_sim = a_m * burst .* sin(2*pi*f_m*t_slow) + ...
                           3.0 * sin(2*pi*0.32*t_slow) + 0.45*randn(1, Nf);
        end
        
        % Compute baseband micro-Doppler phase signal & STFT spectrogram
        s_bb = exp(1j * (4*pi / (lambda * 1e3)) * disp_sim);
        s_bb = s_bb - mean(s_bb);
        num_frames = floor((Nf - win_stft) / hop_stft) + 1;
        stft_mat = zeros(nfft_stft, num_frames);
        for fr = 1:num_frames
            seg = s_bb((fr-1)*hop_stft + (1:win_stft)) .* hann(win_stft).';
            stft_mat(:, fr) = abs(fftshift(fft(seg, nfft_stft)));
        end
        
        if n == 1
            sample_specs{c_id} = 20*log10(stft_mat / max(stft_mat(:)) + 1e-3);
        end
        
        % Extract 16 physics-informed features (Doppler bands + envelope stats)
        doppler_prof = mean(stft_mat, 2);
        doppler_bins = interp1(1:nfft_stft, doppler_prof, linspace(1, nfft_stft, 12)).';
        vel_sig = diff(disp_sim) * fps;
        extra_feats = [rms(disp_sim); max(abs(disp_sim)); rms(vel_sig); kurtosis(disp_sim)];
        feat_vec = [doppler_bins; extra_feats];
        X_data(:, idx) = feat_vec;
    end
end

% Z-score normalization parameters (embedded directly into C++ header!)
feat_mu  = mean(X_data, 2);
feat_std = std(X_data, 0, 2) + 1e-6;
X_norm   = (X_data - feat_mu) ./ feat_std;

% Train/Test split (80% / 20%)
perm = randperm(N_total);
X_norm = X_norm(:, perm);
Y_labels = Y_labels(perm);

n_train = round(0.8 * N_total);
X_tr = X_norm(:, 1:n_train);   Y_tr = Y_labels(1:n_train);
X_te = X_norm(:, n_train+1:end); Y_te = Y_labels(n_train+1:end);

%% 3. Train Compact Edge MLP Classifier (16 -> 24 -> 12 -> 4) in FP32
% Pure vectorized training (zero toolbox dependency, deterministic convergence)
in_dim = 16; h1_dim = 24; h2_dim = 12; out_dim = 4;
W1 = randn(h1_dim, in_dim) * sqrt(2 / in_dim);   b1 = zeros(h1_dim, 1);
W2 = randn(h2_dim, h1_dim) * sqrt(2 / h1_dim);   b2 = zeros(h2_dim, 1);
W3 = randn(out_dim, h2_dim) * sqrt(2 / h2_dim);  b3 = zeros(out_dim, 1);

T_oh = full(ind2vec(Y_tr, out_dim));
lr = 0.04; epochs = 600;

for ep = 1:epochs
    % Forward pass
    Z1 = W1 * X_tr + b1; A1 = max(0, Z1);
    Z2 = W2 * A1 + b2;   A2 = max(0, Z2);
    Z3 = W3 * A2 + b3;
    Z3_shift = Z3 - max(Z3, [], 1);
    P = exp(Z3_shift) ./ sum(exp(Z3_shift), 1);
    
    % Backpropagation
    dZ3 = (P - T_oh) / n_train;
    dW3 = dZ3 * A2.'; db3 = sum(dZ3, 2);
    dA2 = W3.' * dZ3; dZ2 = dA2 .* (Z2 > 0);
    dW2 = dZ2 * A1.'; db2 = sum(dZ2, 2);
    dA1 = W2.' * dZ2; dZ1 = dA1 .* (Z1 > 0);
    dW1 = dZ1 * X_tr.'; db1 = sum(dZ1, 2);
    
    W3 = W3 - lr * dW3; b3 = b3 - lr * db3;
    W2 = W2 - lr * dW2; b2 = b2 - lr * db2;
    W1 = W1 - lr * dW1; b1 = b1 - lr * db1;
end

% Evaluate FP32 Accuracy
fp32_infer = @(X) W3 * max(0, W2 * max(0, W1 * X + b1) + b2) + b3;
logits_fp32 = fp32_infer(X_te);
[~, pred_fp32] = max(logits_fp32, [], 1);
acc_fp32 = mean(pred_fp32 == Y_te) * 100;

%% 4. Symmetric Post-Training Quantization (FP32 -> INT8) & SQNR Audit
quantize_sym = @(W) deal(int8(max(-127, min(127, round(W / (max(abs(W(:))) / 127 + 1e-12))))), ...
                         max(abs(W(:))) / 127 + 1e-12);

[W1_q, sW1] = quantize_sym(W1); [b1_q, sb1] = quantize_sym(b1);
[W2_q, sW2] = quantize_sym(W2); [b2_q, sb2] = quantize_sym(b2);
[W3_q, sW3] = quantize_sym(W3); [b3_q, sb3] = quantize_sym(b3);

% Dequantized INT8 inference check
W1_dq = double(W1_q) * sW1; b1_dq = double(b1_q) * sb1;
W2_dq = double(W2_q) * sW2; b2_dq = double(b2_q) * sb2;
W3_dq = double(W3_q) * sW3; b3_dq = double(b3_q) * sb3;

logits_int8 = W3_dq * max(0, W2_dq * max(0, W1_dq * X_te + b1_dq) + b2_dq) + b3_dq;
[~, pred_int8] = max(logits_int8, [], 1);
acc_int8 = mean(pred_int8 == Y_te) * 100;

% Compute Quantization SQNR (dB) and Memory Footprint
w_all_fp32 = [W1(:); b1(:); W2(:); b2(:); W3(:); b3(:)];
w_all_dq   = [W1_dq(:); b1_dq(:); W2_dq(:); b2_dq(:); W3_dq(:); b3_dq(:)];
sqnr_dB    = 10 * log10(sum(w_all_fp32.^2) / sum((w_all_fp32 - w_all_dq).^2));

num_params = numel(w_all_fp32);
bytes_fp32 = num_params * 4;
bytes_int8 = num_params * 1;

fprintf('[Edge AI] FP32 Test Accuracy : %.1f%% (%d bytes)\n', acc_fp32, bytes_fp32);
fprintf('[Edge AI] INT8 Test Accuracy : %.1f%% (%d bytes | %.1fx compression)\n', ...
        acc_int8, bytes_int8, bytes_fp32 / bytes_int8);
fprintf('[Edge AI] Quantization SQNR  : %.2f dB\n', sqnr_dB);

%% 5. Export Standalone Zero-Dependency C++ Header (phantom_wall_int8_model.h)
hdr_file = 'phantom_wall_int8_model.h';
fid = fopen(hdr_file, 'w');
fprintf(fid, '// ==========================================================================\n');
fprintf(fid, '// PHANTOM-WALL: Quantized INT8 Through-Wall Victim Triage Inference Engine\n');
fprintf(fid, '// Auto-Generated by MATLAB | FP32->INT8 SQNR: %.2f dB | Footprint: %d Bytes\n', sqnr_dB, bytes_int8);
fprintf(fid, '// ==========================================================================\n');
fprintf(fid, '#ifndef PHANTOM_WALL_INT8_MODEL_H\n#define PHANTOM_WALL_INT8_MODEL_H\n\n#include <stdint.h>\n\n');
fprintf(fid, '#define PW_IN_DIM  %d\n#define PW_H1_DIM  %d\n#define PW_H2_DIM  %d\n#define PW_OUT_DIM %d\n\n', ...
        in_dim, h1_dim, h2_dim, out_dim);

write_f32_arr = @(name, arr) fprintf(fid, 'static const float %s[%d] = {%s};\n', ...
    name, numel(arr), strjoin(arrayfun(@(v) sprintf('%.6ff', v), arr(:), 'UniformOutput', false), ', '));
write_i8_arr  = @(name, arr) fprintf(fid, 'static const int8_t %s[%d] = {%s};\n', ...
    name, numel(arr), strjoin(arrayfun(@(v) sprintf('%d', v), arr(:), 'UniformOutput', false), ', '));

write_f32_arr('PW_FEAT_MEAN', feat_mu);
write_f32_arr('PW_FEAT_STD',  feat_std);
fprintf(fid, '\nstatic const float PW_SW1 = %.8ff, PW_SB1 = %.8ff;\n', sW1, sb1);
fprintf(fid, 'static const float PW_SW2 = %.8ff, PW_SB2 = %.8ff;\n', sW2, sb2);
fprintf(fid, 'static const float PW_SW3 = %.8ff, PW_SB3 = %.8ff;\n\n', sW3, sb3);

write_i8_arr('PW_W1_INT8', W1_q.'); write_i8_arr('PW_B1_INT8', b1_q);
write_i8_arr('PW_W2_INT8', W2_q.'); write_i8_arr('PW_B2_INT8', b2_q);
write_i8_arr('PW_W3_INT8', W3_q.'); write_i8_arr('PW_B3_INT8', b3_q);

fprintf(fid, '\nstatic inline int phantom_wall_predict_int8(const float raw_features[PW_IN_DIM], float out_scores[PW_OUT_DIM]) {\n');
fprintf(fid, '    float x[PW_IN_DIM], h1[PW_H1_DIM], h2[PW_H2_DIM];\n');
fprintf(fid, '    for (int i = 0; i < PW_IN_DIM; ++i) x[i] = (raw_features[i] - PW_FEAT_MEAN[i]) / PW_FEAT_STD[i];\n');
fprintf(fid, '    for (int j = 0; j < PW_H1_DIM; ++j) {\n');
fprintf(fid, '        float acc = (float)PW_B1_INT8[j] * PW_SB1;\n');
fprintf(fid, '        for (int i = 0; i < PW_IN_DIM; ++i) acc += ((float)PW_W1_INT8[j * PW_IN_DIM + i] * PW_SW1) * x[i];\n');
fprintf(fid, '        h1[j] = acc > 0.0f ? acc : 0.0f;\n    }\n');
fprintf(fid, '    for (int j = 0; j < PW_H2_DIM; ++j) {\n');
fprintf(fid, '        float acc = (float)PW_B2_INT8[j] * PW_SB2;\n');
fprintf(fid, '        for (int i = 0; i < PW_H1_DIM; ++i) acc += ((float)PW_W2_INT8[j * PW_H1_DIM + i] * PW_SW2) * h1[i];\n');
fprintf(fid, '        h2[j] = acc > 0.0f ? acc : 0.0f;\n    }\n');
fprintf(fid, '    int best_cls = 0;\n');
fprintf(fid, '    for (int j = 0; j < PW_OUT_DIM; ++j) {\n');
fprintf(fid, '        float acc = (float)PW_B3_INT8[j] * PW_SB3;\n');
fprintf(fid, '        for (int i = 0; i < PW_H2_DIM; ++i) acc += ((float)PW_W3_INT8[j * PW_H2_DIM + i] * PW_SW3) * h2[i];\n');
fprintf(fid, '        out_scores[j] = acc;\n        if (acc > out_scores[best_cls]) best_cls = j;\n    }\n');
fprintf(fid, '    return best_cls;\n}\n\n#endif // PHANTOM_WALL_INT8_MODEL_H\n');
fclose(fid);
fprintf('[C++ Export] Saved standalone header: %s (%d Bytes INT8 weights)\n', hdr_file, bytes_int8);

%% 6. Daylight Industrial Micro-Doppler & INT8 Edge Audit Dashboard
fig2 = figure('Name', 'PHANTOM-WALL | Step 2: Micro-Doppler AI & INT8 Edge Export', ...
              'Color', [0.97 0.98 0.99], 'Position', [100 70 1340 780]);

bg_panel = [1.00 1.00 1.00];
ink_dark = [0.10 0.14 0.20];
teal_col = [0.00 0.52 0.56];
amber_col= [0.85 0.42 0.05];
red_col  = [0.80 0.14 0.20];
grid_col = [0.85 0.88 0.91];

% --- Panel 1: Refined Glitch-Free Chest Phase Waveform ---
subplot(2, 3, 1);
plot(t_slow, delta_R_ref, '--', 'Color', [0.45 0.50 0.58], 'LineWidth', 1.6); hold on;
plot(t_slow, disp_clean_mm, '-', 'Color', teal_col, 'LineWidth', 2.2);
set(gca, 'Color', bg_panel, 'XColor', ink_dark, 'YColor', ink_dark, 'GridColor', grid_col, 'FontSize', 9.5);
grid on; ylim([-6.5 6.5]);
legend({'True Chest (4.5 mm)', 'Coherent Unwrapped Phase'}, 'Location', 'northeast');
title(sprintf('1. Refined Coherent Phase (%.1f BPM)', bpm_refined), 'Color', ink_dark, 'FontWeight', 'bold');
xlabel('Slow Time (s)'); ylabel('Displacement (mm)');

% --- Panels 2 & 3: Micro-Doppler STFT Signatures (Normal vs Distressed/Limb) ---
t_stft = linspace(0, 12, size(sample_specs{2}, 2));
f_dopp = linspace(-fps/2, fps/2, nfft_stft);

subplot(2, 3, 2);
imagesc(t_stft, f_dopp, sample_specs{2}, [-30 0]);
set(gca, 'YDir', 'normal', 'XColor', ink_dark, 'YColor', ink_dark, 'FontSize', 9.5);
colorbar; ylim([-4 4]);
title('2. STFT Micro-Doppler: Normal Breathing', 'Color', ink_dark, 'FontWeight', 'bold');
xlabel('Time (s)'); ylabel('Doppler Shift (Hz)');

subplot(2, 3, 3);
imagesc(t_stft, f_dopp, sample_specs{4}, [-30 0]);
set(gca, 'YDir', 'normal', 'XColor', ink_dark, 'YColor', ink_dark, 'FontSize', 9.5);
colorbar; ylim([-6 6]);
title('3. STFT Micro-Doppler: Conscious Limb Motion', 'Color', ink_dark, 'FontWeight', 'bold');
xlabel('Time (s)'); ylabel('Doppler Shift (Hz)');

% --- Panel 4: INT8 Confusion Matrix (80 Test Samples) ---
subplot(2, 3, 4);
cm = zeros(4, 4);
for i = 1:length(Y_te)
    cm(Y_te(i), pred_int8(i)) = cm(Y_te(i), pred_int8(i)) + 1;
end
imagesc(cm); colormap(gca, sky); colorbar;
set(gca, 'XTick', 1:4, 'XTickLabel', {'Empty', 'Normal', 'Shallow', 'Motion'}, ...
         'YTick', 1:4, 'YTickLabel', {'Empty', 'Normal', 'Shallow', 'Motion'}, ...
         'XColor', ink_dark, 'YColor', ink_dark, 'FontSize', 9);
for r = 1:4
    for c_i = 1:4
        text(c_i, r, num2str(cm(r, c_i)), 'HorizontalAlignment', 'center', ...
             'Color', ink_dark, 'FontWeight', 'bold', 'FontSize', 11);
    end
end
title(sprintf('4. INT8 Edge Confusion Matrix (%.1f%% Acc)', acc_int8), 'Color', ink_dark, 'FontWeight', 'bold');
xlabel('Predicted Triage State'); ylabel('True Triage State');

% --- Panel 5: FP32 vs INT8 Weight Quantization Fidelity ---
subplot(2, 3, 5);
scatter(w_all_fp32, w_all_dq, 18, teal_col, 'filled', 'MarkerFaceAlpha', 0.65); hold on;
lim_w = max(abs(w_all_fp32)) * 1.1;
plot([-lim_w lim_w], [-lim_w lim_w], '--', 'Color', red_col, 'LineWidth', 1.5);
set(gca, 'Color', bg_panel, 'XColor', ink_dark, 'YColor', ink_dark, 'GridColor', grid_col, 'FontSize', 9.5);
grid on; xlim([-lim_w lim_w]); ylim([-lim_w lim_w]);
title(sprintf('5. FP32 vs INT8 Weights (SQNR = %.1f dB)', sqnr_dB), 'Color', ink_dark, 'FontWeight', 'bold');
xlabel('Original FP32 Weight'); ylabel('Dequantized INT8 Weight');

% --- Panel 6: Edge Deployment Footprint & Telemetry Summary ---
subplot(2, 3, 6);
bar_vals = [bytes_fp32, bytes_int8];
b_h = bar(categorical({'FP32 Model', 'INT8 C++ Header'}), bar_vals, 0.5);
b_h.FaceColor = 'flat';
b_h.CData(1, :) = amber_col;
b_h.CData(2, :) = teal_col;
set(gca, 'Color', bg_panel, 'XColor', ink_dark, 'YColor', ink_dark, 'GridColor', grid_col, 'FontSize', 9.5);
grid on; ylim([0 bytes_fp32 * 1.25]);
text(1, bytes_fp32 + 120, sprintf('%d B', bytes_fp32), 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', ink_dark);
text(2, bytes_int8 + 120, sprintf('%d B (4.0x Smaller)', bytes_int8), 'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', teal_col);
title('6. Edge Memory Footprint & C++ Export', 'Color', ink_dark, 'FontWeight', 'bold');
ylabel('Model Size (Bytes)');

sgtitle('PHANTOM-WALL | Step 2: Micro-Doppler Triage Classifier, INT8 Quantization & C++ Header Export', ...
        'Color', ink_dark, 'FontSize', 13, 'FontWeight', 'bold');