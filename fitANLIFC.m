function fuzzy_model = fitANLIFC(X, Y, numRules, options)
% fitANLIFC - Reference implementation of ANLI-FC.
%
% This code focuses on:
%   - the decomposition of the β-weighted MAUC surrogate loss into four components
%     (correct / half-correct / wrong terms),
%   - the corresponding gradient accumulation w.r.t. rule consequents W
%     and fuzzy antecedents (centers C and log-sigmas P).
%
% It intentionally omits:
%   - Initilization or clustering-based pre-training for C and sigmas,
%   - the specific optimization algorithm used to update W, C, and sigmas.
%
% Inputs:
%   X        : N x D data matrix
%   Y        : N x 1 labels in {1,...,M}
%   numRules : K, number of fuzzy rules
%   options  : struct with fields
%       .beta    : noise level β in the paper (default: 0.10)
%       .maxIter : number of passes to accumulate loss/gradient (default: 1)
%       .centers : optional K x D initial centers
%       .sigmas  : optional K x D initial sigmas
%       .W_init  : optional K x M initial consequents
%
% (C, sigma) can be initialized by a clustering-based
% method (e.g., FCM/k-means) and then jointly refined with W using
% gradient-based optimization of the β-weighted MAUC surrogate.
% For brevity and IP protection, the initilization and optimizer details
% are omitted in this reference implementation.

    if ~exist('options','var') || ~isstruct(options)
        options = struct();
    end

    beta_noise = gopt(options, 'beta',    0.10);
    maxIter    = gopt(options, 'maxIter', 1);

    [N, D] = size(X);
    classes = unique(Y(:))';
    M = numel(classes);
    K = numRules;

    % ---------- Antecedent init: C, P ----------
    centers = zeros(K, D);
    sigmas  = zeros(K, D);
    % logits for sigma for positivity
    P = log(max(sigmas,1e-8));

    % ---------- Consequent init: W on simplex ----------
    W = rand(K,M);
    W = W ./ sum(W,2);

    % ---------- Precompute class indices ----------
    classIdx = cell(1, M);
    for c = 1:M
        classIdx{c} = find(Y == classes(c));
    end

    for iter = 1:maxIter %#ok<FORUM>
        % refresh sigmas from P (positivity)
        sigmas = exp(P);

        % gradient accumulators
        GW = zeros(K, M);   % dL/dW
        GC = zeros(K, D);   % dL/dC
        GP = zeros(K, D);   % dL/dP   (P = log sigma)
        totalLoss = 0;

        % loop over class pairs (one-vs-one)
        for i = 1:(M-1)
            for j = (i+1):M
                idx_i = classIdx{i};
                idx_j = classIdx{j};
                if isempty(idx_i) || isempty(idx_j), continue; end

                Xi = X(idx_i, :);
                Xj = X(idx_j, :);

                MU_i = mu_batch(Xi, centers, sigmas);   % |Ci| x K
                MU_j = mu_batch(Xj, centers, sigmas);   % |Cj| x K

                % β-weighted coefficients for three groups
                w0 = (1 - beta_noise)^2;
                w1 = beta_noise * (1 - beta_noise) / (M - 1);
                w2 = (beta_noise^2) / ((M - 1)^2);

                % ---------- correct terms ----------
                % direction i (positive) vs j (negative), for class i
                [gWi, gMU_i_pos, gMU_j_neg, Li] = grad_pair_dir(MU_i, MU_j, W(:, i));
                GW(:, i) = GW(:, i) + w0 * gWi;
                [dC_ij, dP_ij] = acc_mu_to_CP(Xi, Xj, gMU_i_pos, gMU_j_neg, MU_i, MU_j, centers, sigmas);
                GC = GC + w0 * dC_ij;
                GP = GP + w0 * dP_ij;
                totalLoss = totalLoss + w0 * Li;

                % direction j (positive) vs i (negative), for class j
                [gWj, gMU_j_pos, gMU_i_neg, Lj] = grad_pair_dir(MU_j, MU_i, W(:, j));
                GW(:, j) = GW(:, j) + w0 * gWj;
                [dC_ji, dP_ji] = acc_mu_to_CP(Xj, Xi, gMU_j_pos, gMU_i_neg, MU_j, MU_i, centers, sigmas);
                GC = GC + w0 * dC_ji;
                GP = GP + w0 * dP_ji;
                totalLoss = totalLoss + w0 * Lj;

                % indices of "other" classes for i and j
                others_i = setdiff(1:M, i);
                others_j = setdiff(1:M, j);

                % ---------- half-correct terms ----------
                for u = others_i
                    [gWu, gMU_i_pos_u, gMU_j_neg_u, Lu] = grad_pair_dir(MU_i, MU_j, W(:, u));
                    GW(:, u) = GW(:, u) + w1 * gWu;
                    [dC_u, dP_u] = acc_mu_to_CP(Xi, Xj, gMU_i_pos_u, gMU_j_neg_u, MU_i, MU_j, centers, sigmas);
                    GC = GC + w1 * dC_u;
                    GP = GP + w1 * dP_u;
                    totalLoss = totalLoss + w1 * Lu;
                end
                for v = others_j
                    [gWv, gMU_j_pos_v, gMU_i_neg_v, Lv] = grad_pair_dir(MU_j, MU_i, W(:, v));
                    GW(:, v) = GW(:, v) + w1 * gWv;
                    [dC_v, dP_v] = acc_mu_to_CP(Xj, Xi, gMU_j_pos_v, gMU_i_neg_v, MU_j, MU_i, centers, sigmas);
                    GC = GC + w1 * dC_v;
                    GP = GP + w1 * dP_v;
                    totalLoss = totalLoss + w1 * Lv;
                end

                % ---------- wrong terms ----------
                for u = others_i
                    for v = others_j
                        [gWu2, gMU_i_pos_u2, gMU_j_neg_u2, Lu2] = grad_pair_dir(MU_i, MU_j, W(:, u));
                        GW(:, u) = GW(:, u) + w2 * gWu2;
                        [dC_u2, dP_u2] = acc_mu_to_CP(Xi, Xj, gMU_i_pos_u2, gMU_j_neg_u2, MU_i, MU_j, centers, sigmas);
                        GC = GC + w2 * dC_u2;
                        GP = GP + w2 * dP_u2;
                        totalLoss = totalLoss + w2 * Lu2;

                        [gWv2, gMU_j_pos_v2, gMU_i_neg_v2, Lv2] = grad_pair_dir(MU_j, MU_i, W(:, v));
                        GW(:, v) = GW(:, v) + w2 * gWv2;
                        [dC_v2, dP_v2] = acc_mu_to_CP(Xj, Xi, gMU_j_pos_v2, gMU_i_neg_v2, MU_j, MU_i, centers, sigmas);
                        GC = GC + w2 * dC_v2;
                        GP = GP + w2 * dP_v2;
                        totalLoss = totalLoss + w2 * Lv2;
                    end
                end
            end
        end

        % At this point:
        %   totalLoss  : β-weighted MAUC surrogate (up to a constant factor)
        %   GW         : d(totalLoss)/dW
        %   GC         : d(totalLoss)/dC
        %   GP         : d(totalLoss)/dP, with P = log(sigmas)
        %
        % Users can plug in any gradient-based optimizer here to update
        % W, centers, and P (thus sigmas). For example:
        %
        %   [W, centers, P] = UpdateParameters(W, centers, P, GW, GC, GP, ...);
        %
        % This reference implementation intentionally leaves W, centers,
        % and P unchanged across iterations.

    end

    sigmas = exp(P);

    fuzzy_model.centers     = centers;
    fuzzy_model.sigmas      = sigmas;
    fuzzy_model.ruleWeights = W;
    fuzzy_model.numRules    = K;
    fuzzy_model.numClasses  = M;
    fuzzy_model.membershipFun = @(x) mu_batch(x, centers, sigmas);
end

% ===== helper: safe option get =====
function v = gopt(s, name, default)
    if ~isstruct(s) || ~isfield(s, name) || isempty(s.(name))
        v = default;
    else
        v = s.(name);
    end
end

% ===== Gaussian membership, per rule and feature =====
function MU = mu_batch(Xb, centers, sigmas)
    [Nb, ~] = size(Xb);
    K = size(centers, 1);
    MU = zeros(Nb, K);
    eps2 = 1e-12;
    for k = 1:K
        dif2 = bsxfun(@minus, Xb, centers(k, :)).^2;   % Nb x D
        den  = 2 * (sigmas(k, :).^2) + eps2;          % 1 x D
        MU(:, k) = exp(-sum(bsxfun(@rdivide, dif2, den), 2));
    end
    MU = max(MU, 1e-16);
end

% ===== pairwise surrogate and gradients for one direction =====
% Inputs:
%   MU_pos : nPos x K membership of positive class
%   MU_neg : nNeg x K membership of negative class
%   Wm     : K x 1 consequent weights for the current class
%
% Outputs:
%   gWm    : K x 1 gradient dL/dWm
%   gMU_pos: nPos x K gradient dL/d(MU_pos)
%   gMU_neg: nNeg x K gradient dL/d(MU_neg)
%   L      : scalar loss for this class-direction
%
% Surrogate:
%   For each pair (p, n), with scores s_pos, s_neg:
%       gap = 1 - (s_pos - s_neg)
%       if gap > 0: contribute gap^2
%
% L is the average over all pairs with gap > 0.
function [gWm, gMU_pos, gMU_neg, L] = grad_pair_dir(MU_pos, MU_neg, Wm)
    nPos = size(MU_pos, 1);
    nNeg = size(MU_neg, 1);
    K    = size(MU_pos, 2);

    gWm     = zeros(K, 1);
    gMU_pos = zeros(nPos, K);
    gMU_neg = zeros(nNeg, K);
    sumLoss = 0;
    cnt     = 0;

    if nPos == 0 || nNeg == 0
        L = 0;
        return;
    end

    s_pos = MU_pos * Wm;   % nPos x 1
    s_neg = MU_neg * Wm;   % nNeg x 1

    for p = 1:nPos
        sp = s_pos(p);
        for n = 1:nNeg
            gap = 1 - (sp - s_neg(n));
            if gap <= 0
                continue;
            end
            wgt = 2 * gap;          % d(gap^2)/d(gap)
            sumLoss = sumLoss + gap^2;
            cnt     = cnt + 1;

            diff_mu = MU_pos(p, :)' - MU_neg(n, :)';    % K x 1
            gWm     = gWm - wgt * diff_mu;

            % chain rule back to membership degrees
            gMU_pos(p, :) = gMU_pos(p, :) - wgt * (Wm.');
            gMU_neg(n, :) = gMU_neg(n, :) + wgt * (Wm.');
        end
    end

    if cnt > 0
        L = sumLoss / cnt;
    else
        L = 0;
    end
end

% ===== accumulate dC,dP from dMU (chain rule through μ) =====
%   μ_k(x) = exp( - sum_d (x_d - c_kd)^2 / (2 σ_kd^2) )
% We differentiate w.r.t. centers C and P = log σ.
function [dC, dP] = acc_mu_to_CP(Xpos, Xneg, gMU_pos, gMU_neg, MU_pos, MU_neg, centers, sigmas)
    K = size(centers, 1);
    D = size(centers, 2);
    dC = zeros(K, D);
    dP = zeros(K, D);  % P = log σ

    eps2 = 1e-12;

    % positive block
    for k = 1:K
        tmp = (gMU_pos(:, k) .* MU_pos(:, k));      % s_pos x 1
        dif = bsxfun(@minus, Xpos, centers(k, :)); % s_pos x D
        den = (sigmas(k, :).^2 + eps2);            % 1 x D

        dC(k, :) = dC(k, :) + sum( bsxfun(@times, tmp, bsxfun(@rdivide, dif, den)), 1 );
        dP(k, :) = dP(k, :) + sum( bsxfun(@times, tmp, bsxfun(@rdivide, dif.^2, den)), 1 );
    end

    % negative block
    for k = 1:K
        tmp = (gMU_neg(:, k) .* MU_neg(:, k));      % s_neg x 1
        dif = bsxfun(@minus, Xneg, centers(k, :)); % s_neg x D
        den = (sigmas(k, :).^2 + eps2);

        dC(k, :) = dC(k, :) + sum( bsxfun(@times, tmp, bsxfun(@rdivide, dif, den)), 1 );
        dP(k, :) = dP(k, :) + sum( bsxfun(@times, tmp, bsxfun(@rdivide, dif.^2, den)), 1 );
    end
end
