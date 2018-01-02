% Copyright (C) 2013
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} plot_and_print ()
%
% Produce plots and print for manuscript.
% @end deftypefn
%
function plot_and_print ()
    figDir = 'figs';
    ps = [5001:20000];

    mkdir(figDir);
    vars = {
        'phi_v';
        'sigma_v';
        'mu_y';
        'sigma_y';
    };
    priors = {
        {0.0; 1.0};
        {0.0; 1.0};
        {-0.1; 0.1};
        {0.0; 1.0};
    };

    % PMMH estimates
    for i = 1:length(vars)
        subplot(2, 2, i);
        bi_hist('results/posterior.nc', vars{i}, [], ps);
        hold on;
        bi_plot_prior(linspace(axis()(1), axis()(2), 500), @unifpdf, ...
                      priors{i});
        hold off;
        legend({'posterior'; 'prior'});
        legend('boxoff');
        grid on;
        xlabel(nice_name(vars{i}));
        ylabel(sprintf('p(%s)', nice_name(vars{i})));
    end
    saveas (figure(1), sprintf('%s/posterior.pdf', figDir));

    % SMC^2 estimates
    for i = 1:length(vars)
        subplot(2, 2, i);
        bi_hist('results/posterior_sir.nc', vars{i});
        hold on;
        bi_plot_prior(linspace(axis()(1), axis()(2), 500), @unifpdf, ...
                      priors{i});
        hold off;
        legend({'posterior'; 'prior'});
        legend('boxoff');
        grid on;
        xlabel(nice_name(vars{i}));
        ylabel(sprintf('p(%s)', nice_name(vars{i})));
    end
    saveas (figure(1), sprintf('%s/posterior_sir.pdf', figDir));
end
