data_index = 0

C0 = load(['/share/home/kma/real_1/data/C_0/C_0', '(', num2str(data_index), ')', '.mat']).value;
segment = load(['/share/home/kma/real_1/data/segment/segment_swr_slicer', '(', num2str(data_index), ')', '.mat']).value;
disp(size(segment));
if data_index == 0
    dt     = [ones(30,1)*2; ones(12,1)*10; ones(6,1)*30; ones(12,1)*120; ones(6,1)*300];
    z_size = 180;
    x_size = 1;
    y_size = 150;
end
if data_index == 1
    dt     = [ones(10,1)*2; ones(8,1)*5; ones(8,1)*15; ones(6,1)*30; ones(54,1)*60];
    z_size = 673;
    x_size = 192;
    y_size = 192;
end
if data_index == 2
    dt     = [ones(10,1)*2; ones(8,1)*5; ones(8,1)*15; ones(6,1)*30; ones(6,1)*60; ones(24,1)*120];
    z_size = 673;
    x_size = 192;
    y_size = 192;
end
if data_index == 3
    dt     = [ones(10,1)*3; ones(22,1)*15; ones(9,1)*60; ones(30,1)*90];
    z_size = 673;
    x_size = 192;
    y_size = 192;
end

opt.ScanTime = [cumsum([0; dt(1:end-1)]), cumsum(dt)];
t = mean(opt.ScanTime,2) / 60.;
numfrm = length(t)
opt.Decay = log(2)/109.8;
opt.TimeStep = 1.0;
opt.PbrParam = [1.0 0 0];
dt_plot=(opt.ScanTime(:,1)+ opt.ScanTime(:,2))./2;

opt.Decay      = log(2)/109.8;
opt.TimeStep   = 1.0; 
opt.MaxIter    = 50;
opt.LowerBound = [0 0 0 0 0];
opt.UpperBound = [1 10 10 10 10];
opt.PrmSens    = [1 1 1 1 0];
opt.Initials   = [0.01 0.01 0.01 0.01 0.0]';
weight_factor = (dt.*exp(-log(2)/109.8*t/60)).^1;%ones(length(t));% 
if weight_factor == ones(length(t))
    method = 'OLS'
else
    method = 'WLS'
end
weight_factor = weight_factor/sum(weight_factor);

p=genpath('DIRECT_v0.1');
addpath(p);

datestr(now)
pi = zeros(z_size, x_size, y_size, 5);
C_T_pre = zeros(z_size, x_size, y_size, numfrm);
for rp_index = 0: 19 
    for z = 1: z_size
        if mod(z, 100) == 0
            z
        end
        imgs = load(['/share/home/kma/real_1/result_guc_group/UCD-GUC-2.0.0(ks_1=1)_bootstrap/imgs/rp','(', num2str(rp_index), ')_','imgs_mat', '(', num2str(data_index), ')/', 'z_', num2str(z), '.mat']).value;
        
        for x = 1: x_size
            for y = 1: y_size
                if segment(z, x, y) == 1
                    CT=imgs(:, x, y);
                    [kfits_feng_input{1}, cfits_feng_input{1}] = kfit_2t5p(CT, C0, opt.ScanTime, opt.Initials, opt, weight_factor);
                    pi(z, x, y, :) = kfits_feng_input{1, 1};
                    C_T_pre(z, x, y, :) = cfits_feng_input{1};
                end
            end
        end
    end
    datestr(now)
    save(['/share/home/kma/real_1/result_guc_group/UCD-GUC-2.0.0(ks_1=1)_bootstrap/result/', 'pi_', method, '_data_index_', num2str(data_index), '_epoch_', num2str(opt.MaxIter),'rp',num2str(rp_index), '.mat'], 'pi');
    save(['/share/home/kma/real_1/result_guc_group/UCD-GUC-2.0.0(ks_1=1)_bootstrap/result/', 'CT_pre_', method, '_data_index_', num2str(data_index), '_epoch_', num2str(opt.MaxIter),'rp',num2str(rp_index), '.mat'], 'C_T_pre', '-v7.3');
end