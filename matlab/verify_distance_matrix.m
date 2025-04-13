%% code to plot distances for two example vertices to verify 
clear all

oak_stem = '/oak/stanford/groups/kalanit/biac2/kgs/projects/';
results_path = fullfile(oak_stem, 'Dawn/NSD/results/spacetorch/brain_figures');

distance_type = 'spherical';
if strcmp(distance_type, 'spherical')
    distance_stem = 'spherical_';
    thresh = 150;
else
    distance_stem = '';
    thresh = 100;
end

hemis = {'lh', 'rh'};
% use ministreams to subselect and save on mems/compute
roi_name = 'ministreams';
roivalsL = cvnloadmgz(sprintf('/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/data/nsddata/freesurfer/fsaverage/label/lh.%s.mgz',roi_name)); 
roiL_idx = find(roivalsL);
roivalsR = cvnloadmgz(sprintf('/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/data/nsddata/freesurfer/fsaverage/label/rh.%s.mgz',roi_name)); 
roiR_idx = find(roivalsR);

dists = struct([]);
for h = 1:length(hemis)
    dists(h).full = load([nsd_datalocation('local') '/fsaverage_space/' sprintf('%s_%s_fsavg_%sdistances.mat',roi_name, hemis{h}, distance_stem)]);
end

if strcmp(distance_type, 'spherical')
    roivalsL(roiL_idx) = dists(1).full.fullspheredists(150,:); %point to plot distances for
    roivalsR(roiR_idx) = dists(2).full.fullspheredists(500,:); %different point to plot dists for
else
    roivalsL(roiL_idx) = dists(1).full.fulldists(1,:); %point to plot distances for
    roivalsR(roiR_idx) = dists(2).full.fulldists(50,:); %different point to plot dists for
end

to_plot = [roivalsL; roivalsR];
view = to_plot(to_plot~=0);
sum(view<5)

extraopts = {'roiname',{'ministreams'},'roicolor',{'k'},'drawroinames',false, 'roiwidth', 2};
[rawimg,Lookup,rgbimg] = cvnlookup('fsaverage',10,to_plot'',[.001,thresh], jet(256), .001,[],1,extraopts);

imwrite(rgbimg,sprintf('%s/fsaverage/verify_%sdistance_calc.png',results_path, distance_stem));