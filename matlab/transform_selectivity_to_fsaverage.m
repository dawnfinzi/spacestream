%% map all the selectivity contrast maps from the floc experiment
% transforms from the native surface space to the fsaverage space for each
% subject using nsd_mapdata.m (which uses nearest neighbor for this
% transformation)
fsdir = [nsd_datalocation '/freesurfer/fsaverage'];

hemis = {'lh', 'rh'};
categories = {'faces', 'places', 'bodies', 'characters', 'objects'};
subjix = 1;

for subjix = 1:8
    for h = 1:length(hemis)
        for c = 1:length(categories)
            % Prep
            sourcedata = sprintf('%s/freesurfer/subj%02d/label/%s.floc%stval.mgz',nsd_datalocation,subjix, hemis{h}, categories{c});   
            output_file = sprintf('%s.floc%stval_subj%02d.mgz',hemis{h}, categories{c},subjix); 

            % Transform
            data = nsd_mapdata(subjix,sprintf('%s.white', hemis{h}),'fsaverage',sourcedata);

            % Write out the results to an .mgz file.
            nsd_savemgz(data,sprintf('%s/label/%s',fsdir,output_file),fsdir);
        end
    end
end