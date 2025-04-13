%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creates a matrix with the pairwise distances (in mm) on the fsaverage sphere
% between each vertex in an ROI

clear all

hemis = {'lh', 'rh'};

% use ministreams to subselect and save on mems/compute
roi_name = 'ministreams';
roivalsL = cvnloadmgz(sprintf('/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/data/nsddata/freesurfer/fsaverage/label/lh.%s.mgz',roi_name)); 
roiL_idx = find(roivalsL);
roivalsR = cvnloadmgz(sprintf('/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD/data/nsddata/freesurfer/fsaverage/label/rh.%s.mgz',roi_name)); 
roiR_idx = find(roivalsR);

% get the coordinates of the fsaverage vertices
[surfL,surfR] = cvnreadsurface('fsaverage',{'lh' 'rh'},'sphere','orig');

r = 100; %radius = 100

for h = 1:length(hemis)
    if strcmp(hemis{h}, 'lh')
        surf = surfL;
        roivals = roivalsL;
        roi_idx = roiL_idx;
    elseif strcmp(hemis{h}, 'rh')
        surf = surfR;
        roivals = roivalsR;
        roi_idx = roiR_idx;
    end
    
    % prepare coordinates as 4 x V
    XYZ = [surf.vertices ones(size(surf.vertices,1),1)]';
    fulldists = zeros(length(roi_idx));
    fullspheredists = zeros(length(roi_idx));
    for i = 1:length(roi_idx)
        ix = roi_idx(i);

        % get 3D coordinates
        coord = surf.vertices(ix,:);  % 1 x 3

        % figure out rotation matrix
        rotmatrix = xyzrotatetoz(coord);

        % rotate all vertices so that the vertex is along z+ axis
        XYZ0 = rotmatrix*XYZ;
        % (Note to self: intuitively, this rotates everything so that your
        % current vertex is basically at the "North Pole" of this image 
        % https://en.wikipedia.org/wiki/Spherical_coordinate_system#/media/File:Spherical_coordinate_system.svg

        dists = sqrt((XYZ0(1,:).^2 + XYZ0(2,:).^2));
        % mask out vertices below the equator
        dists(XYZ0(3,:)<0) = NaN;
        
        roi_dists = dists(roi_idx);

        fulldists(i,:) = roi_dists;
        
        % spherical distance: d(a,b) = r*arccos((a*b)/r^2))
        ab = coord*XYZ(1:3,:);
        sphere_dists = r * acos(ab/(r^2));
        roi_sphere_dists = sphere_dists(roi_idx);
        cast_sphere_dists = nan(1,length(roi_idx));
        for xx = 1:length(roi_idx)
            if isreal(roi_sphere_dists(xx))
                cast_sphere_dists(xx) = roi_sphere_dists(xx);
            end
        end
        fullspheredists(i,:) = cast_sphere_dists;
    end
    
    save([nsd_datalocation('local') '/fsaverage_space/' sprintf('%s_%s_fsavg_spherical_distances.mat',roi_name, hemis{h})],'fullspheredists', '-v7.3');    
end

% outputfile = [nsd_datalocation('local') '/fsaverage_space/' sprintf('%s_lh_fsavg_distances.hdf5',roi_name)];
% delete(outputfile);
% h5create(outputfile,'/distances',size(fulldistsL),'Datatype','double');
% h5write(outputfile,'/distances',fulldistsL);