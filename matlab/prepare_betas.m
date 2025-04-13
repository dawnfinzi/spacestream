% This makes some nice prepared versions of the data:
%  ~/ext/figurefiles/nsd/datab3nativesurface_subj[01-08].mat
%  ~/ext/figurefiles/nsd/datab3nativesurface_subj[01-08]_[lh,rh,subcortex,func1pt8mm,cerebellum]_betas.hdf5
%
% In the .mat file, we have:
%   <ord> is 1 x NTRIALS with 10k indices
%   <ordU> is 1 x NIMAGES with unique 10k indices of images shown to this subject
%   <allixs> is 3 x NIMAGES indicating how to pull trials from the betas
%   <numtrialsperim> is 1 x NIMAGES with the actual number of trials contributing to each (1, 2, 3)
%   <alldata> is 1 x [lh,rh,subcortex,func1pt8mm,cerebellum] with VERTICES/VOXELS x NIMAGES with the responses
%   <d1> is A:B (indices into 1st dimension of func1mm)
%   <d2> is C:D (indices into 2nd dimension of func1mm)
%   <d3> is E:F (indices into 3rd dimension of func1mm)
%   <dsz> is the subcortex matrix size, like [20 42 34]
%   <bmii> is the logical X x Y x Z brainmask for the 1.8-mm volume preparation
%   <cd1>,<cd2>,<cd3>,<cii>,<cdsz> is for cerebellum
%   <datasizes> is 5 x 2 (hh x nimages)
%   <adjparams> has [meanmu meansigma b s2 reversionscale reversionoffset]

%% %%%%%%%%%%%%%%%%%%%%%%% PREPARE DATA (load all data (fsaverage space))
clear all

stem = '/oak/stanford/groups/kalanit/biac2/kgs/projects/Dawn/NSD';

% define
bdir = 'betas_fithrf_GLMdenoise_RR';

% load some basic stuff
a1 = load(sprintf('%s/data/nsddata/experiments/nsd/nsd_expdesign.mat', stem));
nsess = [40 40 32 30 40 32 40 30];
hemis = {'lh', 'rh'};
datanames = {'lh', 'rh'};


% loop
for subjix=8 %1:8
  
  % experimental design stuff
  ord = a1.masterordering(1:750*nsess(subjix));  % 1 x NTRIALS with 10k indices
  ordU = unique(ord);                            % 1 x IMAGES with all unique 10k images shown to this subject
  allixs = [];                                   % 3 x UNIQUEIMAGES (this sets the order)
  badtrialix = length(ord)+1;
  for qq=1:length(ordU)
    ix = find(ord==ordU(qq));                     % indices of trial numbers that ordU(qq) was shown on
    ix = vflatten(ix);                            % make a column vector
    ix = [ix; badtrialix*ones(3-length(ix),1)];   % include some dummy entries if necessary
    allixs(:,end+1) = ix;                         % record
  end
  numtrialsperim = sum(allixs~=badtrialix,1);     % 1 x UNIQUEIMAGES with number of trials (1/2/3)
  
  % now, allixs can be used to pull trials from the betas
  
  % load betas
  betadir0 =  sprintf('%s/data/nsddata_betas/ppdata/subj%02d/fsaverage/%s',stem,subjix,bdir);

  alldata =   cell(1,length(datanames));
  %adjparams = cell(1,length(datanames));
  for hh=1:length(datanames)
    alldata{hh} =   single([]);
    %adjparams{hh} = single([]);
    
    % load
    for nn=nsess(subjix):-1:1, nn
      if hh <= 2 %NOTE DF: should always be true because I've limited to just fsaverage space
        file0 = sprintf('%s/%s.betas_session%02d.mgh',betadir0,hemis{hh},nn);
        temp = squeeze(load_mgh(file0)); % 163842 x 750, double
        %temp = calczscore(temp, 2); 
        %temp = h5read(file0,'/betas',[1 1],[Inf Inf]);  % 227021 x 750, int16
      end
      %temp = single(temp)/300;  % PSC units; some may be all 0 (invalid
      %voxels) - NOTE DF: not needed, fsaverage betas still in double PSC
      alldata{hh}(:,:,nn) = temp;
      clear temp;
    end

    % proceed to normalization

    % calculate the additive adjustment factor
    mus    = mean(alldata{hh},2);   % V x 1 x session
    sigmas = std(alldata{hh},[],2); % V x 1 x session
    b = nanmean(zerodiv(mus,sigmas,NaN),3);  % V x 1
    if ~all(isfinite(b))
      warning('bad! 1');  % TURNS OUT subj06 has a few completely bad voxels on the outskirts.
    end
    clear mus sigmas

    % zscore each session and concatenate (NOTE: invalid voxels will be NaN for a given scan session)
    alldata{hh} = reshape(calczscore(alldata{hh},2),size(alldata{hh},1),[]);  % V x 750*nsess
    %alldata{hh} = reshape(alldata{hh},size(alldata{hh},1),[]);  % V x 750*nsess

    % add adjustment factor
    alldata{hh} = bsxfun(@plus,alldata{hh},b);
    
    % add a fake column
    alldata{hh}(:,end+1) = NaN;
    
    % trial-average (after this step, there could be NaNs for some images, e.g. if all reps of an image occurred in a bad session)
    alldata{hh} = squish(nanmean(reshape(alldata{hh}(:,flatten(allixs)),size(alldata{hh},1),3,[]),2),2);  % V x UNIQUEIMAGES
    
    % to make life easy, set NaNs to 0
    alldata{hh}(isnan(alldata{hh})) = 0;
    
    % record adjustment factors
    %adjparams{hh}(:,[1 2 3]) = [mean(mus,3) mean(sigmas,3) b];
  
  end
  
  
  % "re-zscore"
  for hh=1:length(alldata)
    s2 = std(alldata{hh},[],2);
    if any(s2==0)
      warning('bad! 2');  % TURNS OUT subj06 has a few completely bad voxels on the outskirts.
    end
    alldata{hh} = zerodiv(alldata{hh},s2,0,0);
    %adjparams{hh}(:,4) = s2;
  end
  
  % now, 0 is still meaningful, and the std dev of each vertex is 1 (but note below).
  % since this last step is just scaling the data, the noise ceiling idea is still valid.
  % HOWEVER, there are a few bogus cases where the entire dataset is all 0.

  % save preprocessed data
  for hh=1:length(alldata)
    outputfile = sprintf('%s/local_data/processed/organized_betas/datab3fsaverage_subj%02d_%s_betas.hdf5',stem,subjix,datanames{hh});
    delete(outputfile);
    h5create(outputfile,'/betas',size(alldata{hh}),'Datatype','single','ChunkSize',[1 size(alldata{hh},2)]);
    h5write(outputfile,'/betas',alldata{hh});
  end

end
