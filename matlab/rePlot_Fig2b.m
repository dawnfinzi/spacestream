%% replot figure 2b by ROI
clear all
close all
ROI={"Dorsal", "Lateral","Ventral"}

%% read dataTables
Fig2b=readtable("Fig2b_dataFrame.csv");
Fig2b_noiseCeiling=readtable("Fig2b_noiseCeiling.csv");
%% get all models
% Specify column names and types
disp(Fig2b.Properties.VariableNames)
allmodels =unique(Fig2b.model_type)
allmodels=allmodels([1 2 4 3])% re-order models so plots make sense
nmodels=length(allmodels);

Dorsal_i=find(Fig2b.ROIS=="Dorsal");
Dorsal=Fig2b(Dorsal_i,:);
Lateral_i=find(Fig2b.ROIS=="Lateral");
Lateral=Fig2b(Lateral_i,:);
Ventral_i=find(Fig2b.ROIS=="Ventral");
Ventral=Fig2b(Ventral_i,:);

%%
for model=1:nmodels
    % dorsal
    di=find(strcmp(Dorsal.model_type,allmodels(model)));
    dlh=find(Dorsal.hemi=="lh");
    drh=find(Dorsal.hemi=="rh");
    d_all_lh=intersect(di,dlh);
    d_all_rh=intersect(di,drh);
    mean_d(model)=mean(Dorsal.result(di));
    sd_d(model)=std(Dorsal.result(di));
    Dorsal_i_lh(:,model)=Dorsal.result(d_all_lh);
    Dorsal_i_rh(:,model)=Dorsal.result(d_all_rh);
    Dorsal_sum(model).name=allmodels(model);
    Dorsal_sum(model).mean=mean_d(model);
    Dorsal_sum(model).sd=sd_d(model);
    Dorsal_sum(model).lh=Dorsal.result(d_all_lh);
    Dorsal_sum(model).rh=Dorsal.result(d_all_rh);
    
    %lateral
    li=find(strcmp(Lateral.model_type,allmodels(model)));
    llh=find(Lateral.hemi=="lh");
    lrh=find(Lateral.hemi=="rh");
    l_all_lh=intersect(li,llh);
    l_all_rh=intersect(li,lrh);
    mean_l(model)=mean(Lateral.result(li));
    sd_l(model)=std(Lateral.result(li));
    Lateral_i_lh(:,model)=Lateral.result(l_all_lh);
    Lateral_i_rh(:,model)=Lateral.result(l_all_rh);
    Lateral_sum(model).name=allmodels(model);
    Lateral_sum(model).mean=mean_l(model);
    Lateral_sum(model).sd=sd_l(model);
    Lateral_sum(model).lh=Lateral.result(l_all_lh);
    Lateral_sum(model).rh=Lateral.result(l_all_rh);

    %ventral
    vi=find(strcmp(Ventral.model_type,allmodels(model)));
    vlh=find(Ventral.hemi=="lh");
    vrh=find(Ventral.hemi=="rh");
    v_all_lh=intersect(vi,vlh); % find all lh of this model
    v_all_rh=intersect(vi,vrh);
    mean_v(model)=mean(Ventral.result(vi));
    sd_v(model)=std(Ventral.result(vi));
    Ventral_i_lh(:,model)=Ventral.result(v_all_lh);
    Ventral_i_rh(:,model)=Ventral.result(v_all_rh);
    Ventral_sum(model).name=allmodels(model);
    Ventral_sum(model).mean=mean_v(model);
    Ventral_sum(model).sd=sd_v(model);
    Ventral_sum(model).lh=Ventral.result(v_all_lh);
    Ventral_sum(model).rh=Ventral.result(v_all_rh);
end
%% get mean/sd noise by ROI

for r=1:length(ROI)
    roi_i=find(Fig2b_noiseCeiling.ROI==ROI{r});
    mean_noise(r)=mean(Fig2b_noiseCeiling.result(roi_i));
    sd_noise(r)=std(Fig2b_noiseCeiling.result(roi_i));
end


%% plot Fig 2c
fig2b=figure('Color', [1 1 1], 'Units','normalized','Position',[ 0.1 0.1 .4 0.9],'Name','Fig2b')

% shorten model names
model_names=regexprep(allmodels,'_','.');
%MB_50=find(model_names=="MB.RN50");
%MB_18=find(model_names=="MB.RN18");
chance_level=33;chance_xvals=0:1:5;
model_names=erase(model_names , 'TDANN.' );
model_names{3} = 'Self-Supervised SimCLR';
model_names{4} = 'Supervised Categorization';

%swap MB_50 and MB_18
MB_50=find(model_names=="MB.RN18");
MB_18=find(model_names=="MB.RN50");

xvals=[1 2 3.3 4.3];
xline1=[0.5:.1:2.5];xline2=[2.8:.1:4.8];
% row1={' MB ',' MB ' 'TDANN'};
% row2={'RN50' 'RN18' ' RN18'};
% labelArray =[row1; row2];
% tickLabels = (sprintf('%s\\newline%s\n', labelArray{:}));
 
% dorsal
h1=subplot(1,3,1); 
hold on;

% label by stream predicted tasks
model_names_dorsal=model_names;
model_names_dorsal{MB_50}='Detection RN50'; 
model_names_dorsal{MB_18}='Detection RN18';

% plot noise ceiling
area ([0.1 nmodels+.8] ,[ mean_noise(1)+sd_noise(1), mean_noise(1)+sd_noise(1)],'Facecolor',[ .9 .9 .9],'EdgeColor',"none")
area ([0.1 nmodels+.8] ,[ mean_noise(1)-sd_noise(1), mean_noise(1)-sd_noise(1)],'Facecolor',[ 1 1 1],'EdgeColor',"none")

% plot mean results
bar(xvals(1),mean_d(2),'FaceColor',[0 .5 .1],'BarWidth',.95,'EdgeColor',"none")
bar(xvals(2), mean_d(1), 'FaceColor',[0 .5 .1], 'BarWidth', .95, 'EdgeColor',"none",'FaceAlpha',0.5)
bar(xvals(3),mean_d(3),'FaceColor',[0 .5 .1],'BarWidth',.95,'EdgeColor',"none")
bar(xvals(4), mean_d(4), 'FaceColor',[0 .5 .1], 'BarWidth', .95, 'EdgeColor',"none",'FaceAlpha',0.5)

for i=1:nmodels
    text(xvals(i), 1 ,model_names_dorsal(i), 'Rotation',90,'Color', [ 1 1 1],'FontSize',18,'FontName','Avenir')
end
%plot individual data points
r18= Dorsal_i_rh(:,1);
r50= Dorsal_i_rh(:,2);
l18= Dorsal_i_lh(:,1);
l50= Dorsal_i_lh(:,2);
Dorsal_i_lh(:,1) = l50; Dorsal_i_lh(:,2) = l18;
Dorsal_i_rh(:,1) = r50; Dorsal_i_rh(:,2) = r18;


scatter(xvals,Dorsal_i_lh','^','MarkerEdgeColor',[0 .2 0],'LineWidth',1)
scatter(xvals,Dorsal_i_rh','o','MarkerEdgeColor',[0 .2 0],'LineWidth',1)

% group models to increase interpretability
set(gca,'XTick', [1 4 ],'XTickLabel',{'MB','SC'},'XTickLabelRotation',0,'FontSize',18,'FontName','Avenir');
plot(xline1, zeros(1,length(xline1)),'-','Color', [ .6 .6 .6], 'LineWidth',5)
plot(xline2, zeros(1,length(xline2)),'-','Color', [ .4 .4 .4], 'LineWidth',5)
plot(chance_xvals, chance_level*ones(1,length(chance_xvals)),'k:','LineWidth',3)
axis([ 0 nmodels+1, 0  max(mean_noise+sd_noise)*1.1])
ylabel('% voxels mapped to corresponding stream','FontSize',24,'FontName','Avenir')
title('Dorsal', 'FontSize',24,'FontName','Avenir')

% lateral
subplot(1,3,2); 
hold on
% label by stream predicted tasks
model_names_lateral=model_names;
model_names_lateral{MB_50}='Action RN50'; %action RN50)';
model_names_lateral{MB_18}='Action RN18'; %action RN18)';

% plot noise ceiling
area ([0.1 nmodels+.8] ,[mean_noise(2)+sd_noise(2), mean_noise(2)+sd_noise(2)],'Facecolor',[ .9 .9 .9],'EdgeColor',"none")
area ([0.1 nmodels+.8] ,[mean_noise(2)-sd_noise(2), mean_noise(2)-sd_noise(2)],'Facecolor',[ 1 1 1],'EdgeColor',"none")

% plot means

bar(xvals(1),mean_l(2),'FaceColor',[0.1 0.1 .7],'BarWidth',.95,'EdgeColor',"none")
bar(xvals(2),mean_l(1),'FaceColor',[0.1 0.1 .7],'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
bar(xvals(3),mean_l(3),'FaceColor',[0.1 0.1 .7],'BarWidth',.95,'EdgeColor',"none")
bar(xvals(4),mean_l(4),'FaceColor',[0.1 0.1 .7],'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)

for i=1:nmodels
    text(xvals(i),1 ,model_names_lateral(i), 'Rotation',90,'Color', [ 1 1 1],'FontSize',18,'FontName','Avenir')
end

r18= Lateral_i_rh(:,1);
r50= Lateral_i_rh(:,2);
l18= Lateral_i_lh(:,1);
l50= Lateral_i_lh(:,2);
Lateral_i_lh(:,1) = l50; Lateral_i_lh(:,2) = l18;
Lateral_i_rh(:,1) = r50; Lateral_i_rh(:,2) = r18;
% plot individual data
scatter(xvals,Lateral_i_lh','^','MarkerEdgeColor',[0  0 .2],'LineWidth',1)
scatter(xvals,Lateral_i_rh','o','MarkerEdgeColor',[0  0 .2],'LineWidth',1)

% group models to increase interpretability
set(gca,'XTick', [1 4 ],'XTickLabel',{'MB','SC'},'XTickLabelRotation',0,'FontSize',18,'FontName','Avenir');
plot(xline1, zeros(1,length(xline1)),'-','Color', [ .6 .6 .6], 'LineWidth',5)
plot(xline2, zeros(1,length(xline2)),'-','Color', [ .4 .4 .4], 'LineWidth',5)
plot(chance_xvals, chance_level*ones(1,length(chance_xvals)),'k:','LineWidth',3)
axis([ 0 nmodels+1, 0  max(mean_noise+sd_noise)*1.1])
set(gca,'Ycolor',[1 1 1])
axis([ 0 nmodels+1, 0  max(mean_noise+sd_noise)*1.1])
title('Lateral','FontSize',24,'FontName','Avenir')

%vebtral
subplot(1,3,3); 
hold on
% plot noise ceiling
area ([0.1 nmodels+.8] ,[ mean_noise(3)+sd_noise(3), mean_noise(3)+sd_noise(3)],'Facecolor',[ .9 .9 .9],'EdgeColor',"none")
area ([0.1 nmodels+.8] ,[ mean_noise(3)-sd_noise(3), mean_noise(3)-sd_noise(3)],'Facecolor',[ 1 1 1],'EdgeColor',"none")

% label by stream predicted tasks
model_names_ventral=model_names;
model_names_ventral{MB_50}='Categorization RN50'; %categorization RN50)';
model_names_ventral{MB_18}='Categorization RN18'; %categorization RN18)';


bar(xvals(1),mean_v(2),'FaceColor',[0.6 0 .3],'BarWidth',.95,'EdgeColor',"none")
bar(xvals(2),mean_v(1),'FaceColor',[0.6 0 .3],'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
bar(xvals(3),mean_v(3),'FaceColor',[0.6 0 .3],'BarWidth',.95,'EdgeColor',"none")
bar(xvals(4),mean_v(4),'FaceColor',[0.6 0 .3],'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
for i=1:nmodels
    text(xvals(i), 1,model_names_ventral(i), 'Rotation',90,'Color', [ 1 1 1],'FontSize',18,'FontName','Avenir')
end

r18= Ventral_i_rh(:,1);
r50= Ventral_i_rh(:,2);
l18= Ventral_i_lh(:,1);
l50= Ventral_i_lh(:,2);
Ventral_i_lh(:,1) = l50; Ventral_i_lh(:,2) = l18;
Ventral_i_rh(:,1) = r50; Ventral_i_rh(:,2) = r18;
scatter(xvals,Ventral_i_lh','^','MarkerEdgeColor',[0.2  0 .1],'LineWidth',1)
scatter(xvals,Ventral_i_rh','o','MarkerEdgeColor',[0.2  0 .1],'LineWidth',1)

% group models to increase interpretability
set(gca,'XTick', [1 4 ],'XTickLabel',{'MB','SC'},'XTickLabelRotation',0,'FontSize',18,'FontName','Avenir');
plot(xline1, zeros(1,length(xline1)),'-','Color', [ .6 .6 .6], 'LineWidth',5)
plot(xline2, zeros(1,length(xline2)),'-','Color', [ .4 .4 .4], 'LineWidth',5)
plot(chance_xvals, chance_level*ones(1,length(chance_xvals)),'k:','LineWidth',3)

axis([ 0 nmodels+1, 0  max(mean_noise+sd_noise)*1.1])
set(gca,'Ycolor',[1 1 1])
title('Ventral','FontSize',24,'FontName','Avenir')
axis([ 0 nmodels+1, 0  max(mean_noise+sd_noise)*1.1])

% savefig
saveas(fig2b, 'Fig2b_1023.tif', 'tif')
    
