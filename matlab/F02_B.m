% Fig2b.m
% This script generates Fig 2b.
clear all
ROI={"Dorsal", "Lateral","Ventral"}

%% read dataTables
Fig2b=readtable("new_Fig2b_dataFrame_checkpoint0.csv");
Fig2b_noiseCeiling=readtable("new_Fig2b_noiseCeiling_checkpoint0.csv");
%% get all models
% Specify column names and types
disp(Fig2b.Properties.VariableNames);
allmodels =unique(Fig2b.model_type);
allmodels
%%
%allmodels=allmodels([2 3 1 7 6 5 4])% re-order models so plots make sense
allmodels=allmodels([2 3 1 7 5 6 4])% re-order models so plots make sense
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

%%
% colors
mgreen = [0 .5 .1];
mblue = [0.1 0.1 .7];
mred = [0.6 0 .3];
mpurple = [0.40 0.07 0.57];
mgray = [0.4 0.4 0.4];
teal=[0 .5 .5];
teal=[.5 .5 .5];
gold=[.8 .6 .0];
gold=[.8 .2 .0];


% RN model
model_names{1} = '50'; 
model_names{2} = '50'; 
model_names{3} = '18'; 
model_names{4} = '18';
model_names{5} = '18'; 
model_names{6} = '18'; 
model_names{7} = '18'; 

xvals=[1:2 3.5 4.5 5.5 7 8 ];
XTickLabelString={'MB v1','MB v2', 'MB v1', 'Cat', 'SimCLR', 'TDANN Cat' 'TDANN SimCLR'};

%% plot Fig 2c
Fig2=figure('Color', [1 1 1], 'Units','normalized','Position',[ 0.1 0.1 .6 .6],'Name','Fig2b')
% dorsal
h1=subplot(1,3,1); 
hold on;

% plot noise ceiling
area ([0.1 nmodels+1.8] ,[ mean_noise(1)+sd_noise(1), mean_noise(1)+sd_noise(1)],'Facecolor',[ .9 .9 .9],'EdgeColor',"none")
area ([0.1 nmodels+1.8] ,[ mean_noise(1)-sd_noise(1), mean_noise(1)-sd_noise(1)],'Facecolor',[ 1 1 1],'EdgeColor',"none")

% plot mean results
bar(xvals(1),mean_d(1),'FaceColor',teal,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.35)
bar(xvals(2),mean_d(2),'FaceColor',teal,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.35)
bar(xvals(3),mean_d(3),'FaceColor',teal,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
bar(xvals(4),mean_d(4),'FaceColor',gold,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
bar(xvals(5),mean_d(5),'FaceColor',mpurple,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
bar(xvals(6),mean_d(6),'FaceColor',gold,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',.75)
bar(xvals(7),mean_d(7),'FaceColor',mpurple,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',1.0)

%plot individual data points
scatter(xvals,Dorsal_i_lh','^','MarkerEdgeColor',[0 .2 0],'LineWidth',2)
scatter(xvals,Dorsal_i_rh','o','MarkerEdgeColor',[0 .2 0],'LineWidth',2)

for i=1:nmodels 
     text(xvals(i), .01,model_names(i), 'Rotation',90,'Color', [.8 .8 .8],'FontSize',24,'FontName','Helvetica')
end

set(gca,'XTick', xvals);
ax = gca();
ax.TickLabelInterpreter = 'tex';
ax.XTickLabel=XTickLabelString;
set(gca,'FontSize',24,'FontName','Helvetica','XTickLabelRotation',90)
axis([ 0 nmodels+1.5, 0  max(mean_noise+sd_noise)*1.1])
ylabel('Distance Similarity [r]','FontSize',32,'FontName','Helvetica')
title('Dorsal', 'FontSize',32,'FontName','Helvetica')

% lateral
subplot(1,3,2); 
hold on
% plot noise ceiling
area ([0.1 nmodels+1.8] ,[ mean_noise(2)+sd_noise(2), mean_noise(2)+sd_noise(2)],'Facecolor',[ .9 .9 .9],'EdgeColor',"none")
area ([0.1 nmodels+1.8] ,[ mean_noise(2)-sd_noise(2), mean_noise(2)-sd_noise(2)],'Facecolor',[ 1 1 1],'EdgeColor',"none")

% plot mean results
bar(xvals(1),mean_l(1),'FaceColor',teal,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.35)
bar(xvals(2),mean_l(2),'FaceColor',teal,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.35)
bar(xvals(3),mean_l(3),'FaceColor',teal,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
bar(xvals(4),mean_l(4),'FaceColor',gold,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
bar(xvals(5),mean_l(5),'FaceColor',mpurple,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
bar(xvals(6),mean_l(6),'FaceColor',gold,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',.7)
bar(xvals(7),mean_l(7),'FaceColor',mpurple,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',1.0)

% plot individual data
scatter(xvals,Lateral_i_lh','^','MarkerEdgeColor',[0  0 .2],'LineWidth',2)
scatter(xvals,Lateral_i_rh','o','MarkerEdgeColor',[0  0 .2],'LineWidth',2)
for i=1:nmodels
    text(xvals(i), .01,model_names(i), 'Rotation',90,'Color', [ 0.8 0.8 0.8 ],'FontSize',24,'FontName','Helvetica')
end



set(gca,'XTick', xvals);
ax = gca();
ax.TickLabelInterpreter = 'tex';
ax.XTickLabel=XTickLabelString;
set(gca,'FontSize',24,'FontName','Helvetica','XTickLabelRotation',90)

axis([ 0 nmodels+1.5, 0  max(mean_noise+sd_noise)*1.1])
set(gca,'Ycolor',[1 1 1])
title('Lateral','FontSize',32,'FontName','Helvetica')

%ventral
subplot(1,3,3); 
hold on
% plot noise ceiling
area ([0.1 nmodels+1.8] ,[ mean_noise(3)+sd_noise(3), mean_noise(3)+sd_noise(3)],'Facecolor',[ .9 .9 .9],'EdgeColor',"none")
area ([0.1 nmodels+1.8] ,[ mean_noise(3)-sd_noise(3), mean_noise(3)-sd_noise(3)],'Facecolor',[ 1 1 1],'EdgeColor',"none")

%bar(xvals,mean_v,'FaceColor',[0.6 0 .3],'BarWidth',.95,'EdgeColor',"none")
% plot mean results
bar(xvals(1),mean_v(1),'FaceColor',teal,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.35)
bar(xvals(2),mean_v(2),'FaceColor',teal,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.35)
bar(xvals(3),mean_v(3),'FaceColor',teal,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
bar(xvals(4),mean_v(4),'FaceColor',gold,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
bar(xvals(5),mean_v(5),'FaceColor',mpurple,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
bar(xvals(6),mean_v(6),'FaceColor',gold,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',.75)
bar(xvals(7),mean_v(7),'FaceColor',mpurple,'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',1.0)


scatter(xvals,Ventral_i_lh','^','MarkerEdgeColor',[0.2  0 .1],'LineWidth',2)
scatter(xvals,Ventral_i_rh','o','MarkerEdgeColor',[0.2  0 .1],'LineWidth',2)

for i=1:nmodels
    text(xvals(i), .01,model_names(i), 'Rotation',90,'Color', [  .8 .8 .8],'FontSize',24,'FontName','Helvetica')
end



% group models to increase interpretability
%plot(xline1, zeros(1,length(xline1)),'-','Color', [ .6 .6 .6], 'LineWidth',5)
%plot(xline2, zeros(1,length(xline2)),'-','Color', [ .6 .6 .6], 'LineWidth',5)
%plot(xline3, zeros(1,length(xline3)),'-','Color', [ .4 .4 .4], 'LineWidth',5)
set(gca,'XTick', xvals);%, [2,5,8 ]);
ax = gca();
ax.TickLabelInterpreter = 'tex';
ax.XTickLabel=XTickLabelString;
set(gca,'FontSize',24,'FontName','Helvetica','XTickLabelRotation',90)
axis([ 0 nmodels+1.5, 0  max(mean_noise+sd_noise)*1.1])
set(gca,'Ycolor',[1 1 1])
title('Ventral','FontSize',32,'FontName','Helvetica')

%add legend
%legend('','','','lh','','','','','','','','rh','Box',"off",'Location','best','FontSize',18)

%% savefig
exportgraphics(Fig2,'Fig2b.tif','Resolution', 600);

    
