%% replot figure 2c by ROI
%dataDir='/Users/kalanit/Projects/Dawn/SpaceStreamPaper/NN/'
%cd(dataDir)
ROI={"Dorsal", "Lateral","Ventral"}

%% read dataTables
Fig2c=readtable("Fig2c_dataFrame.csv");
Fig2c_noiseCeiling=readtable("Fig2c_noiseCeiling.csv");
%% get all models
% Specify column names and types
disp(Fig2c.Properties.VariableNames);
allmodels =unique(Fig2c.model_type);
allmodels=allmodels([6 4 5 3 1 2 8 7])% re-order models so plots make sense
nmodels=length(allmodels);

Dorsal_i=find(Fig2c.ROIS=="Dorsal");
Dorsal=Fig2c(Dorsal_i,:);
Lateral_i=find(Fig2c.ROIS=="Lateral");
Lateral=Fig2c(Lateral_i,:);
Ventral_i=find(Fig2c.ROIS=="Ventral");
Ventral=Fig2c(Ventral_i,:);

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
    roi_i=find(Fig2c_noiseCeiling.ROI==ROI{r});
    mean_noise(r)=mean(Fig2c_noiseCeiling.result(roi_i));
    sd_noise(r)=std(Fig2c_noiseCeiling.result(roi_i));
end


%% plot Fig 2c
fig2c=figure('Color', [1 1 1], 'Units','normalized','Position',[ 0.1 0.1 .7 .6],'Name','Fig2c')

% shorten model names
model_names=regexprep(allmodels,'_','.');
model_names=erase(model_names , 'MB.' );
model_names=erase(model_names,'RN18.')
model_names=erase(model_names,'RN50.')
model_names=erase(model_names , 'TDANN.' );
xvals=[1:3 4.2:1:6.2 7.4 8.4];
xline1=[0.5:.1:3.5];xline2=[3.7:.1:6.7]; xline3=[6.9:.1:8.9];
row1={' MB ',' MB ' 'TDANN'};
row2={'RN50' 'RN18' ' RN18'};
labelArray =[row1; row2];
tickLabels = (sprintf('%s\\newline%s\n', labelArray{:}));
model_names{1} = 'Det.';
model_names{2} = 'Act.';
model_names{3} = 'Cat.';
model_names{4} = 'Det.';
model_names{5} = 'Act.';
model_names{6} = 'Cat.';
model_names{7} = 'Self-Supervised';
model_names{8} = 'Supervised';
% dorsal
h1=subplot(1,3,1); 
hold on;

% plot noise ceiling
area ([0.1 nmodels+.8] ,[ mean_noise(1)+sd_noise(1), mean_noise(1)+sd_noise(1)],'Facecolor',[ .9 .9 .9],'EdgeColor',"none")
area ([0.1 nmodels+.8] ,[ mean_noise(1)-sd_noise(1), mean_noise(1)-sd_noise(1)],'Facecolor',[ 1 1 1],'EdgeColor',"none")

% plot mean results
bar(xvals(1:3),mean_d(1:3),'FaceColor',[0 .5 .1],'BarWidth',.95,'EdgeColor',"none")
bar(xvals(4:6),mean_d(4:6),'FaceColor',[0 .5 .1],'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
bar(xvals(7),mean_d(7),'FaceColor',[0 .5 .1],'BarWidth',.95,'EdgeColor',"none")
bar(xvals(8),mean_d(8),'FaceColor',[0 .5 .1],'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)

for i=1:nmodels

    text(xvals(i), .005,model_names(i), 'Rotation',90,'Color', [ 1 1 1],'FontSize',18,'FontName','Avenir')
end
%plot individual data points

scatter(xvals,Dorsal_i_lh','^','MarkerEdgeColor',[0 .2 0],'LineWidth',1)
scatter(xvals,Dorsal_i_rh','o','MarkerEdgeColor',[0 .2 0],'LineWidth',1)

% group models to increase interpretability
plot(xline1, zeros(1,length(xline1)),'-','Color', [ .6 .6 .6], 'LineWidth',5)
plot(xline2, zeros(1,length(xline2)),'-','Color', [ .6 .6 .6], 'LineWidth',5)
plot(xline3, zeros(1,length(xline3)),'-','Color', [ .4 .4 .4], 'LineWidth',5)
set(gca,'XTick', [2,5,8 ]);
ax = gca();
ax.TickLabelInterpreter = 'tex';
ax.XTickLabel=tickLabels
set(gca,'FontSize',18,'FontName','Avenir','XTickLabelRotation',0)
axis([ 0 nmodels+1, 0  max(mean_noise+sd_noise)*1.1])
ylabel('Functional Similarity [r]','FontSize',24,'FontName','Avenir')
title('Dorsal', 'FontSize',24,'FontName','Avenir')

% lateral
subplot(1,3,2); 
hold on
% plot noise ceiling
area ([0.1 nmodels+.8] ,[ mean_noise(2)+sd_noise(2), mean_noise(2)+sd_noise(2)],'Facecolor',[ .9 .9 .9],'EdgeColor',"none")
area ([0.1 nmodels+.8] ,[ mean_noise(2)-sd_noise(2), mean_noise(2)-sd_noise(2)],'Facecolor',[ 1 1 1],'EdgeColor',"none")

% plot means
bar(xvals(1:3),mean_l(1:3),'FaceColor',[0.1 .1 .7],'BarWidth',.95,'EdgeColor',"none")
bar(xvals(4:6),mean_l(4:6),'FaceColor',[0.1 .1 .7],'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
bar(xvals(7),mean_l(7),'FaceColor',[.1 .1 .7],'BarWidth',.95,'EdgeColor',"none")
bar(xvals(8),mean_l(8),'FaceColor',[.1 .1 .7],'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
%bar(xvals,mean_l,'FaceColor',[0.1 0.1 .7],'BarWidth',.95,'EdgeColor',"none")
for i=1:nmodels
    text(xvals(i), .005,model_names(i), 'Rotation',90,'Color', [ 1 1 1],'FontSize',18,'FontName','Avenir')
end

% plot individual data
scatter(xvals,Lateral_i_lh','^','MarkerEdgeColor',[0  0 .2],'LineWidth',1)
scatter(xvals,Lateral_i_rh','o','MarkerEdgeColor',[0  0 .2],'LineWidth',1)

% group models to increase interpretability
plot(xline1, zeros(1,length(xline1)),'-','Color', [ .6 .6 .6], 'LineWidth',5)
plot(xline2, zeros(1,length(xline2)),'-','Color', [ .6 .6 .6], 'LineWidth',5)
plot(xline3, zeros(1,length(xline3)),'-','Color', [ .4 .4 .4], 'LineWidth',5)
set(gca,'XTick', [2,5,8 ]);
ax = gca();
ax.TickLabelInterpreter = 'tex';
ax.XTickLabel=tickLabels
set(gca,'FontSize',18,'FontName','Avenir','XTickLabelRotation',0)
axis([ 0 nmodels+1, 0  max(mean_noise+sd_noise)*1.1])
set(gca,'Ycolor',[1 1 1])
axis([ 0 nmodels+1, 0  max(mean_noise+sd_noise)*1.1])
title('Lateral','FontSize',24,'FontName','Avenir')

%vebtral
model_names{7} = 'Self-Supervised SimCLR';
model_names{8} = 'Supervised Cat.';
subplot(1,3,3); 
hold on
% plot noise ceiling
area ([0.1 nmodels+.8] ,[ mean_noise(3)+sd_noise(3), mean_noise(3)+sd_noise(3)],'Facecolor',[ .9 .9 .9],'EdgeColor',"none")
area ([0.1 nmodels+.8] ,[ mean_noise(3)-sd_noise(3), mean_noise(3)-sd_noise(3)],'Facecolor',[ 1 1 1],'EdgeColor',"none")

%bar(xvals,mean_v,'FaceColor',[0.6 0 .3],'BarWidth',.95,'EdgeColor',"none")
% plot mean results
bar(xvals(1:3),mean_v(1:3),'FaceColor',[0.6 0 .3],'BarWidth',.95,'EdgeColor',"none")
bar(xvals(4:6),mean_v(4:6),'FaceColor',[0.6 0 .3],'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
bar(xvals(7),mean_v(7),'FaceColor',[0.6 0 .3],'BarWidth',.95,'EdgeColor',"none")
bar(xvals(8),mean_v(8),'FaceColor',[0.6 0 .3],'BarWidth',.95,'EdgeColor',"none",'FaceAlpha',0.5)
for i=1:nmodels
    text(xvals(i), .005,model_names(i), 'Rotation',90,'Color', [ 1 1 1],'FontSize',18)
end
scatter(xvals,Ventral_i_lh','^','MarkerEdgeColor',[0.2  0 .1],'LineWidth',1)
scatter(xvals,Ventral_i_rh','o','MarkerEdgeColor',[0.2  0 .1],'LineWidth',1)

% group models to increase interpretability
plot(xline1, zeros(1,length(xline1)),'-','Color', [ .6 .6 .6], 'LineWidth',5)
plot(xline2, zeros(1,length(xline2)),'-','Color', [ .6 .6 .6], 'LineWidth',5)
plot(xline3, zeros(1,length(xline3)),'-','Color', [ .4 .4 .4], 'LineWidth',5)
set(gca,'XTick', [2,5,8 ]);
ax = gca();
ax.TickLabelInterpreter = 'tex';
ax.XTickLabel=tickLabels
set(gca,'FontSize',18,'FontName','Avenir','XTickLabelRotation',0)
axis([ 0 nmodels+1, 0  max(mean_noise+sd_noise)*1.1])
set(gca,'Ycolor',[1 1 1])
title('Ventral','FontSize',24,'FontName','Avenir')
axis([ 0 nmodels+1, 0  max(mean_noise+sd_noise)*1.1])

%add legend
%legend('','','','lh','','','','','','','','rh','Box',"off",'Location','best','FontSize',18)

%% savefig
saveas(fig2c, 'Fig2c_1023.tif', 'tif');
    
