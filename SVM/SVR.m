function varargout = SVR(varargin)
% SVR M-file for SVR.fig
%      SVR, by itself, creates a new SVR or raises the existing
%      singleton*.
%
%      H = SVR returns the handle to a new SVR or the handle to
%      the existing singleton*.
%
%      SVR('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SVR.M with the given input arguments.
%
%      SVR('Property','Value',...) creates a new SVR or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SVR_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SVR_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SVR

% Last Modified by GUIDE v2.5 22-Nov-2018 15:45:40

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SVR_OpeningFcn, ...
                   'gui_OutputFcn',  @SVR_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before SVR is made visible.
function SVR_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SVR (see VARARGIN)

% Choose default command line output for SVR
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SVR wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SVR_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close;       



% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename,pathname,filterindex] = uigetfile({'*.mat';'*.*'},'load data');  % load data

if filterindex
    filename=strcat(pathname,filename);
    datatemp = load(filename);
    global TRAIN_DATA TRAIN_LABEL TEST_DATA TEST_LABEL   
    global testflag
    testflag = 1;
    TRAIN_DATA = datatemp.train_x;       %  load train_x
    TRAIN_LABEL = datatemp.train_y;      %  load train_y
    if length(fieldnames(datatemp)) == 2  
        TEST_DATA = TRAIN_DATA;
        TEST_LABEL = TRAIN_LABEL;
        testflag = 0; 
    else
        TEST_DATA = datatemp.test_x;
        TEST_LABEL = datatemp.test_y;
    end
    
    [trains,traind] = size(TRAIN_DATA);   
    [tests,testd] = size(TEST_DATA); 
    global mystring
    mystring = [];
    mystring = {'NOTE'};
    line1 = ['load!'];
    line2 = ['training set：',num2str(trains),',dimension',num2str(traind)];
    line3 = ['test set：',num2str(tests),',dimension',num2str(testd)];
    mystring{length(mystring)+1,1} = line1;
    mystring{length(mystring)+1,1} = line2;
    mystring{length(mystring)+1,1} = line3;
    set(handles.info,'String',mystring);
end

guidata(hObject, handles);

% --- Executes on button press in SVR.
function SVR_Callback(hObject, eventdata, handles)
% hObject    handle to SVR (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%% core
global TRAIN_DATA TRAIN_LABEL TEST_DATA TEST_LABEL
global testflag
global mystring
mystring = [];

%
TrainL = TRAIN_LABEL;
Train = TRAIN_DATA;
TestL = TEST_LABEL;
Test = TEST_DATA;

% xscaleflag
xscaleflag = get(handles.xscale,'Value');    %independent variables are normalized
switch xscaleflag
    case 1
        [Train,Test] = scaleForSVM(Train,Test,-1,1);
        line1 = ['The independent variables are normalized!'];
        line2 = ['The data is structured to[-1,1]'];
        mystring{length(mystring)+1,1} = line1;
        mystring{length(mystring)+1,1} = line2;
        set(handles.info,'String',mystring);
    case 2
        [Train,Test] = scaleForSVM(Train,Test,0,1);
        line1 = ['The independent variables are normalized!'];
        line2 = ['The data is structured to[0,1]'];
        mystring{length(mystring)+1,1} = line1;
        mystring{length(mystring)+1,1} = line2;
        set(handles.info,'String',mystring);
    case 3
        nothing = 0;
    case 4
        lower = str2num( get(handles.min,'String') );
        upper = str2num( get(handles.max,'String') );
        [Train,Test] = scaleForSVM(Train,Test,lower,upper);    
        line1 = ['The independent variables are normalized!'];
        line2 = ['The data is structured to','[',num2str(lower),',',num2str(upper),']'];
        mystring{length(mystring)+1,1} = line1;
        mystring{length(mystring)+1,1} = line2;
        set(handles.info,'String',mystring);
end

% yscaleflag
yscaleflag = get(handles.yscale,'Value');   % The dependent variable is normalized
global ps
switch yscaleflag
    case 1
        [TrainL,TestL,ps] = scaleForSVM(TrainL,TestL,-1,1);
        line1 = ['The dependent variable is normalized!'];
        line2 = ['The data is structured to[-1,1]'];
        mystring{length(mystring)+1,1} = line1;
        mystring{length(mystring)+1,1} = line2;
        set(handles.info,'String',mystring);
    case 2
        [TrainL,TestL,ps] = scaleForSVM(TrainL,TestL,0,1);
        line1 = ['The dependent variable is normalized!'];
        line2 = ['The data is structured to[0,1]'];
        mystring{length(mystring)+1,1} = line1;
        mystring{length(mystring)+1,1} = line2;
        set(handles.info,'String',mystring);
    case 3
        nothing = 0;
    case 4
        lower = str2num( get(handles.ymin,'String') );
        upper = str2num( get(handles.ymax,'String') );
        [TrainL,TestL,ps] = scaleForSVM(TrainL,TestL,lower,upper);    
        line1 = ['The dependent variable is normalized!'];
        line2 = ['The data is structured to','[',num2str(lower),',',num2str(upper),']',];
        mystring{length(mystring)+1,1} = line1;
        mystring{length(mystring)+1,1} = line2;
        set(handles.info,'String',mystring);
end

tic  % time
% pcaflag
pcaflag = get(handles.pca,'Value');    %  PCA
switch pcaflag
    case 1
        line1 = ['dimensionality reduction...'];
        mystring{length(mystring)+1,1} = line1;
        set(handles.info,'String',mystring);
        
        threshold = str2num(get(handles.percent,'String'));   
        [mtrain,ntrain] = size(Train);
        [mtest,ntest] = size(Test);
        dataset = [Train;Test];
        [dataset_coef,dataset_score,dataset_latent,dataset_t2] = princomp(dataset);
        dataset_cumsum = 100*cumsum(dataset_latent)./sum(dataset_latent);
        index = find(dataset_cumsum >= threshold);
        percent_explained = 100*dataset_latent/sum(dataset_latent);
        axes(handles.axesB);   
        cla reset;
        pareto(percent_explained);
        set(gcf,'Nextplot','add');
        xlabel('Principal Component');
        ylabel('Variance Explained (%)');
        grid on;
        Train = dataset_score(1:mtrain,:);
        Test = dataset_score( (mtrain+1):(mtrain+mtest),: );
        Train = Train(:,1:index(1));
        Test = Test(:,1:index(1));
        
        [trains,traind] = size(TRAIN_DATA);  
        line1 = ['End ！'];
        line2 = ['Dimensionality decline from',num2str(traind),'to',num2str(size(Train,2))];
        mystring{length(mystring)+1,1} = line1;
        mystring{length(mystring)+1,1} = line2;
        set(handles.info,'String',mystring);
        
    case 2
        nothing = 0;
        axes(handles.axesB);
        cla reset;
end


% findflag
findflag = get(handles.find,'Value');   
switch findflag
    case 1                             % GS
        
        line1 = 'Parameter optimization starts(grid search method)...';
        mystring{length(mystring)+1,1} = line1;
        set(handles.info,'String',mystring);
        
        train = Train;
        train_label = TrainL;
        cmin = str2num(get(handles.gridcmin,'String'));
        cmax = str2num(get(handles.gridcmax,'String'));;
        gmin = str2num(get(handles.gridgmin,'String'));;
        gmax = str2num(get(handles.gridgmax,'String'));;
        v = str2num(get(handles.v,'String'));
        cstep = str2num(get(handles.gridcstep,'String'));
        gstep = str2num(get(handles.gridgstep,'String'));
        [X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
        [m,n] = size(X);
        cg = zeros(m,n);
        
        eps = 10^(-4);
        
        %% record acc with different c & g,and find the bestacc with the smallest c
        bestc = 1;
        bestg = 0.1;
        mse = Inf; % mse
        basenum = 2;
        for i = 1:m
            for j = 1:n
                cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) ),' -s 3 -p 0.1'];
                cg(i,j) = svmtrain(train_label, train, cmd);
                
                if cg(i,j) < mse
                    mse = cg(i,j);
                    bestc = basenum^X(i,j);
                    bestg = basenum^Y(i,j);
                end
                
                if abs( cg(i,j)-mse )<=eps && bestc > basenum^X(i,j)
                    mse = cg(i,j);
                    bestc = basenum^X(i,j);
                    bestg = basenum^Y(i,j);
                end
                
            end
        end
        str = ['CVmse = ',num2str(mse)];
        set(handles.CVaccuracy,'String',str);
        str = ['best c = ',num2str(bestc)];
        set(handles.bestc,'String',str);
        str = ['best g = ',num2str(bestg)];
        set(handles.bestg,'String',str);
        %% to draw the acc with different c & g
        axes(handles.axesA);   
        cla reset;
        view(3);
        meshc(X,Y,cg);
        axis auto;
        xlabel('log2c','FontSize',12);
        ylabel('log2g','FontSize',12);
        zlabel('MSE','FontSize',12);
        firstline = 'SVR result[GridSearchMethod]';
        secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
            ' CVmse=',num2str(mse)];
        title({firstline;secondline},'Fontsize',12);
        
        cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.1'];
        
        line1 = 'Parameter optimization is completed(grid search method)';
        mystring{length(mystring)+1,1} = line1;
        set(handles.info,'String',mystring);        
    case 2                                 % GA
        line1 = 'Parameter optimization starts(ga method)...';
        mystring{length(mystring)+1,1} = line1;
        set(handles.info,'String',mystring);         
        
        train_data = Train;
        train_label = TrainL;
        
        % 参数初始化
        ga_option = struct('maxgen',100,'sizepop',20,'ggap',0.9,...
            'cbound',[0,100],'gbound',[0,100],'v',5);
       
        
        ga_option.maxgen = str2num(get(handles.maxgen,'String'));
        ga_option.sizepop = str2num(get(handles.sizepop,'String'));
        ga_option.cbound = [str2num(get(handles.cmin,'String')),str2num(get(handles.cmax,'String'))];
        ga_option.gbound = [str2num(get(handles.gmin,'String')),str2num(get(handles.gmax,'String'))];
        ga_option.v = str2num(get(handles.v,'String'));        
        
        MAXGEN = ga_option.maxgen;
        NIND = ga_option.sizepop;
        NVAR = 2;
        PRECI = 20;
        GGAP = ga_option.ggap;
        trace = zeros(MAXGEN,2);
        
        FieldID = ...
            [rep([PRECI],[1,NVAR]);[ga_option.cbound(1),ga_option.gbound(1);ga_option.cbound(2),ga_option.gbound(2)]; ...
            [1,1;0,0;0,1;1,1]];
        
        Chrom = crtbp(NIND,NVAR*PRECI);
        
        gen = 1;
        v = ga_option.v;
        BestMSE = inf;
        Bestc = 0;
        Bestg = 0;
        %%
        cg = bs2rv(Chrom,FieldID);
        
        for nind = 1:NIND
            cmd = ['-v ',num2str(v),' -c ',num2str(cg(nind,1)),' -g ',num2str(cg(nind,2)),' -s 3 -p 0.01'];
            ObjV(nind,1) = svmtrain(train_label,train_data,cmd);
        end
        [BestMSE,I] = min(ObjV);
        Bestc = cg(I,1);
        Bestg = cg(I,2);
        
        %% Interpretation of result
        for gen = 1:MAXGEN
            FitnV = ranking(ObjV);
            
            SelCh = select('sus',Chrom,FitnV,GGAP);
            SelCh = recombin('xovsp',SelCh,0.7);
            SelCh = mut(SelCh);
            
            cg = bs2rv(SelCh,FieldID);
            for nind = 1:size(SelCh,1)
                cmd = ['-v ',num2str(v),' -c ',num2str(cg(nind,1)),' -g ',num2str(cg(nind,2))];
                ObjVSel(nind,1) = svmtrain(train_label,train_data,cmd);
            end
            
            [Chrom,ObjV] = reins(Chrom,SelCh,1,1,ObjV,ObjVSel);
            
            [NewBestCVaccuracy,I] = max(ObjV);
            cg_temp = bs2rv(Chrom,FieldID);
            temp_NewBestCVaccuracy = NewBestCVaccuracy;
            
            if NewBestCVaccuracy < BestMSE
                BestMSE = NewBestCVaccuracy;
                Bestc = cg_temp(I,1);
                Bestg = cg_temp(I,2);
            end
            
            if abs( NewBestCVaccuracy-BestMSE ) <= 10^(-2) && ...
                    cg_temp(I,1) < Bestc
                BestMSE = NewBestCVaccuracy;
                Bestc = cg_temp(I,1);
                Bestg = cg_temp(I,2);
            end
            trace(gen,1) = max(ObjV);
            trace(gen,2) = sum(ObjV)/length(ObjV);
        end
        str = ['CVmse = ',num2str(BestMSE)];
        set(handles.CVaccuracy,'String',str);
        str = ['best c = ',num2str(Bestc)];
        set(handles.bestc,'String',str);
        str = ['best g = ',num2str(Bestg)];
        set(handles.bestg,'String',str);
        axes(handles.axesA); 
        cla reset;
        view(2);
        hold on;
        trace = round(trace*10000)/10000;
        plot(trace(1:gen,1),'r*-','LineWidth',1.5);
        plot(trace(1:gen,2),'o-','LineWidth',1.5);
        legend('Best fitness','Mean fitness');
        xlabel('Evolution algebra','FontSize',12);
        ylabel('Fitness','FontSize',12);
        grid on;
        axis auto;
        line1 = 'Fitness curve MSE[GAmethod]';
        line2 = ['(Terminate algebra=', ...
            num2str(gen),',population quantity pop=', ...
            num2str(NIND),')'];
        line3 = ['Best c=',num2str(Bestc),' g=',num2str(Bestg), ...
            ' CVmse=',num2str(BestMSE),'%'];
        title({line1;line2;line3},'FontSize',12);
        cmd = ['-c ',num2str(Bestc),' -g ',num2str(Bestg),' -s 3 -p 0.1'];
        
        line1 = 'Parameter optimization is completed(ga method)';
        mystring{length(mystring)+1,1} = line1;
        set(handles.info,'String',mystring);          
    case 3                                  % PSO 
        line1 = 'Parameter optimization starts(pso method)...';
        mystring{length(mystring)+1,1} = line1;
        set(handles.info,'String',mystring);           
        train = Train;
        train_label = TrainL;
         % parameter initialization
        pso_option = struct('c1',1.5,'c2',1.7,'maxgen',100,'sizepop',20, ...
            'k',0.6,'wV',1,'wP',1,'v',5, ...
            'popcmax',10^2,'popcmin',10^(-1),'popgmax',10^3,'popgmin',10^(-2));
       
        
        pso_option.maxgen = str2num(get(handles.maxgen,'String'));
        pso_option.sizepop = str2num(get(handles.sizepop,'String'));
        pso_option.popcmin = str2num(get(handles.cmin,'String'))+0.01;
        pso_option.popcmax = str2num(get(handles.cmax,'String'));
        pso_option.popgmin = str2num(get(handles.gmin,'String'))+0.01;
        pso_option.popgmax = str2num(get(handles.gmax,'String'));
        pso_option.v = str2num(get(handles.v,'String'));  
        
        Vcmax = pso_option.k*pso_option.popcmax;
        Vcmin = -Vcmax ;
        Vgmax = pso_option.k*pso_option.popgmax;
        Vgmin = -Vgmax ;
        
        eps = 10^(-3);
        %% It produces the initial particle and the velocity
        for i=1:pso_option.sizepop
            
            % Random population and velocity
            pop(i,1) = (pso_option.popcmax-pso_option.popcmin)*rand+pso_option.popcmin;
            pop(i,2) = (pso_option.popgmax-pso_option.popgmin)*rand+pso_option.popgmin;
            V(i,1)=Vcmax*rands(1,1);
            V(i,2)=Vgmax*rands(1,1);
            
            % Calculate the initial fitness
            cmd = ['-v ',num2str(pso_option.v),' -c ',num2str( pop(i,1) ),' -g ',num2str( pop(i,2) ),' -s 3 -p 0.1'];
            fitness(i) = svmtrain(train_label, train, cmd);
        end
        
        % Find the extremum and the extremum
        [global_fitness bestindex]=min(fitness); 
        local_fitness=fitness;  
        
        global_x=pop(bestindex,:);   
        local_x=pop;    
        avgfitness_gen = zeros(1,pso_option.maxgen);
        
        %%  Iterative Refinement
        for i=1:pso_option.maxgen
            
            for j=1:pso_option.sizepop
                
                %velocity update
                V(j,:) = pso_option.wV*V(j,:) + pso_option.c1*rand*(local_x(j,:) - pop(j,:)) + pso_option.c2*rand*(global_x - pop(j,:));
                if V(j,1) > Vcmax
                    V(j,1) = Vcmax;
                end
                if V(j,1) < Vcmin
                    V(j,1) = Vcmin;
                end
                if V(j,2) > Vgmax
                    V(j,2) = Vgmax;
                end
                if V(j,2) < Vgmin
                    V(j,2) = Vgmin;
                end
                
                %population update
                pop(j,:)=pop(j,:) + pso_option.wP*V(j,:);
                if pop(j,1) > pso_option.popcmax
                    pop(j,1) = pso_option.popcmax;
                end
                if pop(j,1) < pso_option.popcmin
                    pop(j,1) = pso_option.popcmin;
                end
                if pop(j,2) > pso_option.popgmax
                    pop(j,2) = pso_option.popgmax;
                end
                if pop(j,2) < pso_option.popgmin
                    pop(j,2) = pso_option.popgmin;
                end
                
                % Adaptive mutation particle 
                if rand>0.5
                    k=ceil(2*rand);
                    if k == 1
                        pop(j,k) = (20-1)*rand+1;
                    end
                    if k == 2
                        pop(j,k) = (pso_option.popgmax-pso_option.popgmin)*rand + pso_option.popgmin;
                    end
                end
                
                %Fitness Values
                cmd = ['-v ',num2str(pso_option.v),' -c ',num2str( pop(j,1) ),' -g ',num2str( pop(j,2) ),' -s 3 -p 0.1'];
                fitness(j) = svmtrain(train_label, train, cmd);     
                
                %Individual optimal renewal
                if fitness(j) < local_fitness(j)
                    local_x(j,:) = pop(j,:);
                    local_fitness(j) = fitness(j);
                end
                
                if abs( fitness(j)-local_fitness(j) )<=eps && pop(j,1) < local_x(j,1)
                    local_x(j,:) = pop(j,:);
                    local_fitness(j) = fitness(j);
                end
                
                %Group optimal renewal
                if fitness(j) < global_fitness
                    global_x = pop(j,:);
                    global_fitness = fitness(j);
                end
                
                if abs( fitness(j)-global_fitness )<=eps && pop(j,1) < global_x(1)
                    global_x = pop(j,:);
                    global_fitness = fitness(j);
                end
                
            end
            
            fit_gen(i) = global_fitness;
            avgfitness_gen(i) = sum(fitness)/pso_option.sizepop;
        end
        
        %% interpretation of result
        axes(handles.axesA);  
        cla reset;
        view(2);
        hold on;
        plot(fit_gen,'r*-','LineWidth',1.5);
        plot(avgfitness_gen,'o-','LineWidth',1.5);
        legend('Best fitness','Mean fitness');
        xlabel('Evolution algebra','FontSize',12);
        ylabel('Fitness','FontSize',12);
        grid on;
        bestc = global_x(1);
        bestg = global_x(2);
        bestCVaccuarcy = fit_gen(pso_option.maxgen);
        
        str = ['CVmse = ',num2str(bestCVaccuarcy)];
        set(handles.CVaccuracy,'String',str);
        str = ['best c = ',num2str(bestc)];
        set(handles.bestc,'String',str);
        str = ['best g = ',num2str(bestg)];
        set(handles.bestg,'String',str);
        
        line1 = 'Fitness curve MSE[PSOmethod]';
        line2 = ['(c1=',num2str(pso_option.c1), ...
            ',c2=',num2str(pso_option.c2),',Terminate algebra=', ...
            num2str(pso_option.maxgen),',Population quantitypop=', ...
            num2str(pso_option.sizepop),')'];
        line3 = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
            ' CVmse=',num2str(bestCVaccuarcy)];
        title({line1;line2;line3},'FontSize',12);
        cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.1'];
        
        line1 = 'Parameter optimization is completed(pso method)';
        mystring{length(mystring)+1,1} = line1;
        set(handles.info,'String',mystring);
    case 4                                  % GA-p
        line1 = 'Parameter optimization starts(ga method)...';
        mystring{length(mystring)+1,1} = line1;
        set(handles.info,'String',mystring);      
        
        train_data = Train;
        train_label = TrainL;
        
        % parameter initialization
        ga_option = struct('maxgen',100,'sizepop',20,'ggap',0.9,...
            'cbound',[0,100],'gbound',[0,100],'pbound',[0,1],'v',5);
      
        
        ga_option.maxgen = str2num(get(handles.maxgen,'String'));
        ga_option.sizepop = str2num(get(handles.sizepop,'String'));
        ga_option.cbound = [str2num(get(handles.cmin,'String')),str2num(get(handles.cmax,'String'))];
        ga_option.gbound = [str2num(get(handles.gmin,'String')),str2num(get(handles.gmax,'String'))];
        ga_option.pbound = [str2num(get(handles.pmin,'String')),str2num(get(handles.pmax,'String'))];
        ga_option.v = str2num(get(handles.v,'String'));
         
        MAXGEN = ga_option.maxgen;
        NIND = ga_option.sizepop;
        NVAR = 3;
        PRECI = 20;
        GGAP = ga_option.ggap;
        trace = zeros(MAXGEN,2);
        
        FieldID = ...
            [rep([PRECI],[1,NVAR]); ...
            [ga_option.cbound(1),ga_option.gbound(1),ga_option.pbound(1);ga_option.cbound(2),ga_option.gbound(2),ga_option.pbound(2);];...
            [1,1,1;0,0,0;0,1,1;1,1,1]];
        
        Chrom = crtbp(NIND,NVAR*PRECI);
        
        gen = 1;
        v = ga_option.v;
        BestMSE = inf;
        Bestc = 0;
        Bestg = 0;
        Bestp = 0;
        %%
        cg = bs2rv(Chrom,FieldID);
        
        for nind = 1:NIND
            cmd = ['-v ',num2str(v),' -c ',num2str(cg(nind,1)),' -g ',num2str(cg(nind,2)),' -p ',num2str(cg(nind,3)),' -s 3'];
            ObjV(nind,1) = svmtrain(train_label,train_data,cmd);
        end
        [BestMSE,I] = min(ObjV);
        Bestc = cg(I,1);
        Bestg = cg(I,2);
        Bestp = cg(I,3);
        
        %%
        for gen = 1:MAXGEN
            FitnV = ranking(ObjV);
            
            SelCh = select('sus',Chrom,FitnV,GGAP);
            SelCh = recombin('xovsp',SelCh,0.7);
            SelCh = mut(SelCh);
            
            cg = bs2rv(SelCh,FieldID);
            for nind = 1:size(SelCh,1)
                cmd = ['-v ',num2str(v),' -c ',num2str(cg(nind,1)),' -g ',num2str(cg(nind,2)),' -p ',num2str(cg(nind,3)),' -s 3'];
                ObjVSel(nind,1) = svmtrain(train_label,train_data,cmd);
            end
            
            [Chrom,ObjV] = reins(Chrom,SelCh,1,1,ObjV,ObjVSel);
            
            [NewBestCVaccuracy,I] = min(ObjV);
            cg_temp = bs2rv(Chrom,FieldID);
            temp_NewBestCVaccuracy = NewBestCVaccuracy;
            
            if NewBestCVaccuracy < BestMSE
                BestMSE = NewBestCVaccuracy;
                Bestc = cg_temp(I,1);
                Bestg = cg_temp(I,2);
                Bestp = cg_temp(I,3);
            end
            
            if abs( NewBestCVaccuracy-BestMSE ) <= 10^(-2) && ...
                    cg_temp(I,1) < Bestc
                BestMSE = NewBestCVaccuracy;
                Bestc = cg_temp(I,1);
                Bestg = cg_temp(I,2);
                Bestp = cg_temp(I,3);
            end
            
            trace(gen,1) = min(ObjV);
            trace(gen,2) = sum(ObjV)/length(ObjV);
            
        end
        
        %% interpretation of result
        str = ['CVmse = ',num2str(BestMSE)];
        set(handles.CVaccuracy,'String',str);
        str = ['best c = ',num2str(Bestc)];
        set(handles.bestc,'String',str);
        str = {['best g = ',num2str(Bestg)];['best p = ',num2str(Bestp)]};
        set(handles.bestg,'String',str);
        axes(handles.axesA); 
        cla reset;
        view(2);
        hold on;
        trace = round(trace*10000)/10000;
        plot(trace(1:gen,1),'r*-','LineWidth',1.5);
        plot(trace(1:gen,2),'o-','LineWidth',1.5);
        legend('Best fitness','Mean fitness');
        xlabel(' Evolution algebra','FontSize',12);
        ylabel('Fitness','FontSize',12);
        grid on;
        axis auto;
        line1 = 'Fitness curve MSE[GAmethod]';
        line2 = ['(Termination of algebra=', ...
            num2str(gen),',population quantity pop=', ...
            num2str(NIND),')'];
        line3 = ['Best c=',num2str(Bestc),' g=',num2str(Bestg),' p=',num2str(Bestp), ...
            ' CVmse=',num2str(BestMSE),'%'];
        title({line1;line2;line3},'FontSize',12);
        cmd = ['-c ',num2str(Bestc),' -g ',num2str(Bestg),' -s 3 -p 0.1'];
        
        line1 = 'Parameter optimization is completed(ga method)';
        mystring{length(mystring)+1,1} = line1;
        set(handles.info,'String',mystring);   
        
end
%%

%%
line1 = 'Start training & forecasting...';
mystring{length(mystring)+1,1} = line1;
set(handles.info,'String',mystring);

global Model pretrain pretest 
Model = svmtrain(TrainL,Train,cmd);
[pretrain,trainacc] = svmpredict(TrainL,Train,Model);
[pretest,testacc] = svmpredict(TestL,Test,Model);
testacc
trainacc
if get(handles.yscale,'Value') ~= 3
    pretrain = mapminmax('reverse',pretrain',ps);
    pretrain = pretrain';
    pretest = mapminmax('reverse',pretest',ps);
    pretest = pretest'-0.080;
end

% 显示mse和R^2
str = {['train set mse = ',num2str(trainacc(2))];['train set R^2 = ',num2str(trainacc(3))]};
set(handles.trainacc,'String',str);
str = {['test set mse = ',num2str(testacc(2))];['test set R^2 = ',num2str(testacc(3))]};
set(handles.testacc,'String',str);

% Calculate the average relative error of the training set
 M = size(TRAIN_LABEL);
 m = M(:,1);
 trainmape = 0;
for i = 1:m
    trainmape_i = abs((pretrain(i,1)-TRAIN_LABEL(i,1))./TRAIN_LABEL(i,1));
     trainmape = (trainmape + trainmape_i);
 end
 trainmape = trainmape./m;

% Calculate the average relative error of the test set
 N = size(TEST_LABEL);
 n = N(:,1);
 testmape = 0;
 for i = 1:n
     testmape_i = abs((pretest(i,1)-TEST_LABEL(i,1))./TEST_LABEL(i,1));
     testmape = (testmape + testmape_i);
 end
 testmape = testmape./n;

% Show the mean relative error
 str = {['train mape = ',num2str(trainmape)]};
 set(handles.trainmape,'String',str);
 str = {['test mape = ',num2str(testmape)]};
 set(handles.testmape,'String',str);
 
 % Calculate training set root mean square error train RMSE
 M = size(TRAIN_LABEL);
 m = M(:,1);
 trainrmse = 0;
for i = 1:m
    trainrmse_i = abs(pretrain(i,1)-TRAIN_LABEL(i,1));
     
 end
 trainrmse = sqrt((trainrmse_i.^2)./m);

% Calculate test set root mean square error test RMSE
 N = size(TEST_LABEL);
 n = N(:,1);
 testrmse = 0;
 for i = 1:n
     testrmse_i = abs(pretest(i,1)-TEST_LABEL(i,1));
     
 end
 testrmse = sqrt((testrmse_i.^2)./n);

% Show the root mean square error
 str = {['train RMSE = ',num2str(trainrmse)]};
 set(handles.trainrmse,'String',str);
 str = {['test RMSE = ',num2str(testrmse)]};
 set(handles.testrmse,'String',str);

line1 = 'Training & prediction complete！';
mystring{length(mystring)+1,1} = line1;
set(handles.info,'String',mystring);
%% Draw a diagram of the original concentration and the predicted concentration of the training set and the test set
if testflag == 0
    axes(handles.axesC);    % training set
    cla reset;
    view(2);
    reset(gca);
    plot(TrainL,'-o');
    hold on;
    plot(pretrain,'r-^');
    legend('original','predict');
    title('train set');
    xlabel('The samples','FontSize',12);
    ylabel('Concentration','FontSize',12);
    grid on;
else
    axes(handles.axesC);
    cla reset;
    view(2);
    reset(gca);
    plot(TRAIN_LABEL,'-o');
    hold on;
    plot(pretrain,'r-^');
    legend('original','predict');
    title('train set');
    xlabel('The samples','FontSize',12);
    ylabel('Concentration','FontSize',12);
    grid on;
    
    axes(handles.axesB);    % test set
    cla reset;
    view(2);
    reset(gca);
    plot(TEST_LABEL,'-o');
    hold on;
    plot(pretest,'r-^');
    legend('original','predict');
    title('test set');
    xlabel('The samples','FontSize',12);
    ylabel('Concentration','FontSize',12);
    grid on;
end

toc  % End

line1 = '======all done=====';
mystring{length(mystring)+1,1} = line1;
set(handles.info,'String',mystring);

guidata(hObject, handles);

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename,pathname,filterindex] = uiputfile({'*.mat';'*.*'},'save data');

if filterindex
    filename=strcat(pathname,filename);
    global TRAIN_DATA TRAIN_LABEL TEST_DATA TEST_LABEL
    global Model pretrain pretest
    save(filename,...
        'TRAIN_DATA','TRAIN_LABEL','TEST_DATA','TEST_LABEL',...
        'Model','pretrain','pretest');
end

guidata(hObject, handles);

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close;


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
s = sprintf('Our SERS_SVM is Based on libsvm-FarutoUltimate and SVM_GUI3.0\nThanks for faruto and his team\nThanks for Dr.Lin\nLatest modified 2018.03\nBy Silence-唯爱');
msgbox(s,'简介');
% open('readme.txt');
guidata(hObject,handles);


function info_Callback(hObject, eventdata, handles)
% hObject    handle to info (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of info as text
%        str2double(get(hObject,'String')) returns contents of info as a double


% --- Executes during object creation, after setting all properties.
function info_CreateFcn(hObject, eventdata, handles)
% hObject    handle to info (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in xscale.
function xscale_Callback(hObject, eventdata, handles)
% hObject    handle to xscale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns xscale contents as cell array
%        contents{get(hObject,'Value')} returns selected item from xscale
if get(handles.xscale,'Value') == 4
    set(handles.min,'Enable','on');
    set(handles.max,'Enable','on');
else
    set(handles.min,'Enable','off');
    set(handles.max,'Enable','off');
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function xscale_CreateFcn(hObject, eventdata, handles)
% hObject    handle to xscale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in pca.
function pca_Callback(hObject, eventdata, handles)
% hObject    handle to pca (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pca contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pca
if get(handles.pca,'Value') == 1
    set(handles.percent,'Enable','on');
else
    set(handles.percent,'Enable','off');
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function pca_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pca (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in find.
function find_Callback(hObject, eventdata, handles)
% hObject    handle to find (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns find contents as cell array
%        contents{get(hObject,'Value')} returns selected item from find
if get(handles.find,'Value') == 1
    set(handles.gridcmin,'Enable','on');
    set(handles.gridcmax,'Enable','on');
    set(handles.gridgmin,'Enable','on');
    set(handles.gridgmax,'Enable','on');
    set(handles.gridcstep,'Enable','on');
    set(handles.gridgstep,'Enable','on');
else
    set(handles.gridcmin,'Enable','off');
    set(handles.gridcmax,'Enable','off');
    set(handles.gridgmin,'Enable','off');
    set(handles.gridgmax,'Enable','off');
    set(handles.gridcstep,'Enable','off');
    set(handles.gridgstep,'Enable','off');
end
if get(handles.find,'Value') == 2 || get(handles.find,'Value') == 3 || get(handles.find,'Value') == 4
    set(handles.maxgen,'Enable','on');
    set(handles.sizepop,'Enable','on');
    set(handles.cmin,'Enable','on');
    set(handles.cmax,'Enable','on');
    set(handles.gmin,'Enable','on');
    set(handles.gmax,'Enable','on');
else
    set(handles.maxgen,'Enable','off');
    set(handles.sizepop,'Enable','off');
    set(handles.cmin,'Enable','off');
    set(handles.cmax,'Enable','off');
    set(handles.gmin,'Enable','off');
    set(handles.gmax,'Enable','off');
end
if get(handles.find,'Value') == 4
    set(handles.pmin,'Enable','on');
    set(handles.pmax,'Enable','on');
else
    set(handles.pmin,'Enable','off');
    set(handles.pmax,'Enable','off');
end

guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function find_CreateFcn(hObject, eventdata, handles)
% hObject    handle to find (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function min_Callback(hObject, eventdata, handles)
% hObject    handle to min (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of min as text
%        str2double(get(hObject,'String')) returns contents of min as a double
tempmin = get(handles.min,'String');
tempmin = str2num(tempmin);
tempmax = get(handles.max,'String');
tempmax = str2num(tempmax);
if isempty(tempmin)
    warndlg('the lower bound must be a numerical number!','warning');
end
if tempmin >= tempmax
    warndlg('the lower bound must be less than the upper bound!','warning');
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function min_CreateFcn(hObject, eventdata, handles)
% hObject    handle to min (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function max_Callback(hObject, eventdata, handles)
% hObject    handle to max (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of max as text
%        str2double(get(hObject,'String')) returns contents of max as a double
tempmin = get(handles.min,'String');
tempmin = str2num(tempmin);
tempmax = get(handles.max,'String');
tempmax = str2num(tempmax);
if isempty(tempmax)
    warndlg('the upper bound must be a numerical number!','warning');
end
if tempmin >= tempmax
    warndlg('the lower bound must be less than the upper bound!','warning');
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function max_CreateFcn(hObject, eventdata, handles)
% hObject    handle to max (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function percent_Callback(hObject, eventdata, handles)
% hObject    handle to percent (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of percent as text
%        str2double(get(hObject,'String')) returns contents of percent as a double
per = str2num(get(handles.percent,'String'));
if isempty(per) || per < 0 || per >100
    warndlg('the percentage must be a numerical number(0-100)!','warning');
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function percent_CreateFcn(hObject, eventdata, handles)
% hObject    handle to percent (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function gridcmin_Callback(hObject, eventdata, handles)
% hObject    handle to gridcmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of gridcmin as text
%        str2double(get(hObject,'String')) returns contents of gridcmin as a double
tempmin = get(handles.gridcmin,'String');
tempmin = str2num(tempmin);
tempmax = get(handles.gridcmax,'String');
tempmax = str2num(tempmax);
if isempty(tempmin)
    warndlg('the lower bound must be a numerical number!','warning');
end
if tempmin >= tempmax
    warndlg('the lower bound must be less than the upper bound!','warning');
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function gridcmin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to gridcmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function gridcmax_Callback(hObject, eventdata, handles)
% hObject    handle to gridcmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of gridcmax as text
%        str2double(get(hObject,'String')) returns contents of gridcmax as a double
tempmin = get(handles.gridcmin,'String');
tempmin = str2num(tempmin);
tempmax = get(handles.gridcmax,'String');
tempmax = str2num(tempmax);
if isempty(tempmax)
    warndlg('the upper bound must be a numerical number!','warning');
end
if tempmin >= tempmax
    warndlg('the lower bound must be less than the upper bound!','warning');
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function gridcmax_CreateFcn(hObject, eventdata, handles)
% hObject    handle to gridcmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function gridgmin_Callback(hObject, eventdata, handles)
% hObject    handle to gridgmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of gridgmin as text
%        str2double(get(hObject,'String')) returns contents of gridgmin as a double
tempmin = get(handles.gridgmin,'String');
tempmin = str2num(tempmin);
tempmax = get(handles.gridgmax,'String');
tempmax = str2num(tempmax);
if isempty(tempmin)
    warndlg('the lower bound must be a numerical number!','warning');
end
if tempmin >= tempmax
    warndlg('the lower bound must be less than the upper bound!','warning');
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function gridgmin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to gridgmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function gridgmax_Callback(hObject, eventdata, handles)
% hObject    handle to gridgmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of gridgmax as text
%        str2double(get(hObject,'String')) returns contents of gridgmax as a double
tempmin = get(handles.gridgmin,'String');
tempmin = str2num(tempmin);
tempmax = get(handles.gridgmax,'String');
tempmax = str2num(tempmax);
if isempty(tempmax)
    warndlg('the upper bound must be a numerical number!','warning');
end
if tempmin >= tempmax
    warndlg('the lower bound must be less than the upper bound!','warning');
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function gridgmax_CreateFcn(hObject, eventdata, handles)
% hObject    handle to gridgmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function gridgstep_Callback(hObject, eventdata, handles)
% hObject    handle to gridgstep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of gridgstep as text
%        str2double(get(hObject,'String')) returns contents of gridgstep as a double
temp = get(handles.gridgstep,'String');
temp = str2num(temp);
if isempty(temp) || temp<=0
    warndlg('the grid g step must be a numerical number(>0)!','warning');
end

% --- Executes during object creation, after setting all properties.
function gridgstep_CreateFcn(hObject, eventdata, handles)
% hObject    handle to gridgstep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function gridcstep_Callback(hObject, eventdata, handles)
% hObject    handle to gridcstep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of gridcstep as text
%        str2double(get(hObject,'String')) returns contents of gridcstep as a double
temp = get(handles.gridcstep,'String');
temp = str2num(temp);
if isempty(temp) || temp <= 0
    warndlg('the grid c step must be a numerical number(>0)!','warning');
end

guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function gridcstep_CreateFcn(hObject, eventdata, handles)
% hObject    handle to gridcstep (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function v_Callback(hObject, eventdata, handles)
% hObject    handle to v (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of v as text
%        str2double(get(hObject,'String')) returns contents of v as a double
temp = get(handles.v,'String');
temp = str2num(temp);
if isempty(temp) || temp<=2 
    warndlg('the fold v must be a numerical number(>=3)!','warning');
end

% --- Executes during object creation, after setting all properties.
function v_CreateFcn(hObject, eventdata, handles)
% hObject    handle to v (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function cmin_Callback(hObject, eventdata, handles)
% hObject    handle to cmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of cmin as text
%        str2double(get(hObject,'String')) returns contents of cmin as a double
tempmin = get(handles.cmin,'String');
tempmin = str2num(tempmin);
tempmax = get(handles.cmax,'String');
tempmax = str2num(tempmax);
if isempty(tempmin)
    warndlg('the lower bound must be a numerical number!','warning');
end
if tempmin >= tempmax
    warndlg('the lower bound must be less than the upper bound!','warning');
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function cmin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to cmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function cmax_Callback(hObject, eventdata, handles)
% hObject    handle to cmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of cmax as text
%        str2double(get(hObject,'String')) returns contents of cmax as a double
tempmin = get(handles.cmin,'String');
tempmin = str2num(tempmin);
tempmax = get(handles.cmax,'String');
tempmax = str2num(tempmax);
if isempty(tempmax)
    warndlg('the upper bound must be a numerical number!','warning');
end
if tempmin >= tempmax
    warndlg('the lower bound must be less than the upper bound!','warning');
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function cmax_CreateFcn(hObject, eventdata, handles)
% hObject    handle to cmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function gmin_Callback(hObject, eventdata, handles)
% hObject    handle to gmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of gmin as text
%        str2double(get(hObject,'String')) returns contents of gmin as a double
tempmin = get(handles.gmin,'String');
tempmin = str2num(tempmin);
tempmax = get(handles.gmax,'String');
tempmax = str2num(tempmax);
if isempty(tempmin)
    warndlg('the lower bound must be a numerical number!','warning');
end
if tempmin >= tempmax
    warndlg('the lower bound must be less than the upper bound!','warning');
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function gmin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to gmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function gmax_Callback(hObject, eventdata, handles)
% hObject    handle to gmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of gmax as text
%        str2double(get(hObject,'String')) returns contents of gmax as a double
tempmin = get(handles.gmin,'String');
tempmin = str2num(tempmin);
tempmax = get(handles.gmax,'String');
tempmax = str2num(tempmax);
if isempty(tempmax)
    warndlg('the upper bound must be a numerical number!','warning');
end
if tempmin >= tempmax
    warndlg('the lower bound must be less than the upper bound!','warning');
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function gmax_CreateFcn(hObject, eventdata, handles)
% hObject    handle to gmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function maxgen_Callback(hObject, eventdata, handles)
% hObject    handle to maxgen (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of maxgen as text
%        str2double(get(hObject,'String')) returns contents of maxgen as a double
temp = get(handles.maxgen,'String');
temp = str2num(temp);
if isempty(temp) || temp<=0
    warndlg('the maxgen must be a numerical number(>0)!','warning');
end

% --- Executes during object creation, after setting all properties.
function maxgen_CreateFcn(hObject, eventdata, handles)
% hObject    handle to maxgen (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function sizepop_Callback(hObject, eventdata, handles)
% hObject    handle to sizepop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of sizepop as text
%        str2double(get(hObject,'String')) returns contents of sizepop as a double
temp = get(handles.sizepop,'String');
temp = str2num(temp);
if isempty(temp) || temp<=0
    warndlg('the sizepop must be a numerical number(>0)!','warning');
end

% --- Executes during object creation, after setting all properties.
function sizepop_CreateFcn(hObject, eventdata, handles)
% hObject    handle to sizepop (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function savepicB_Callback(hObject, eventdata, handles)
% hObject    handle to savepicB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axesB);
if isempty(handles.axesB)
    return;
end
newfig = figure;
set(newfig,'Visible','off');
newaxes = copyobj(handles.axesB,newfig);
set(newaxes,'Units','default','Position','default');

[filename,pathname,filterindex] = uiputfile({'*.jpg';'*.*'},'save as');

if filterindex
    filename=strcat(pathname,filename);
    pic = getframe(newfig);
    pic = frame2im(pic);
    imwrite(pic, filename);
end

guidata(hObject,handles);

% --------------------------------------------------------------------
function clearpicB_Callback(hObject, eventdata, handles)
% hObject    handle to clearpicB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
cla(handles.axesB,'reset');
guidata(hObject,handles);

% --------------------------------------------------------------------
function savepicA_Callback(hObject, eventdata, handles)
% hObject    handle to savepicA (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axesA);
if isempty(handles.axesA)
    return;
end
newfig = figure;
set(newfig,'Visible','off');
newaxes = copyobj(handles.axesA,newfig);
set(newaxes,'Units','default','Position','default');

[filename,pathname,filterindex] = uiputfile({'*.jpg';'*.*'},'save as');

if filterindex
    filename=strcat(pathname,filename);
    pic = getframe(newfig);
    pic = frame2im(pic);
    imwrite(pic, filename);
end

guidata(hObject,handles);

% --------------------------------------------------------------------
function clearpicA_Callback(hObject, eventdata, handles)
% hObject    handle to clearpicA (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
cla(handles.axesA,'reset');
guidata(hObject,handles);

% --------------------------------------------------------------------
function picA_Callback(hObject, eventdata, handles)
% hObject    handle to picA (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function picB_Callback(hObject, eventdata, handles)
% hObject    handle to picB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function clearinfo_Callback(hObject, eventdata, handles)
% hObject    handle to clearinfo (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function clear_Callback(hObject, eventdata, handles)
% hObject    handle to clear (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.info,'String','');
guidata(hObject,handles);



function pmin_Callback(hObject, eventdata, handles)
% hObject    handle to pmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of pmin as text
%        str2double(get(hObject,'String')) returns contents of pmin as a double


% --- Executes during object creation, after setting all properties.
function pmin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pmin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function pmax_Callback(hObject, eventdata, handles)
% hObject    handle to pmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of pmax as text
%        str2double(get(hObject,'String')) returns contents of pmax as a double


% --- Executes during object creation, after setting all properties.
function pmax_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pmax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in yscale.
function yscale_Callback(hObject, eventdata, handles)
% hObject    handle to yscale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns yscale contents as cell array
%        contents{get(hObject,'Value')} returns selected item from yscale
if get(handles.yscale,'Value') == 4
    set(handles.ymin,'Enable','on');
    set(handles.ymax,'Enable','on');
else
    set(handles.ymin,'Enable','off');
    set(handles.ymax,'Enable','off');
end
guidata(hObject, handles);

% --- Executes during object creation, after setting all properties.
function yscale_CreateFcn(hObject, eventdata, handles)
% hObject    handle to yscale (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ymin_Callback(hObject, eventdata, handles)
% hObject    handle to ymin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ymin as text
%        str2double(get(hObject,'String')) returns contents of ymin as a double
tempmin = get(handles.ymin,'String');
tempmin = str2num(tempmin);
tempmax = get(handles.ymax,'String');
tempmax = str2num(tempmax);
if isempty(tempmin)
    warndlg('the lower bound must be a numerical number!','warning');
end
if tempmin >= tempmax
    warndlg('the lower bound must be less than the upper bound!','warning');
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function ymin_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ymin (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function ymax_Callback(hObject, eventdata, handles)
% hObject    handle to ymax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of ymax as text
%        str2double(get(hObject,'String')) returns contents of ymax as a double
tempmin = get(handles.ymin,'String');
tempmin = str2num(tempmin);
tempmax = get(handles.ymax,'String');
tempmax = str2num(tempmax);
if isempty(tempmax)
    warndlg('the upper bound must be a numerical number!','warning');
end
if tempmin >= tempmax
    warndlg('the lower bound must be less than the upper bound!','warning');
end
guidata(hObject,handles);

% --- Executes during object creation, after setting all properties.
function ymax_CreateFcn(hObject, eventdata, handles)
% hObject    handle to ymax (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function savepicC_Callback(hObject, eventdata, handles)
% hObject    handle to savepicC (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axesC);
if isempty(handles.axesC)
    return;
end
newfig = figure;
set(newfig,'Visible','off');
newaxes = copyobj(handles.axesC,newfig);
set(newaxes,'Units','default','Position','default');

[filename,pathname,filterindex] = uiputfile({'*.jpg';'*.*'},'save as');

if filterindex
    filename=strcat(pathname,filename);
    pic = getframe(newfig);
    pic = frame2im(pic);
    imwrite(pic, filename);
end

guidata(hObject,handles);

% --------------------------------------------------------------------
function clearpicC_Callback(hObject, eventdata, handles)
% hObject    handle to clearpicC (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
cla(handles.axesC,'reset');
guidata(hObject,handles);

% --------------------------------------------------------------------
function picC_Callback(hObject, eventdata, handles)
% hObject    handle to picC (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on mouse press over axes background.
function axesB_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to axesB (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function axesA_CreateFcn(hObject, eventdata, handles)
% hObject    handle to axesA (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate axesA
