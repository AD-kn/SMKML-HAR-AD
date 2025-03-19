function update_data_labels()
   
   

    data_path = fullfile(pwd, '..', filesep, 'data', filesep);
    addpath(data_path);
    lib_path = fullfile(pwd, '..', filesep, 'lib', filesep);
    addpath(lib_path);


    datasetCandi = dir(fullfile(data_path, '*.mat'));
    
    
    for iDataSet = 1:length(datasetCandi)
        data_name = datasetCandi(iDataSet).name;
        data_file_path = fullfile(data_path, data_name);
        
        toy_data = load(data_file_path);
        
      
        if isfield(toy_data, 'Y')
           
            y = double(toy_data.Y);  
            X = toy_data.X;        
            
        
            save(data_file_path, 'X', 'y');  
            disp(['Updated Y to y in ', data_name]); 
        else
            disp(['No Y field found in ', data_name]); 
        end
    end
end
