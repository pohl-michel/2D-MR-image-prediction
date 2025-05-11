function set_python_path_all_workers()
% Adding the transformer python path to all workers

    pool = gcp();  % Get parallel pool
    moduleDir = get_python_transformers_module_dir();

    % Setup Python path on all workers
    futures = parfevalOnAll(@() ...
        insert(py.sys.path, int32(0), moduleDir), 0);
    
    % Wait for all workers to be ready
    wait(futures);    

end