function hyp_vec = WrapStruct2Vec(hyp_struct)
    hyp_vec = [hyp_struct.logs;hyp_struct.logl;hyp_struct.logsn];
end