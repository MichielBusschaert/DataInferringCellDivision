function hyp_struct = WrapVec2Struct(hyp_vec)
    hyp_struct = struct('logs',hyp_vec(1),'logl',hyp_vec(2),'logsn',hyp_vec(3));
end