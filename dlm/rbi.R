library('rbi')

model_file <- system.file(package="rbi", "SIR.bi")
SIRmodel <- bi_model(model_file) # load model


bi_object <- rbi::libbi(model =SIRmodel ,path_to_libbi ='/Users/gcgibson/Desktop/bayesian_non_parametric/LibBi/script/libbi' )

rbi::run(bi_object, client="sample")