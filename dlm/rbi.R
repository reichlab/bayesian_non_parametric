library('rbi')

model_file <- system.file(package="rbi", "SIR.bi")
SIRmodel <- bi_model(model_file) # load model


out<-rbi::libbi(SIRmodel,"/Users/gcgibson/Desktop/bayesian_non_parametric/LibBi/script/libbi")

