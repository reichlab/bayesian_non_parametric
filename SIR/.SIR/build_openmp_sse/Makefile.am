



AUTOMAKE_OPTIONS = subdir-objects

# compile and link flags
AM_CPPFLAGS = -Isrc $(OPENMP_CPPFLAGS)
AM_CXXFLAGS = $(OPENMP_CXXFLAGS)
AM_LDFLAGS = $(OPENMP_LDFLAGS)

# CUDA files setup
NVCPPFLAGS = $(AM_CPPFLAGS) $(CPPFLAGS) $(DEFS) -DBOOST_NOINLINE
NVCXXFLAGS = -w -arch sm_20 -Xcompiler="$(AM_CXXFLAGS) $(CXXFLAGS)"
LINK = $(CXXLINK) # force C++ linker for CUDA files

# libraries
lib_LIBRARIES = libbi.a
libbi_a_SOURCES = \
  src/bi/bi.cpp \
  src/bi/adapter/AdapterFactory.cpp \
  src/bi/adapter/GaussianAdapter.cpp \
  src/bi/netcdf/KalmanFilterNetCDFBuffer.cpp \
  src/bi/netcdf/netcdf.cpp \
  src/bi/netcdf/NetCDFBuffer.cpp \
  src/bi/netcdf/OptimiserNetCDFBuffer.cpp \
  src/bi/netcdf/ParticleFilterNetCDFBuffer.cpp \
  src/bi/netcdf/MCMCNetCDFBuffer.cpp \
  src/bi/netcdf/SMCNetCDFBuffer.cpp \
  src/bi/netcdf/SimulatorNetCDFBuffer.cpp \
  src/bi/netcdf/InputNetCDFBuffer.cpp \
  src/bi/null/InputNullBuffer.cpp \
  src/bi/null/KalmanFilterNullBuffer.cpp \
  src/bi/null/MCMCNullBuffer.cpp \
  src/bi/null/OptimiserNullBuffer.cpp \
  src/bi/null/ParticleFilterNullBuffer.cpp \
  src/bi/null/SimulatorNullBuffer.cpp \
  src/bi/null/SMCNullBuffer.cpp \
  src/bi/cache/Cache.cpp \
  src/bi/host/math/cblas.cpp \
  src/bi/host/math/lapack.cpp \
  src/bi/host/math/qrupdate.cpp \
  src/bi/host/ode/IntegratorConstants.cpp \
  src/bi/host/random/RandomHost.cpp \
  src/bi/misc/omp.cpp \
  src/bi/mpi/mpi.cpp \
  src/bi/random/Random.cpp \
  src/bi/resampler/ResamplerFactory.cpp \
  src/bi/stopper/StopperFactory.cpp

if ENABLE_SSE
libbi_a_SOURCES += 
endif

if ENABLE_CUDA
libbi_a_SOURCES += \
  src/bi/cuda/device.cu \
  src/bi/cuda/math/cublas.cu \
  src/bi/cuda/random/RandomGPU.cu \
  src/bi/cuda/random/RandomKernel.cu
endif

if ENABLE_MPI
libbi_a_SOURCES += \
  src/bi/mpi/adapter/DistributedAdapterFactory.cpp \
  src/bi/mpi/resampler/DistributedResamplerFactory.cpp \
  src/bi/mpi/stopper/DistributedStopperFactory.cpp \
  src/bi/mpi/Client.cpp \
  src/bi/mpi/Server.cpp \
  src/bi/mpi/TreeNetworkNode.cpp
endif

# programs
bin_PROGRAMS =  optimise_cpu optimise_gpu filter_cpu filter_gpu sample_cpu sample_gpu test_cpu test_gpu test_resampler_cpu test_resampler_gpu


optimise_cpu_LDADD = $(DEPS_LIBS) libbi.a
optimise_cpu_SOURCES = src/optimise_cpu.cpp src/model/ModelSIR.cpp
optimise_gpu_LDADD = $(DEPS_LIBS) libbi.a
optimise_gpu_SOURCES = src/optimise_gpu.cu  src/model/ModelSIR.cpp

filter_cpu_LDADD = $(DEPS_LIBS) libbi.a
filter_cpu_SOURCES = src/filter_cpu.cpp src/model/ModelSIR.cpp
filter_gpu_LDADD = $(DEPS_LIBS) libbi.a
filter_gpu_SOURCES = src/filter_gpu.cu  src/model/ModelSIR.cpp

sample_cpu_LDADD = $(DEPS_LIBS) libbi.a
sample_cpu_SOURCES = src/sample_cpu.cpp src/model/ModelSIR.cpp
sample_gpu_LDADD = $(DEPS_LIBS) libbi.a
sample_gpu_SOURCES = src/sample_gpu.cu  src/model/ModelSIR.cpp

test_cpu_LDADD = $(DEPS_LIBS) libbi.a
test_cpu_SOURCES = src/test_cpu.cpp src/model/ModelSIR.cpp
test_gpu_LDADD = $(DEPS_LIBS) libbi.a
test_gpu_SOURCES = src/test_gpu.cu  src/model/ModelSIR.cpp

test_resampler_cpu_LDADD = $(DEPS_LIBS) libbi.a
test_resampler_cpu_SOURCES = src/test_resampler_cpu.cpp src/model/ModelSIR.cpp
test_resampler_gpu_LDADD = $(DEPS_LIBS) libbi.a
test_resampler_gpu_SOURCES = src/test_resampler_gpu.cu  src/model/ModelSIR.cpp


# other
dist_noinst_SCRIPTS = autogen.sh

# command line options
if ENABLE_SINGLE
CPPFLAGS += -DENABLE_SINGLE
endif

if ENABLE_CUDA
CPPFLAGS += -DENABLE_CUDA
# ensure dependency files included
include src/$(DEPDIR)/optimise_gpu.Po

include src/$(DEPDIR)/filter_gpu.Po

include src/$(DEPDIR)/sample_gpu.Po

include src/$(DEPDIR)/test_gpu.Po

include src/$(DEPDIR)/test_resampler_gpu.Po

endif

if ENABLE_CUDA_FAST_MATH
CPPFLAGS += -DCUDA_FAST_MATH=1
else
CPPFLAGS += -DCUDA_FAST_MATH=0
endif

if ENABLE_GPU_CACHE
CPPFLAGS += -DENABLE_GPU_CACHE
endif

if ENABLE_AVX
CPPFLAGS += -DENABLE_AVX
CXXFLAGS += -mavx
endif

if ENABLE_SSE
CPPFLAGS += -DENABLE_SSE
CXXFLAGS += -msse3
endif

if ENABLE_OPENMP
CPPFLAGS += -DENABLE_OPENMP
endif

if ENABLE_MPI
CPPFLAGS += -DENABLE_MPI
endif

if ENABLE_VAMPIR
CPPFLAGS += -DENABLE_VAMPIR -DVTRACE
endif

if ENABLE_GPERFTOOLS
CPPFLAGS += -DENABLE_GPERFTOOLS
endif

# suffix rules for CUDA files
.cu.o:
	depbase=`echo $@ | sed 's|[^/]*$$|.deps/&|;s|\.o$$||'` && \
	srcbase=`echo $@ | sed 's|/[^/]*$$||'` && \
	perl nvcc_wrapper.pl $(NVCC) -M $(NVCXXFLAGS) $(NVCPPFLAGS) -odir $$srcbase -o $$depbase.Tpo $< && \
	perl nvcc_wrapper.pl $(NVCC) -c $(NVCXXFLAGS) $(NVCPPFLAGS) -o $@ $< && \
	cat $$depbase.Tpo > $$depbase.Po && \
	rm -f $$depbase.Tpo

.cu.lo:
	depbase=`echo $@ | sed 's|[^/]*$$|.deps/&|;s|\.lo$$||'` && \
	srcbase=`echo $@ | sed 's|/[^/]*$$||'` && \
	perl nvcc_wrapper.pl $(NVCC) -M $(NVCXXFLAGS) $(NVCPPFLAGS) -odir $$srcbase -o $$depbase.Tpo $< && \
	$(LIBTOOL) --tag=CC --mode=compile perl nvcc_wrapper.pl $(NVCC) -c $(NVCXXFLAGS) $(NVCPPFLAGS) -o $@ $< && \
	cat $$depbase.Tpo > $$depbase.Plo && \
	rm -f $$depbase.Tpo
