module LIBSVM

export svm_train, svm_predict, svm_free, get_dual_variables, 
       get_primal_variables, parse_libSVM

const CSVC = Int32(0)
const NuSVC = Int32(1)
const OneClassSVM = Int32(2)
const EpsilonSVR = Int32(3)
const NuSVR = Int32(4)

const Linear = Int32(0)
const Polynomial = Int32(1)
const RBF = Int32(2)
const Sigmoid = Int32(3)
const Precomputed = Int32(4)

verbosity = false

immutable SVMNode
  index::Int32
  value::Float64
end

immutable SVMProblem
  l::Int32
  y::Ptr{Float64}
  x::Ptr{Ptr{SVMNode}}
  W::Ptr{Float64}
end

immutable SVMParameter
  svm_type::Int32
  kernel_type::Int32
  degree::Int32
  gamma::Float64
  coef0::Float64

  cache_size::Float64
  eps::Float64
  C::Float64
  nr_weight::Int32
  weight_label::Ptr{Int32}
  weight::Ptr{Float64}
  nu::Float64
  p::Float64
  shrinking::Int32
  probability::Int32
end

immutable SVMModel
  param::SVMParameter
  nr_class::Int32
  l::Int32
  SV::Ptr{Ptr{SVMNode}}
  sv_coef::Ptr{Ptr{Float64}}
  rho::Ptr{Float64}
  probA::Ptr{Float64}
  probB::Ptr{Float64}
  sv_indices::Ptr{Int32}

  label::Ptr{Int32}
  nSV::Ptr{Int32}

  free_sv::Int32
end

type JuliaSVMModel{T}
  ptr::Ptr{SVMModel}
  param::Vector{SVMParameter}

  # Prevent these from being garbage collected
  problem::Vector{SVMProblem} 
  nodes::Array{SVMNode}
  nodeptr::Vector{Ptr{SVMNode}}
  W::Vector{Float64}

  labels::Vector{T}
  weight_labels::Vector{Int32}
  weights::Vector{Float64}
  nfeatures::Int
  ninstances::Int
  verbose::Bool
end

let libsvm = C_NULL
  global get_libsvm
  function get_libsvm()
    if libsvm == C_NULL
      path = ""
      if OS_NAME == :Darwin
        path=joinpath(Pkg.dir(), "LIBSVM", "deps", "libsvm.so.2")
      end
      if OS_NAME == :Windows
        path=joinpath(Pkg.dir(), "LIBSVM", "deps", "libsvm$(WORD_SIZE).dll")
      end
      libsvm = Libdl.dlopen(path)
      ccall(Libdl.dlsym(libsvm, :svm_set_print_string_function), Void,
          (Ptr{Void},), cfunction(svmprint, Void, (Ptr{UInt8},)))
    end
    libsvm
  end
end

macro cachedsym(symname)
  cached = gensym()
  quote
    let $cached = C_NULL
      global ($symname)
      ($symname)() = ($cached) == C_NULL ?
          ($cached = Libdl.dlsym(get_libsvm(), $(string(symname)))) : $cached
    end
  end
end

@cachedsym svm_train
@cachedsym svm_predict_values
@cachedsym svm_predict_probability
@cachedsym svm_free_model_content

function instances2nodes{U<:Real}(instances::Union{AbstractMatrix{U},
  AbstractVector{U}})

  nfeatures = size(instances, 1)
  ninstances = size(instances, 2)
  nodeptrs = Array(Ptr{SVMNode}, ninstances)
  nodes = Array(SVMNode, nfeatures + 1, ninstances)

  for i=1:ninstances
    k = 1
    for j=1:nfeatures
      nodes[k, i] = SVMNode(Int32(j), Float64(instances[j, i]))
      k += 1
    end
    nodes[k, i] = SVMNode(Int32(-1), NaN)
    nodeptrs[i] = pointer(nodes, (i-1)*(nfeatures+1)+1)
  end
  (nodes, nodeptrs)
end

function instances2nodes{U<:Real}(instances::SparseMatrixCSC{U})
  ninstances = size(instances, 2)
  nodeptrs = Array(Ptr{SVMNode}, ninstances)
  nodes = Array(SVMNode, nnz(instances)+ninstances)

  j = 1
  k = 1
  for i=1:ninstances
    nodeptrs[i] = pointer(nodes, k)
    while j < instances.colptr[i+1]
      val = instances.nzval[j]
      nodes[k] = SVMNode(Int32(instances.rowval[j]), Float64(val))
      k += 1
      j += 1
    end
    nodes[k] = SVMNode(Int32(-1), NaN)
    k += 1
  end
  (nodes, nodeptrs)
end

function svmprint(str::Ptr{UInt8})
  if verbosity::Bool
    print(bytestring(str))
  end
  nothing
end

function svm_train{U<:Real}(idx::Array{Float64,1},
  instances::Union{AbstractMatrix{U},SparseMatrixCSC{U}},
  W::Array{Float64,1} = ones(size(instances,2)); 
  svm_type::Int32=CSVC,
  kernel_type::Int32=Linear, 
  degree::Integer=3,
  gamma::Float64=1.0/size(instances, 1), 
  coef0::Float64=0.0, 
  C::Float64=1.0, 
  nu::Float64=0.5, 
  p::Float64=0.1,
  cache_size::Float64=100.0, 
  eps::Float64=0.001, 
  shrinking::Bool=true,
  probability_estimates::Bool=false,
  weights::Array{Float64,1}=Float64[],
  weight_labels::Array{Int32,1}=Int32[],
  verbose::Bool=false)

  global verbosity

  param = Array(SVMParameter, 1)
  param[1] = SVMParameter(svm_type, 
                          kernel_type, 
                          Int32(degree), 
                          Float64(gamma),
                          coef0, 
                          cache_size, 
                          eps, 
                          C, 
                          Int32(length(weights)),
                          pointer(weight_labels), 
                          pointer(weights), 
                          nu, 
                          p, 
                          Int32(shrinking),
                          Int32(probability_estimates))

  # Construct SVMProblem
  (nodes, nodeptrs) = instances2nodes(instances)

  problem = SVMProblem[SVMProblem(Int32(size(instances, 2)), 
                       pointer(idx),
                       pointer(nodeptrs), 
                       pointer(W))]

  verbosity = verbose
  ptr = ccall(svm_train(), Ptr{SVMModel}, (Ptr{SVMProblem},
      Ptr{SVMParameter}), problem, param)

  JuliaSVMModel(ptr,
                param,
                problem,
                nodes, 
                nodeptrs,
                W,
                idx,
                weight_labels,
                weights,
                size(instances,1),
                size(instances,2),
                verbose)

end

function svm_predict{U<:Real}(model::JuliaSVMModel,
  instances::AbstractVector{U})
  
  mdl = unsafe_load(model.ptr)
  (nodes, nodeptrs) = instances2nodes(instances)
  decvalues = nothing
  svm_pred = nothing

  # Check if probability_estimates has been set to positive.
  if UInt(mdl.probA) == 0
    svm_pred = svm_predict_values()
    if mdl.nr_class == 2
      decvalues = Inf*ones(mdl.nr_class - 1, 1)
    else
      decvalues = Inf*ones(mdl.nr_class, 1)
    end
  else
    svm_pred = svm_predict_probability()
    decvalues = Array(Float64, mdl.nr_class, 1)
  end

  output = ccall(svm_pred, 
                 Float64, 
                 (Ptr{SVMModel}, Ptr{SVMNode}, 
                 Ptr{Float64}),
                 model.ptr, 
                 nodeptrs[1], 
                 pointer(decvalues))

  (Int(output), decvalues)

end

function get_dual_variables_raw(model)
  
  In = Int[]
  v = zeros(0)

  mdl = unsafe_load(model.ptr)
  for i = 1:mdl.l; push!(v,  unsafe_load(unsafe_load(mdl.sv_coef),i)); end
  for i = 1:mdl.l; push!(In, unsafe_load(mdl.sv_indices,i)); end

  return (In,v)

end

function get_dual_variables(model)
  
  (In,v) = get_dual_variables_raw(model)
  return sparsevec(In,v, model.ninstances)

end

function get_primal_variables(model)

  if model.param[1].kernel_type != 0
    error("Primal Variables are only available for linear kernels")
  end
  mdl = unsafe_load(model.ptr)

  (In,v) = get_dual_variables_raw(model)
  instances = nodes2instance(model)

  n = size(instances,1)
  x = zeros(n)

  # Indicies may be out of order
  for i = 1:size(instances,2)
    x = x + v[i]*instances[:,i]
  end

  (x, unsafe_load(mdl.rho))

end

function nodes2instance(model)

  m = unsafe_load(model.ptr)
  J = Int[]
  I = Int[]
  V = zeros(0)
  for i = 1:m.l
    insᵢ = unsafe_load(m.SV,i)
    j = 1
    for j = 1:model.nfeatures
      nodeᵢ = unsafe_load(insᵢ,j)
      if nodeᵢ.index == -1
        break
      end
      push!(J,i); push!(I,nodeᵢ.index); push!(V,nodeᵢ.value)
    end
  end

  SV = sparse(I,J,V,model.nfeatures, m.l)

  SV

end

function parse_libSVM(filename)

  d = readdlm(filename)
  # File, number of datapoints
  n = size(d,1) # Number of Data points
  c = zeros(n)
  d = d'
  J   = Array(Int,0)
  I   = Array(Int,0)
  Val = Array(Float64,0)

  for i = 1:n
    di = d[:,i]
    c[i] = di[1];
    for f = di[2:end]
      if length(f) != 0
        ab = split(f, ':')
        push!(J, i)
        push!(I, parse(Int, ab[1]))
        push!(Val, float(ab[2]))
      end
    end
  end

  return (c, sparse(I,J,Val))

end

svm_free(model::SVMModel) = ccall(svm_free_model_content(), 
                                  Void, 
                                  (Ptr{Void},),
                                  model.ptr)

end