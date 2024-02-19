struct AcquisitionPlanCPUMultithreaded{S,DS,CS,P,P2,PS,T} <: AbstractAcquisitionPlan where T
    system::S
    signal_length::Int
    sampling_freq::typeof(1.0Hz)
    dopplers::DS
    codes_freq_domain::CS
    signal_baseband_arenas::T
    signal_baseband_freq_domain_arenas::T
    code_freq_baseband_freq_domain_arenas::T
    code_baseband_arenas::T
    signal_powers::Vector{Matrix{Float32}}
    forward_fft_plan::P
    inverse_fft_plan::P2
    avail_prn_channels::PS
    compensate_doppler_code::Bool
    noncoherent_rounds::Int
    n_threads::Int
    flip_correlation::Bool
end


function AcquisitionPlanCPUMultithreaded(
    system,
    signal_length,
    sampling_freq;
    max_doppler = 7000Hz,
    dopplers = -max_doppler:1/3/(signal_length/sampling_freq):max_doppler,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
    compensate_doppler_code = false,
    noncoherent_rounds = 1,
    n_threads=Threads.nthreads(),
    flip_correlation= false,
)
#=     signal_baseband,
    signal_baseband_freq_domain,
    code_freq_baseband_freq_domain,
    code_baseband,
    codes_freq_domain,
    fft_plan = common_buffers(system, signal_length, sampling_freq, prns, fft_flag)

 =#

    #We merge common_buffers here, its a 'misleading' function as it actually do a lot of things at once 
    #instead of just allocating several work buffers

    #Preallocate baseband buffers and resulting correlator output



    signal_baseband_arenas = [
        AllocBuffer() for _ in 1:n_threads
    ]
#=     signal_baseband_freq_domain = similar(signal_baseband)
    code_freq_baseband_freq_domain = similar(signal_baseband)
    signal_baseband_freq_domain = similar(signal_baseband) =#

    signal_baseband_freq_domain_arenas = [
        AllocBuffer() for _ in 1:n_threads
    ]

    code_freq_baseband_freq_domain_arenas = [
        AllocBuffer() for _ in 1:n_threads
    ]

    code_baseband_arenas = [
        AllocBuffer() for _ in 1:n_threads
    ]
    
    forward_fft_plan = plan_fft(Vector{ComplexF32}(undef, signal_length); flags = fft_flag)
    inverse_fft_plan = plan_ifft(Vector{ComplexF32}(undef, signal_length); flags = fft_flag)

    codes = [gen_code(signal_length, system, sat_prn, sampling_freq) for sat_prn in prns]
    codes_freq_domain = [forward_fft_plan * code for code in codes]

    Δt = signal_length / sampling_freq
    code_interval = get_code_length(system) / get_code_frequency(system)
#=     signal_powers = [
        Matrix{Float32}(
            undef,
            ceil(Int, sampling_freq * min(Δt, code_interval)),
            length(dopplers),
        ) for _ in prns
    ]
 =#

    signal_powers = [
            zeros(Float32,
                ceil(Int, sampling_freq * min(Δt, code_interval)),
                length(dopplers),
            ) for _ in prns
        ]


        AcquisitionPlanCPUMultithreaded(
        system,
        signal_length,
        sampling_freq,
        dopplers,
        codes_freq_domain,
        signal_baseband_arenas,
        signal_baseband_freq_domain_arenas,
        code_freq_baseband_freq_domain_arenas,
        code_baseband_arenas,
        signal_powers,
        forward_fft_plan,
        inverse_fft_plan,
        prns,
        compensate_doppler_code,
        noncoherent_rounds,
        n_threads,
        flip_correlation
    )
end

function Base.show(io::IO, ::MIME"text/plain", plan::AcquisitionPlanCPUMultithreaded)
    println(io,"Acquisition plan for GPS L1:\n")
    println(io,"Implementation: Multithreaded CPU")
    println(io, "Number of threads: $(plan.n_threads)")
    println(io,"Planned PRNs: $(plan.avail_prn_channels)")
    println(io,"Sample rate: $(plan.sampling_freq)")
    println(io,"Doppler taps: $(plan.dopplers)")
    println(io,"Coherent integration length: $(plan.signal_length) samples ($(upreferred(plan.signal_length/plan.sampling_freq)))")
    println(io,"Noncoherent integration round: $(plan.noncoherent_rounds) ($(upreferred(plan.signal_length*plan.noncoherent_rounds/plan.sampling_freq)))")
    println(io, "Doppler compensation: $(plan.compensate_doppler_code)")
    println(io,"FFTW plan: $(plan.forward_fft_plan)")
end

#= """
$(SIGNATURES)
This acquisition function uses a predefined acquisition plan to accelerate the computing time.
This will be useful, if you have to calculate the acquisition multiple time in a row.
""" =#
function acquire!(
    acq_plan::AcquisitionPlanCPUMultithreaded,
    signal,
    prns::AbstractVector{<:Integer};
    interm_freq = 0.0Hz,
    doppler_offsets = (0.0Hz for _ in prns),
    noise_powers = nothing,    
)

#= 
This is a rewrite of upstream acquire! 

Previously they are scattered in multiple source files, which makes it hard to follow.
For this implementation we merge the actual correlation code and the acquire! function

TODO don't use this until it passes regression test

=#


#=     all(map(prn -> prn in acq_plan.avail_prn_channels, prns)) ||
        throw(ArgumentError("You'll need to plan every PRN"))
 =#    code_period = get_code_length(acq_plan.system) / get_code_frequency(acq_plan.system)

    #Currently we parallelize over doppler taps 
    #It might be faster if we parallelize over PRNs, 
    #but currently the noncoherent integration loop is probably not thread safe

    #preallocate noncoherent integration workbuffer
    # we want a 3d array with dimensions of (1ms signal * doppler taps * prns)

    #= 
    read only shared variables:
    signal
    interm_freq
    sampling_freq
    fft_plan
    code_freq_domain

    read only thread local:
    doppler

    read/write buffers:
    signal_baseband
    =#
    begin 
        plan = acq_plan
        intermediate_freq = interm_freq
        noncoherent_rounds = plan.noncoherent_rounds
        compensate_doppler_code = plan.compensate_doppler_code
        signal_powers = plan.signal_powers
    end


    #n_chunk = length(plan.dopplers) ÷ plan.n_threads
    n_chunk = Int(ceil(length(plan.dopplers) / plan.n_threads))
#=     signal_baseband_arenas_chunk = Iterators.partition(plan.signal_baseband_arenas, n_chunk)
    signal_baseband_freq_domain_arenas_chunk = Iterators.partition(plan.signal_baseband_freq_domain_arenas, n_chunk)
    code_freq_baseband_freq_domain_arenas_chunk = Iterators.partition(plan.code_freq_baseband_freq_domain_arenas, n_chunk)
    code_baseband_arenas_chunk = Iterators.partition(cplan.ode_baseband_arenas, n_chun =#
    doppler_idx_chunk = Iterators.partition(1:length(plan.dopplers), n_chunk)
    doppler_chunk = Iterators.partition(plan.dopplers, n_chunk)
    signal_chunk = Iterators.partition(1:(noncoherent_rounds*plan.signal_length),plan.signal_length)

    for (chunk_idxs,round_idx) in zip(signal_chunk, 0:noncoherent_rounds-1)

        chunk = view(signal, chunk_idxs)

         begin

            @sync begin
                
                for (signal_baseband_arena, 
                    signal_baseband_freq_domain_arena, 
                    code_freq_baseband_freq_domain_arena, 
                    code_baseband_arena, doppler_idx_c, doppler_c) in zip(
                        plan.signal_baseband_arenas, 
                        plan.signal_baseband_freq_domain_arenas, 
                        plan.code_freq_baseband_freq_domain_arenas, 
                        plan.code_baseband_arenas, 
                        doppler_idx_chunk,
                        doppler_chunk)

                    Threads.@spawn begin
                        for (doppler_idx, doppler) in zip(doppler_idx_c, doppler_c)
                             begin
                                for (signal_power_each_prn,code_fd, doppler_offset) in zip(signal_powers,plan.codes_freq_domain,doppler_offsets)
                                    @no_escape signal_baseband_arena begin
                                        signal_baseband = @alloc(ComplexF32, plan.signal_length)
                                        downconvert!(signal_baseband, chunk, interm_freq + doppler + doppler_offset, plan.sampling_freq)
                                        @no_escape signal_baseband_freq_domain_arena begin
                                            signal_baseband_freq_domain = @alloc(ComplexF32,plan.signal_length)
                                            mul!(signal_baseband_freq_domain, plan.forward_fft_plan, signal_baseband)    
                                            @no_escape code_freq_baseband_freq_domain_arena begin
                                                code_freq_baseband_freq_domain = @alloc(ComplexF32, plan.signal_length)
                                                if plan.flip_correlation
                                                    @. code_freq_baseband_freq_domain = conj(code_fd) * signal_baseband_freq_domain
                                                else
                                                    @. code_freq_baseband_freq_domain = code_fd * conj(signal_baseband_freq_domain)
                                                end
                                                @no_escape code_baseband_arena begin
                                                    code_baseband = @alloc(ComplexF32,plan.signal_length)
                                                    mul!(code_baseband, plan.inverse_fft_plan, code_freq_baseband_freq_domain)
                                                    #TODO add another arena here #
                                                    if round_idx == 0 #For first noncoherent integration we overwrite the output buffer
                                                        signal_power_each_prn[:,doppler_idx] .= abs2.(code_baseband)
                                                    else
                                                        if plan.compensate_doppler_code != false
                                                            Ns = sign(ustrip(doppler) + ustrip(doppler_offset)) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(ustrip(doppler) + ustrip(doppler_offset))/ustrip(get_center_frequency(plan.system)), RoundNearest)
                                                            signal_power_each_prn[:,doppler_idx] .= signal_power_each_prn[:,doppler_idx] .+ abs2.(circshift(code_baseband,-Ns)) #TODO circshift is allocating
                                                        else
                                                            signal_power_each_prn[:,doppler_idx] .= signal_power_each_prn[:,doppler_idx] .+ abs2.(code_baseband)
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end

            end

        end
    end

    #Analyse the resulting power bins
    # For GNSS reflectometry application we can skip this, GNSSReflectometry.jl have their own postprocessing algos
    if isnothing(noise_powers)
        noise_powers = [nothing for _ in prns]
    end

   map(signal_powers, prns, doppler_offsets, noise_powers) do powers, prn, doppler_offset, noise_power
        signal_power, noise_power, code_index, doppler_index = est_signal_noise_power(
            powers,
            acq_plan.sampling_freq,
            get_code_frequency(acq_plan.system), 
            noise_power,
        )
        CN0 = 10 * log10(signal_power / noise_power / code_period / 1.0Hz)
        doppler =
            (doppler_index - 1) * step(acq_plan.dopplers) +
            first(acq_plan.dopplers) +
            doppler_offset
        code_phase =
            (code_index - 1) /
            (acq_plan.sampling_freq / get_code_frequency(acq_plan.system))
        AcquisitionResults(
            acq_plan.system,
            prn,
            acq_plan.sampling_freq,
            doppler,
            code_phase,
            CN0,
            noise_power,
            powers,
            (acq_plan.dopplers .+ doppler_offset) / 1.0Hz,
        )
    end 
end
