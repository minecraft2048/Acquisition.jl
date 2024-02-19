struct AcquisitionPlanCPU{S,DS,CS,P,PS} <: AbstractAcquisitionPlan
    system::S
    signal_length::Int
    sampling_freq::typeof(1.0Hz)
    dopplers::DS
    codes_freq_domain::CS
    signal_baseband::Vector{ComplexF32}
    signal_baseband_freq_domain::Vector{ComplexF32}
    code_freq_baseband_freq_domain::Vector{ComplexF32}
    code_baseband::Vector{ComplexF32}
    signal_powers::Vector{Matrix{Float32}}
    fft_plan::P
    avail_prn_channels::PS
    compensate_doppler_code::Bool
    noncoherent_rounds::Int
end

function AcquisitionPlanCPU(
    system,
    signal_length,
    sampling_freq;
    max_doppler = 7000Hz,
    dopplers = -max_doppler:1/3/(signal_length/sampling_freq):max_doppler,
    prns = 1:34,
    fft_flag = FFTW.MEASURE,
    compensate_doppler_code = false,
    noncoherent_rounds = 1,
    flip_correlation = false
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

    signal_baseband = Vector{ComplexF32}(undef, signal_length)
    signal_baseband_freq_domain = similar(signal_baseband)
    code_freq_baseband_freq_domain = similar(signal_baseband)
    signal_baseband_freq_domain = similar(signal_baseband)
    code_baseband = similar(signal_baseband)
    fft_plan = plan_fft(signal_baseband; flags = fft_flag)

    codes = [gen_code(signal_length, system, sat_prn, sampling_freq) for sat_prn in prns]
    codes_freq_domain = [fft_plan * code for code in codes]

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

    AcquisitionPlanCPU(
        system,
        signal_length,
        sampling_freq,
        dopplers,
        codes_freq_domain,
        signal_baseband,
        signal_baseband_freq_domain,
        code_freq_baseband_freq_domain,
        code_baseband,
        signal_powers,
        fft_plan,
        prns,
        compensate_doppler_code,
        noncoherent_rounds
    )
end

"""
$(SIGNATURES)
This acquisition function uses a predefined acquisition plan to accelerate the computing time.
This will be useful, if you have to calculate the acquisition multiple time in a row.
"""
function acquire!(
    acq_plan::AcquisitionPlanCPU,
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


     all(map(prn -> prn in acq_plan.avail_prn_channels, prns)) ||
        throw(ArgumentError("You'll need to plan every PRN"))
     code_period = get_code_length(acq_plan.system) / get_code_frequency(acq_plan.system)

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

    #= Loop count calculation 

        For:
        noncoherent rounds = 1000
        prn = 2
        doppler taps = 43

        Loop order 1:

        for each 1ms signal_chunk (1000)
            for each prns (2)
                for each doppler frequency (43)
                    signal_chunk = downconvert(signal_chunk, doppler) (1000*43*2 = 86000)
                    signal_freq_domain = fft(signal_chunk) (1000*43*2 = 86000)
                    correlation_freq_domain = signal_freq_domain * conj(code_freq_domain) (1000*43*2 = 86000)
                    correlation = ifft(correlation_freq_domain) (1000*43*2 = 86000)
                    power_array = power_array + doppler_compensation() (1000*43*2 = 86000)

        total ops = 5*86000 = 430000


        Loop order 2:
        
        for each 1ms signal_chunk (1000)
            for each doppler frequency (43)
                signal_chunk = downconvert(signal_chunk, doppler) (1000*43 = 43000)
                signal_freq_domain = fft(signal_chunk) (1000*43 = 43000)
                for each prns (2)
                    correlation_freq_domain = signal_freq_domain * conj(code_freq_domain) (1000*43*2 = 86000)
                    correlation = ifft(correlation_freq_domain) (1000*43*2 = 86000)
                    power_array = power_array + doppler_compensation() (1000*43*2 = 86000)

        total ops = 43000+43000+86000+86000+86000 = 344000 but this might have worse memory store locality

        Variables that depends on number of PRNs are:

        signal_powers
        codes_freq_domain

        Variables that depends on number of threads are:
        (currently none)
        
    =#
    #loop order 2

     for (chunk,round_idx) in zip(Iterators.partition(signal[1:(noncoherent_rounds*plan.signal_length)],plan.signal_length), 0:noncoherent_rounds-1)
         begin
            for (doppler_idx,doppler) in enumerate(plan.dopplers)
                 begin
                    for (signal_power_each_prn,code_fd, doppler_offset) in zip(signal_powers,plan.codes_freq_domain, doppler_offsets)
                         begin
                             downconvert!(plan.signal_baseband, chunk, interm_freq + doppler + doppler_offset , plan.sampling_freq) 
                             mul!(plan.signal_baseband_freq_domain, plan.fft_plan, plan.signal_baseband)        
                              @. plan.code_freq_baseband_freq_domain = code_fd * conj(plan.signal_baseband_freq_domain) 
                             ldiv!(plan.code_baseband, plan.fft_plan, plan.code_freq_baseband_freq_domain)
                             begin 
                                if round_idx == 0 #For first noncoherent integration we overwrite the output buffer
                                     signal_power_each_prn[:,doppler_idx] .= abs2.(plan.code_baseband)
                                else
                                    if plan.compensate_doppler_code != false
                                         Ns = sign(ustrip(doppler) - ustrip(doppler_offset)) * round(round_idx*0.001* ustrip(plan.sampling_freq) * abs(ustrip(doppler) - ustrip(doppler_offset))/ustrip(get_center_frequency(plan.system)), RoundNearest)
                                         tmp = circshift(plan.code_baseband,-Ns)
                                         tmp .= abs2.(tmp)
                                         signal_power_each_prn[:,doppler_idx] .= signal_power_each_prn[:,doppler_idx] .+ tmp
                                    else
                                        signal_power_each_prn[:,doppler_idx] .= signal_power_each_prn[:,doppler_idx] .+ abs2.(plan.code_baseband)
                                    end
                                end
                            end
                                #=                             
                                for i in eachindex(plan.code_baseband)

                                @inbounds signal_power_each_prn[i,doppler_idx] = signal_power_each_prn[i,doppler_idx] + abs2(plan.code_baseband[i])
                            end
 =#                     end
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

function Base.show(io::IO, ::MIME"text/plain", plan::AcquisitionPlanCPU)
    println(io,"Acquisition plan for GPS L1:\n")
    println(io,"Implementation: Single threaded CPU")
    println(io,"Planned PRNs: $(plan.avail_prn_channels)")
    println(io,"Sample rate: $(plan.sampling_freq)")
    println(io,"Doppler taps: $(plan.dopplers)")
    println(io,"Coherent integration length: $(plan.signal_length) samples ($(upreferred(plan.signal_length/plan.sampling_freq)))")
    println(io,"Noncoherent integration round: $(plan.noncoherent_rounds) ($(upreferred(plan.signal_length*plan.noncoherent_rounds/plan.sampling_freq)))")
    println(io, "Doppler compensation: $(plan.compensate_doppler_code)")
    println(io,"FFTW plan: $(plan.fft_plan)")
end
